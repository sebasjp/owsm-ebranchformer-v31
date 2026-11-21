"""Extracción de la cross-attention (decoder → encoder) de OWSM v3.1.

Idea:
  1. Se decodifica normalmente (beam search) para obtener la secuencia de tokens.
  2. Se vuelve a pasar esa secuencia completa por el decoder ("teacher forcing"),
     junto con la salida del encoder, y se leen los pesos de atención que cada
     capa de cross-attention guarda en `src_attn.attn`.

En la cross-attention de cada capa del decoder:
  Q = W_Q · (estado oculto del decoder en cada token)    -> (L_tokens, d)
  K = W_K · (salida del encoder en cada frame de audio)  -> (T_frames, d)
  V = W_V · (salida del encoder en cada frame de audio)  -> (T_frames, d)
  pesos = softmax(Q·Kᵀ / √d_k)                           -> (L_tokens, T_frames)
La fila i de `pesos` indica a qué parte del audio miró el decoder para generar
el token i+1: es una alineación suave texto ↔ audio.
"""

from dataclasses import dataclass

import numpy as np
import torch
from espnet2.bin.s2t_inference import Speech2Text

from owsm_demo.modelo import FRECUENCIA_MUESTREO


@dataclass
class ResultadoAtencion:
    texto: str                 # texto final (sin tokens especiales)
    texto_crudo: str           # texto con tokens especiales (p. ej. timestamps)
    prompt: list[str]          # tokens de control: <sos> <spa> <asr> <notimestamps>
    tokens_generados: list[str]  # tokens que generó el modelo (incluye <eos>)
    probabilidades: np.ndarray   # prob. que el modelo asignó a cada token generado
    atencion: np.ndarray       # (capas, cabezas, tokens_generados, frames_audio)
    features: np.ndarray       # log-Mel de entrada al encoder (frames_10ms, n_mels)
    seg_por_frame_enc: float   # segundos que representa cada frame del encoder
    duracion: float            # duración real del audio (s)


@torch.no_grad()
def inferir_con_atencion(
    modelo: Speech2Text,
    audio: np.ndarray,
    idioma: str,
    tarea: str,
    timestamps: bool = False,
) -> ResultadoAtencion:
    s2t = modelo.s2t_model
    duracion = len(audio) / FRECUENCIA_MUESTREO

    # ---- 1. Decodificación normal (beam search) ---------------------------------
    texto_crudo, _, _, texto, hipotesis = modelo(
        audio, lang_sym=idioma, task_sym=tarea, predict_time=timestamps
    )[0]
    yseq = hipotesis.yseq.tolist()  # [<sos>, <idioma>, <tarea>, (<notimestamps>), ..., <eos>]
    n_prompt = len(modelo._build_hyp_primer(idioma, tarea, timestamps))

    # ---- 2. Encoder: audio (relleno a 30 s) -> log-Mel -> E-Branchformer -> H ------
    voz = modelo._pad_or_trim(torch.tensor(audio)).unsqueeze(0).to(modelo.device)
    largos = torch.tensor([voz.size(1)], device=modelo.device)
    features, _ = s2t.frontend(voz, largos)          # (1, 3000, n_mels): un frame cada 10 ms
    enc, enc_largos = s2t.encode(voz, largos)        # (1, T_enc, 384): memoria acústica H
    if isinstance(enc, tuple):                       # algunos encoders devuelven salidas intermedias
        enc = enc[0]
    seg_por_frame_enc = (voz.size(1) / FRECUENCIA_MUESTREO) / enc.size(1)

    # ---- 3. Decoder con teacher forcing, leyendo src_attn.attn de cada capa --------
    # Se fuerza la implementación "manual" de la atención (sin SDPA/flash), que es
    # la única que guarda los pesos softmax(QKᵀ/√d) en `.attn`.
    capas = s2t.decoder.decoders
    flags_originales = [
        (getattr(c.src_attn, "use_sdpa", False), getattr(c.src_attn, "use_flash_attn", False))
        for c in capas
    ]
    for c in capas:
        c.src_attn.use_sdpa = False
        c.src_attn.use_flash_attn = False
    try:
        entrada = torch.tensor([yseq[:-1]], device=modelo.device)  # todo menos <eos>
        logits, _ = s2t.decoder(enc, enc_largos, entrada, torch.tensor([entrada.size(1)]))
        # attn de cada capa: (1, cabezas, L, T_enc) -> apilado: (capas, cabezas, L, T_enc)
        atencion = torch.stack([c.src_attn.attn[0] for c in capas]).float().cpu().numpy()
    finally:
        for c, (sdpa, flash) in zip(capas, flags_originales):
            c.src_attn.use_sdpa = sdpa
            c.src_attn.use_flash_attn = flash

    # Probabilidad que el decoder asignó a cada token realmente generado.
    probs = torch.softmax(logits[0].float(), dim=-1)                 # (L, vocab)
    siguientes = torch.tensor(yseq[1:], device=probs.device)
    prob_tokens = probs[torch.arange(len(siguientes)), siguientes].cpu().numpy()

    # La fila i de la atención corresponde a la PREDICCIÓN del token yseq[i+1].
    # Nos quedamos con las filas de los tokens generados por el modelo (después del
    # prompt) y con los frames que contienen audio real (el resto es relleno).
    desde = n_prompt - 1
    n_frames_audio = min(enc.size(1), int(np.ceil(duracion / seg_por_frame_enc)))
    atencion = atencion[:, :, desde:, :n_frames_audio]

    ids2tok = modelo.converter.ids2tokens
    n_feat_audio = min(features.size(1), int(np.ceil(duracion * 100)))  # 100 frames/s

    return ResultadoAtencion(
        texto=texto,
        texto_crudo=texto_crudo,
        prompt=ids2tok(yseq[:n_prompt]),
        tokens_generados=ids2tok(yseq[n_prompt:]),
        probabilidades=prob_tokens[desde:],
        atencion=atencion,
        features=features[0, :n_feat_audio].float().cpu().numpy(),
        seg_por_frame_enc=seg_por_frame_enc,
        duracion=duracion,
    )
