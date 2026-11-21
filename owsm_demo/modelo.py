"""Carga del modelo OWSM v3.1 e inferencia (ASR / traducción).

El modelo se descarga desde Hugging Face Hub (repositorio oficial de ESPnet)
la primera vez y queda en caché para las siguientes ejecuciones.
"""

import os
import sys
from pathlib import Path

# Caché de Hugging Face (debe definirse ANTES de importar espnet / huggingface_hub,
# que leen HF_HOME al importarse). Si el usuario ya definió HF_HOME, se respeta.
# En Windows se usa una ruta CORTA en la raíz de la unidad del proyecto (p. ej. D:\hf_cache):
# los archivos del modelo tienen rutas muy largas y superan el límite de 260 caracteres
# de Windows si el caché vive dentro de la carpeta del proyecto.
# En Linux/Colab no se toca: se usa el caché por defecto (~/.cache/huggingface).
if sys.platform == "win32":
    unidad = Path(__file__).resolve().anchor  # p. ej. "D:\\"
    os.environ.setdefault("HF_HOME", str(Path(unidad) / "hf_cache"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from espnet2.bin.s2t_inference import Speech2Text  # noqa: E402

# Modelo usado: OWSM v3.1 base (101M parámetros, encoder E-Branchformer).
# Existen también small (367M) y medium (1.02B), pero no caben bien en un PC sin GPU.
MODELO_HF = "espnet/owsm_v3.1_ebf_base"

FRECUENCIA_MUESTREO = 16_000  # OWSM espera audio mono a 16 kHz


def dispositivo() -> str:
    """GPU si está disponible (p. ej. Colab), si no CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def cargar_modelo(
    nombre_hf: str = MODELO_HF,
    idioma: str = "<spa>",
    tarea: str = "<asr>",
    beam_size: int = 5,
) -> Speech2Text:
    """Descarga (si hace falta) y construye el objeto de inferencia de ESPnet.

    ctc_weight=0.0: se decodifica solo con el decoder (atención), que es el
    camino que usa la cross-attention que queremos visualizar.
    """
    return Speech2Text.from_pretrained(
        nombre_hf,
        device=dispositivo(),
        beam_size=beam_size,
        ctc_weight=0.0,
        maxlenratio=0.0,
        lang_sym=idioma,
        task_sym=tarea,
        predict_time=False,
    )


def cargar_audio(ruta_o_archivo) -> np.ndarray:
    """Lee un audio (wav/mp3/flac...) y lo convierte a mono 16 kHz float32."""
    audio, _ = librosa.load(ruta_o_archivo, sr=FRECUENCIA_MUESTREO, mono=True)
    return audio.astype(np.float32)


def transcribir(
    modelo: Speech2Text,
    audio: np.ndarray,
    idioma: str | None = None,
    tarea: str | None = None,
    timestamps: bool = False,
) -> dict:
    """Ejecuta la inferencia sobre un segmento de audio (<= 30 s).

    ESPnet rellena/recorta internamente el audio a 30 s, igual que en el
    entrenamiento. Devuelve el texto y los tokens generados.
    """
    resultados = modelo(audio, lang_sym=idioma, task_sym=tarea, predict_time=timestamps)
    texto, tokens, ids_tokens, texto_limpio, _hipotesis = resultados[0]
    return {
        "texto": texto,
        "texto_sin_especiales": texto_limpio,
        "tokens": tokens,
        "ids_tokens": ids_tokens,
    }
