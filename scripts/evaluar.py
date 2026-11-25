"""Evaluación por lotes de OWSM v3.1 base: tablas y figuras para la sección 6 del README.

Lee audios/referencias.csv (columnas: archivo, tipo, idioma, tarea, figura, referencia)
y los WAV de audios/. Genera en resultados/ (por defecto):

  p1_resultados.csv        una fila por audio: texto, WER, CER, tiempo, RTF, confianza
  p1_wer_por_frase.png     WER de cada frase
  p1_rtf_vs_duracion.png   tiempo de inferencia contra duración (¿es casi constante?)
  p1_diagonalidad.png      mapa 6x6 (capa x cabeza): qué tan diagonal es la atención
  atencion_<audio>.png     cross-attention promedio de los audios con figura=1
  cabezas_<audio>_capaN.png las 6 cabezas de una capa
  p2_ruido.csv/.png        WER contra nivel de ruido (opcional, --ruido)
  p3_beam.csv/.png         beam 1 contra beam 5 (opcional, --beams)
  resumen.json             cifras clave para citar en el README

Uso (desde la carpeta del repo):
    uv run --with jiwer --with num2words python scripts/evaluar.py --prueba
    uv run --with jiwer --with num2words python scripts/evaluar.py
    uv run --with jiwer --with num2words python scripts/evaluar.py --ruido 20 10 5 --beams 1 5

--prueba procesa solo el primer audio, para comprobar que todo funciona.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from num2words import num2words
except ImportError:  # sin num2words los números no se normalizan
    num2words = None


# ----------------------------------------------------------------- utilidades
def normalizar(texto: str) -> str:
    """Minúsculas, números a palabras y sin puntuación, para no contar formato como error."""
    t = texto.lower()
    if num2words is not None:
        t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="es") + " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalizar_en(texto: str) -> str:
    t = re.sub(r"[^\w\s]", " ", texto.lower())
    return re.sub(r"\s+", " ", t).strip()


def wer_cer(referencia: str, hipotesis: str, idioma: str):
    import jiwer

    norm = normalizar_en if idioma == "<eng>" else normalizar
    r, h = norm(referencia), norm(hipotesis)
    if not r:
        return None, None, 0
    return jiwer.wer(r, h), jiwer.cer(r, h), len(r.split())


def mezclar_ruido(audio: np.ndarray, snr_db: float, semilla: int = 0) -> np.ndarray:
    """Suma ruido blanco gaussiano con la relación señal/ruido pedida (en dB)."""
    rng = np.random.default_rng(semilla)
    ruido = rng.normal(0, 1, len(audio))
    rms_s = np.sqrt(np.mean(audio**2)) + 1e-9
    ruido *= rms_s / (10 ** (snr_db / 20)) / (np.sqrt(np.mean(ruido**2)) + 1e-9)
    return np.clip(audio + ruido, -1, 1).astype("float32")


def rangos(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def diagonalidad(atencion: np.ndarray) -> np.ndarray:
    """Para cada (capa, cabeza): correlación de Spearman entre el índice del token y el
    frame donde más atiende. Cerca de 1 = patrón diagonal (palabras en el orden del audio).
    atencion: (capas, cabezas, tokens, frames). Se excluye el último token (<eos>)."""
    capas, cabezas, n_tok, _ = atencion.shape
    salida = np.full((capas, cabezas), np.nan)
    if n_tok - 1 < 4:
        return salida
    idx = rangos(np.arange(n_tok - 1))
    for c in range(capas):
        for h in range(cabezas):
            pico = rangos(atencion[c, h, : n_tok - 1].argmax(axis=1).astype(float))
            if pico.std() > 0:
                salida[c, h] = np.corrcoef(idx, pico)[0, 1]
    return salida


def leer_manifiesto(carpeta: Path):
    with open(carpeta / "referencias.csv", encoding="utf-8") as f:
        return [fila for fila in csv.DictReader(f) if (carpeta / fila["archivo"]).exists()]


def guardar_csv(ruta: Path, filas: list[dict]):
    if not filas:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


# ------------------------------------------------------------------- etapas
def etapa_p1(modelo, filas, carpeta_audios, salida, figuras_dir):
    from owsm_demo.atencion import inferir_con_atencion
    from owsm_demo.graficos import figura_atencion, figura_cabezas
    from owsm_demo.modelo import cargar_audio

    resultados, diag_acum = [], []
    for i, fila in enumerate(filas, 1):
        audio = cargar_audio(carpeta_audios / fila["archivo"])
        t0 = time.perf_counter()
        r = inferir_con_atencion(modelo, audio, fila["idioma"], fila["tarea"])
        t = time.perf_counter() - t0
        ref = fila["referencia"]
        wer = cer = None
        n_pal = 0
        if fila["tipo"] in ("frase", "largo", "ingles"):
            wer, cer, n_pal = wer_cer(ref, r.texto, fila["idioma"])
        fila_res = {
            "archivo": fila["archivo"], "tipo": fila["tipo"],
            "idioma": fila["idioma"], "tarea": fila["tarea"],
            "duracion_s": round(r.duracion, 2), "tiempo_s": round(t, 2),
            "rtf": round(t / r.duracion, 2), "tokens": len(r.tokens_generados),
            "confianza_media": round(float(np.mean(r.probabilidades)), 3),
            "palabras_ref": n_pal,
            "wer": None if wer is None else round(wer, 3),
            "cer": None if cer is None else round(cer, 3),
            "referencia": ref, "salida": r.texto,
        }
        resultados.append(fila_res)
        print(f"[{i}/{len(filas)}] {fila['archivo']}: {t:.1f} s | WER={fila_res['wer']} | {r.texto!r}")
        if fila["tipo"] in ("frase", "largo", "ingles") and r.atencion.shape[2] > 1:
            diag_acum.append(diagonalidad(r.atencion))
        if fila.get("figura") == "1":
            base = Path(fila["archivo"]).stem
            fig = figura_atencion(r, None, None)
            fig.savefig(figuras_dir / f"atencion_{base}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            for capa in (0, 2, 5):
                fig = figura_cabezas(r, capa)
                fig.savefig(figuras_dir / f"cabezas_{base}_capa{capa + 1}.png", dpi=130,
                            bbox_inches="tight")
                plt.close(fig)
        guardar_csv(salida / "p1_resultados.csv", resultados)  # guarda al avanzar

    # --- figuras de P1
    frases = [x for x in resultados if x["tipo"] == "frase" and x["wer"] is not None]
    if frases:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([x["archivo"].replace(".wav", "") for x in frases], [x["wer"] * 100 for x in frases],
               color="#6A9BC3")
        ax.set_ylabel("WER (%)")
        ax.set_title("WER por frase (español, voz propia)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(salida / "p1_wer_por_frase.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([x["duracion_s"] for x in resultados], [x["tiempo_s"] for x in resultados],
               color="#9A86B8")
    for x in resultados:
        if x["tipo"] in ("largo", "silencio", "ingles"):
            ax.annotate(x["archivo"].replace(".wav", ""), (x["duracion_s"], x["tiempo_s"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Duración del audio (s)")
    ax.set_ylabel("Tiempo de inferencia (s)")
    ax.set_ylim(bottom=0)
    ax.set_title("Tiempo de inferencia vs duración (CPU, beam 5, con extracción de atención)")
    fig.tight_layout()
    fig.savefig(salida / "p1_rtf_vs_duracion.png", dpi=150)
    plt.close(fig)

    resumen = {}
    if diag_acum:
        d = np.nanmean(np.stack(diag_acum), axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(d, cmap="viridis", vmin=-0.2, vmax=1, aspect="auto")
        for c in range(d.shape[0]):
            for h in range(d.shape[1]):
                ax.text(h, c, f"{d[c, h]:.2f}", ha="center", va="center",
                        color="white" if d[c, h] < 0.6 else "black", fontsize=9)
        ax.set_xticks(range(d.shape[1]))
        ax.set_xticklabels([f"{h + 1}" for h in range(d.shape[1])])
        ax.set_yticks(range(d.shape[0]))
        ax.set_yticklabels([f"{c + 1}" for c in range(d.shape[0])])
        ax.set_xlabel("Cabeza")
        ax.set_ylabel("Capa del decoder")
        ax.set_title("Diagonalidad de la cross-attention\n(1 = sigue el orden del audio)")
        fig.colorbar(im, ax=ax, label="correlación de Spearman")
        fig.tight_layout()
        fig.savefig(salida / "p1_diagonalidad.png", dpi=150)
        plt.close(fig)
        c, h = np.unravel_index(np.nanargmax(d), d.shape)
        resumen["mejor_cabeza_diagonal"] = {"capa": int(c + 1), "cabeza": int(h + 1),
                                            "valor": round(float(d[c, h]), 3)}
        resumen["diagonalidad_media"] = round(float(np.nanmean(d)), 3)
        np.save(salida / "p1_diagonalidad.npy", d)

    if frases:
        palabras = sum(x["palabras_ref"] for x in frases)
        err = sum(x["wer"] * x["palabras_ref"] for x in frases)
        resumen["wer_frases_global_%"] = round(100 * err / palabras, 1)
        resumen["wer_frases_n_palabras"] = palabras
    reales = [x for x in resultados if x["tipo"] in ("frase", "largo")]
    if reales:
        resumen["tiempo_medio_s"] = round(float(np.mean([x["tiempo_s"] for x in reales])), 1)
        resumen["rtf_medio"] = round(float(np.mean([x["rtf"] for x in reales])), 2)
    return resultados, resumen


def etapa_ruido(modelo, filas, carpeta_audios, salida, niveles):
    from owsm_demo.modelo import cargar_audio, transcribir

    frases = [f for f in filas if f["tipo"] == "frase"]
    filas_res = []
    for snr in [None] + list(niveles):
        errs = pal = 0
        for fila in frases:
            audio = cargar_audio(carpeta_audios / fila["archivo"])
            if snr is not None:
                audio = mezclar_ruido(audio, snr)
            texto = transcribir(modelo, audio, fila["idioma"], fila["tarea"])["texto_sin_especiales"]
            wer, _, n = wer_cer(fila["referencia"], texto, fila["idioma"])
            errs += wer * n
            pal += n
        w = 100 * errs / max(pal, 1)
        filas_res.append({"snr_db": "sin ruido" if snr is None else snr, "wer_%": round(w, 1)})
        print(f"ruido {snr}: WER {w:.1f} %")
        guardar_csv(salida / "p2_ruido.csv", filas_res)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([str(x["snr_db"]) for x in filas_res], [x["wer_%"] for x in filas_res], "o-",
            color="#D49A5A")
    ax.set_xlabel("Relación señal/ruido (dB; menor = más ruido)")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER de las 10 frases al añadir ruido blanco")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(salida / "p2_ruido.png", dpi=150)
    plt.close(fig)
    return filas_res


def etapa_beam(filas, carpeta_audios, salida, beams):
    from owsm_demo.modelo import cargar_audio, cargar_modelo, transcribir

    frases = [f for f in filas if f["tipo"] == "frase"]
    filas_res = []
    for b in beams:
        m = cargar_modelo(beam_size=b)
        transcribir(m, np.zeros(16000 * 3, dtype="float32"), "<spa>", "<asr>")  # calentamiento
        errs = pal = tiempo = 0
        for fila in frases:
            audio = cargar_audio(carpeta_audios / fila["archivo"])
            t0 = time.perf_counter()
            texto = transcribir(m, audio, fila["idioma"], fila["tarea"])["texto_sin_especiales"]
            tiempo += time.perf_counter() - t0
            wer, _, n = wer_cer(fila["referencia"], texto, fila["idioma"])
            errs += wer * n
            pal += n
        filas_res.append({"beam": b, "wer_%": round(100 * errs / max(pal, 1), 1),
                          "tiempo_medio_s": round(tiempo / len(frases), 1)})
        print(filas_res[-1])
        guardar_csv(salida / "p3_beam.csv", filas_res)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"beam {x['beam']}" for x in filas_res], [x["tiempo_medio_s"] for x in filas_res],
           color="#7FA866")
    for i, x in enumerate(filas_res):
        ax.text(i, x["tiempo_medio_s"], f"WER {x['wer_%']} %", ha="center", va="bottom")
    ax.set_ylabel("Tiempo medio por frase (s)")
    ax.set_title("Beam search: tiempo y WER")
    fig.tight_layout()
    fig.savefig(salida / "p3_beam.png", dpi=150)
    plt.close(fig)
    return filas_res


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audios", default="audios")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--prueba", action="store_true", help="solo el primer audio")
    ap.add_argument("--ruido", type=float, nargs="*", default=None, help="SNR en dB, ej. 20 10 5")
    ap.add_argument("--beams", type=int, nargs="*", default=None, help="ej. 1 5")
    a = ap.parse_args()

    carpeta = Path(a.audios)
    salida = Path(a.salida)
    salida.mkdir(exist_ok=True)
    filas = leer_manifiesto(carpeta)
    if not filas:
        raise SystemExit(f"No hay audios en {carpeta}/ que coincidan con referencias.csv")
    if a.prueba:
        filas = filas[:1]

    from owsm_demo.modelo import cargar_modelo, transcribir

    print("Cargando el modelo (la primera vez descarga ≈400 MB)...")
    modelo = cargar_modelo(beam_size=5)
    transcribir(modelo, np.zeros(16000 * 3, dtype="float32"), "<spa>", "<asr>")  # calentamiento

    resultados, resumen = etapa_p1(modelo, filas, carpeta, salida, salida)
    if a.ruido and not a.prueba:
        resumen["ruido"] = etapa_ruido(modelo, filas, carpeta, salida, a.ruido)
    if a.beams and not a.prueba:
        resumen["beam"] = etapa_beam(filas, carpeta, salida, a.beams)
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print("\nResumen:", json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nListo. Resultados en {salida}/")


if __name__ == "__main__":
    main()
