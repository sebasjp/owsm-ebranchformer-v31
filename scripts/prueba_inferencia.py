"""Prueba mínima de inferencia (fase 2): carga el modelo, transcribe un audio
y mide tiempos y memoria.

Uso:
    uv run python scripts/prueba_inferencia.py ruta/al/audio.wav [--idioma <spa>] [--tarea <asr>]
"""

import argparse
import sys
import time

import psutil

from owsm_demo.modelo import MODELO_HF, cargar_audio, cargar_modelo, dispositivo, transcribir


def ram_mb() -> float:
    """Memoria RAM usada por este proceso, en MB."""
    return psutil.Process().memory_info().rss / 1024**2


def main():
    # La consola de Windows (cp1252) no imprime el carácter "▁" que usa el BPE
    # para marcar inicio de palabra; forzamos UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--modelo", default=MODELO_HF)
    p.add_argument("--idioma", default="<spa>")
    p.add_argument("--tarea", default="<asr>")
    args = p.parse_args()

    print(f"Dispositivo: {dispositivo()} | RAM inicial: {ram_mb():.0f} MB")

    t0 = time.perf_counter()
    modelo = cargar_modelo(args.modelo, idioma=args.idioma, tarea=args.tarea)
    n_params = sum(p.numel() for p in modelo.s2t_model.parameters())
    print(f"Modelo {args.modelo}: {n_params / 1e6:.1f}M parámetros, "
          f"cargado en {time.perf_counter() - t0:.1f} s | RAM: {ram_mb():.0f} MB")

    audio = cargar_audio(args.audio)
    print(f"Audio: {len(audio) / 16000:.1f} s")

    t0 = time.perf_counter()
    salida = transcribir(modelo, audio)
    print(f"Inferencia: {time.perf_counter() - t0:.1f} s | RAM: {ram_mb():.0f} MB")
    print("Tokens:", salida["tokens"])
    print("Texto:", salida["texto"])


if __name__ == "__main__":
    main()
