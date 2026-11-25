"""Graba los audios de la prueba, uno por uno, leyendo audios/referencias.csv.

Uso (desde la carpeta del repo):
    uv run --with sounddevice --with soundfile python scripts/grabar.py --dispositivos
    uv run --with sounddevice --with soundfile python scripts/grabar.py --test
    uv run --with sounddevice --with soundfile python scripts/grabar.py
Opciones: --dispositivo N  (micrófono a usar), --solo frase03.wav  (repetir uno),
          --rehacer (volver a grabar los que ya existen).

Guarda WAV mono de 16 kHz en audios/. Mientras grabas se muestra el texto a leer.
Se graba hasta que presiones Enter.
"""

import argparse
import csv
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16_000


def grabar(dispositivo):
    """Graba hasta que el usuario presione Enter. Devuelve audio float32 mono a 16 kHz."""
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, tiempo, estado):
        q.put(indata.copy())

    try:
        sr_real = SR
        flujo = sd.InputStream(samplerate=sr_real, channels=1, dtype="float32",
                               device=dispositivo, callback=cb)
    except sd.PortAudioError:
        sr_real = int(sd.query_devices(dispositivo, "input")["default_samplerate"])
        flujo = sd.InputStream(samplerate=sr_real, channels=1, dtype="float32",
                               device=dispositivo, callback=cb)
    with flujo:
        input("   ● grabando... presiona Enter para detener ")
    partes = []
    while not q.empty():
        partes.append(q.get())
    audio = np.concatenate(partes)[:, 0] if partes else np.zeros(0, dtype="float32")
    if sr_real != SR and len(audio):
        n = int(len(audio) * SR / sr_real)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype("float32")


def nivel(audio):
    if len(audio) == 0:
        return 0.0, -120.0
    pico = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
    return pico, 20 * np.log10(rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispositivos", action="store_true", help="lista los micrófonos")
    ap.add_argument("--dispositivo", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="prueba rápida del micrófono")
    ap.add_argument("--solo", default=None)
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--audios", default="audios")
    a = ap.parse_args()

    if a.dispositivos:
        print(sd.query_devices())
        print("\nPredeterminado de entrada:", sd.default.device[0])
        return

    if a.test:
        print("Di algo como 'uno, dos, tres, probando el micrófono'.")
        audio = grabar(a.dispositivo)
        pico, rms = nivel(audio)
        print(f"Duración {len(audio)/SR:.1f} s | pico {pico:.2f} | nivel {rms:.0f} dBFS")
        if pico < 0.05:
            print("⚠ Casi no se oye: sube el volumen de entrada o elige otro micrófono.")
        elif pico > 0.98:
            print("⚠ Se satura: baja el volumen de entrada o aléjate del micrófono.")
        else:
            print("✔ Nivel correcto.")
        return

    carpeta = Path(a.audios)
    carpeta.mkdir(exist_ok=True)
    with open(carpeta / "referencias.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        if a.solo and fila["archivo"] != a.solo:
            continue
        destino = carpeta / fila["archivo"]
        if destino.exists() and not a.rehacer and not a.solo:
            print(f"= {fila['archivo']} ya existe, se omite")
            continue
        print(f"\n[{fila['archivo']}]  tipo: {fila['tipo']}")
        if fila["tipo"] == "silencio":
            print("   No hables: deja el micrófono abierto unos 10 s en silencio.")
        else:
            if fila["tipo"] == "largo":
                print("   Lee TODO seguido, a ritmo natural, sin pausas largas:")
            else:
                print("   Lee:")
            print("   " + (fila.get("leer") or fila["referencia"]))
        while True:
            input("   Enter para empezar a grabar ")
            audio = grabar(a.dispositivo)
            pico, rms = nivel(audio)
            print(f"   duración {len(audio)/SR:.1f} s | pico {pico:.2f} | nivel {rms:.0f} dBFS")
            if fila["tipo"] != "silencio" and pico < 0.05:
                print("   ⚠ Muy bajo. Repite.")
                continue
            repetir = False
            while True:
                op = input("   [Enter = guardar | p = escuchar | r = repetir] ").strip().lower()
                if op == "p":
                    sd.play(audio, SR)
                    sd.wait()
                elif op == "r":
                    repetir = True
                    break
                else:
                    break
            if repetir:
                continue
            sf.write(destino, audio, SR)
            print(f"   guardado en {destino}")
            break


if __name__ == "__main__":
    main()
