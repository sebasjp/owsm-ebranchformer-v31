"""Gráficas adicionales a partir de resultados/p1_resultados.csv (no vuelve a correr el modelo).

Uso:  uv run --with matplotlib --with numpy python scripts/graficar_resultados.py
Genera en resultados/:
  p1_tiempo_vs_tokens.png  tiempo de inferencia contra tokens generados
  p1_rtf_sin_silencio.png  RTF por audio (el silencio se muestra aparte)
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados")
filas = list(csv.DictReader(open(carpeta / "p1_resultados.csv", encoding="utf-8")))
for f in filas:
    for k in ("duracion_s", "tiempo_s", "rtf", "tokens"):
        f[k] = float(f[k])

normales = [f for f in filas if f["tipo"] != "silencio"]
sil = [f for f in filas if f["tipo"] == "silencio"]

# --- tiempo contra tokens generados
x = np.array([f["tokens"] for f in normales])
y = np.array([f["tiempo_s"] for f in normales])
pend, base = np.polyfit(x, y, 1)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(x, y, color="#6A9BC3", label="audios con voz")
xs = np.linspace(0, x.max() + 5, 50)
ax.plot(xs, pend * xs + base, "--", color="#555",
        label=f"ajuste: {base:.1f} s + {pend:.2f} s por token")
for f in normales:
    if f["tipo"] in ("largo", "traduccion", "ingles"):
        ax.annotate(f["archivo"].replace(".wav", ""), (f["tokens"], f["tiempo_s"]),
                    fontsize=8, xytext=(4, -10), textcoords="offset points")
ax.set_xlabel("Tokens generados")
ax.set_ylabel("Tiempo de inferencia (s)")
ax.set_ylim(bottom=0)
ax.set_title("El tiempo crece con los tokens que escribe el decoder\n(CPU, beam 5, con extracción de atención)")
ax.legend()
fig.tight_layout()
fig.savefig(carpeta / "p1_tiempo_vs_tokens.png", dpi=150)
plt.close(fig)
print(f"Ajuste lineal: tiempo ≈ {base:.2f} s + {pend:.3f} s × tokens (n={len(x)})")
r = np.corrcoef(x, y)[0, 1]
print(f"Correlación tokens-tiempo: {r:.3f}")
d = np.array([f["duracion_s"] for f in normales if f["archivo"] != "muylarga.wav"])
t = np.array([f["tiempo_s"] for f in normales if f["archivo"] != "muylarga.wav"])
print(f"Correlación duración-tiempo (sin muylarga): {np.corrcoef(d, t)[0, 1]:.3f}")

# --- RTF por audio, silencio aparte
fig, ax = plt.subplots(figsize=(9, 4))
nombres = [f["archivo"].replace(".wav", "") for f in normales]
ax.bar(nombres, [f["rtf"] for f in normales], color="#9A86B8")
ax.axhline(1, color="#C0392B", ls="--", lw=1)
ax.text(len(nombres) - 0.5, 1.02, "RTF = 1 (tiempo real)", ha="right", fontsize=8, color="#C0392B")
ax.set_ylabel("RTF = tiempo / duración del audio")
ax.set_title("RTF por audio (CPU, beam 5)"
             + (f"\nSilencio no se muestra: RTF {sil[0]['rtf']:.1f} ({sil[0]['tiempo_s']:.0f} s)" if sil else ""))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
fig.savefig(carpeta / "p1_rtf_sin_silencio.png", dpi=150)
plt.close(fig)

# --- WER de muylarga contra la parte que el modelo realmente vio (aprox.)
mu = [f for f in filas if f["archivo"] == "muylarga.wav"]
if mu:
    try:
        import jiwer
        import re
        m = mu[0]
        ref = re.sub(r"[^\w\s]", " ", m["referencia"].lower()).split()
        frac = min(1.0, 30.0 / m["duracion_s"])
        k = round(len(ref) * frac)
        hip = re.sub(r"[^\w\s]", " ", m["salida"].lower())
        w = jiwer.wer(" ".join(ref[:k]), hip)
        print(f"muylarga: se procesaron ~30 de {m['duracion_s']:.0f} s -> referencia recortada a las "
              f"primeras {k} de {len(ref)} palabras (suponiendo ritmo constante): WER ≈ {w:.3f}")
    except ImportError:
        print("(instala jiwer para calcular el WER de muylarga sobre la parte procesada)")

# --- libro contra frases cotidianas (WER global por grupo, con IC95 bootstrap por frase)
import random

def _grupo(f):
    a = f["archivo"]
    if a.startswith(("corta", "media")):
        return "libro"
    if a.startswith("diaria"):
        return "cotidiano"
    return None

def _wer(rows):
    e = sum(float(f["wer"]) * int(f["palabras_ref"]) for f in rows)
    p = sum(int(f["palabras_ref"]) for f in rows)
    return 100 * e / p

grupos = {"libro": [], "cotidiano": []}
for f in filas:
    g = _grupo(f)
    if g:
        grupos[g].append(f)
if grupos["cotidiano"]:
    random.seed(1)
    est = {}
    for g, rows in grupos.items():
        boot = sorted(_wer([random.choice(rows) for _ in rows]) for _ in range(5000))
        est[g] = (_wer(rows), boot[125], boot[4875], sum(int(f["palabras_ref"]) for f in rows))
        print(f"{g}: WER {est[g][0]:.1f} % (IC95 {est[g][1]:.1f}-{est[g][2]:.1f}), {est[g][3]} palabras")
    dif = sorted(_wer([random.choice(grupos['libro']) for _ in grupos['libro']])
                 - _wer([random.choice(grupos['cotidiano']) for _ in grupos['cotidiano']]) for _ in range(5000))
    print(f"Diferencia libro - cotidiano: {est['libro'][0]-est['cotidiano'][0]:.1f} puntos (IC95 {dif[125]:.1f} a {dif[4875]:.1f})")
    fig, ax = plt.subplots(figsize=(6, 4))
    xs_ = list(est)
    vals = [est[g][0] for g in xs_]
    err = [[est[g][0] - est[g][1] for g in xs_], [est[g][2] - est[g][0] for g in xs_]]
    ax.bar(xs_, vals, yerr=err, capsize=6, color=["#C98B6B", "#6A9BC3"])
    ax.axhline(18.5, color="#555", ls="--", lw=1)
    ax.text(1.45, 19.2, "artículo: 18.5 %", ha="right", fontsize=8, color="#555")
    for i, g in enumerate(xs_):
        ax.text(i, 1.5, f"n = {est[g][3]} palabras", ha="center", fontsize=8, color="white")
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, 40)
    ax.set_title("WER: frases del libro vs frases cotidianas\n(barras de error: IC 95 % por remuestreo de frases)")
    fig.tight_layout()
    fig.savefig(carpeta / "p1_libro_vs_cotidiano.png", dpi=150)
    plt.close(fig)
