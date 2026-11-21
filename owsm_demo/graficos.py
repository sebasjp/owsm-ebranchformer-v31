"""Gráficos: log-Mel de entrada + mapa de calor de la cross-attention."""

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (para Streamlit / servidores)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from owsm_demo.atencion import ResultadoAtencion  # noqa: E402


def seleccionar_atencion(r: ResultadoAtencion, capa: int | None, cabeza: int | None) -> np.ndarray:
    """Devuelve la matriz (tokens, frames) de una capa/cabeza, o su promedio si es None."""
    a = r.atencion  # (capas, cabezas, tokens, frames)
    a = a.mean(axis=0) if capa is None else a[capa]
    a = a.mean(axis=0) if cabeza is None else a[cabeza]
    return a


def figura_atencion(r: ResultadoAtencion, capa: int | None, cabeza: int | None):
    """Log-Mel (arriba) y cross-attention tokens × tiempo (abajo), con el mismo eje de tiempo."""
    matriz = seleccionar_atencion(r, capa, cabeza)
    n_tokens, n_frames = matriz.shape
    fin_enc = n_frames * r.seg_por_frame_enc  # segundos cubiertos por los frames mostrados

    alto = 2.2 + 0.28 * n_tokens
    fig, (ax_mel, ax_att) = plt.subplots(
        2, 1, figsize=(12, alto), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.28 * n_tokens]},
    )

    # Entrada del encoder: log-Mel (80 bandas, un frame cada 10 ms)
    ax_mel.imshow(
        r.features.T, origin="lower", aspect="auto", cmap="magma",
        extent=[0, r.features.shape[0] / 100, 0, r.features.shape[1]],
    )
    ax_mel.set_ylabel("Banda Mel")
    ax_mel.set_title("Entrada del encoder: espectrograma log-Mel (K y V se calculan a partir de su codificación)")

    # Cross-attention: filas = tokens generados (de las Q), columnas = frames del encoder (K)
    im = ax_att.imshow(
        matriz, aspect="auto", cmap="viridis", interpolation="nearest",
        extent=[0, fin_enc, n_tokens - 0.5, -0.5],
    )
    ax_att.set_yticks(range(n_tokens))
    ax_att.set_yticklabels(r.tokens_generados, fontsize=9)
    ax_att.set_xlabel("Tiempo del audio (s) — cada columna es un frame del encoder de "
                      f"{r.seg_por_frame_enc * 1000:.0f} ms")
    ax_att.set_ylabel("Token generado (Q)")
    nombre_capa = "promedio de capas" if capa is None else f"capa {capa + 1}"
    nombre_cab = "promedio de cabezas" if cabeza is None else f"cabeza {cabeza + 1}"
    ax_att.set_title(f"Cross-attention softmax(QKᵀ/√d) — {nombre_capa}, {nombre_cab}")
    fig.colorbar(im, ax=[ax_mel, ax_att], fraction=0.02, pad=0.01, label="peso de atención")
    return fig


def figura_cabezas(r: ResultadoAtencion, capa: int):
    """Las 6 cabezas de una capa lado a lado, para comparar su especialización."""
    n_cabezas = r.atencion.shape[1]
    n_tokens, n_frames = r.atencion.shape[2:]
    fig, ejes = plt.subplots(1, n_cabezas, figsize=(3 * n_cabezas, 0.2 * n_tokens + 1.5), sharey=True)
    for h, ax in enumerate(ejes):
        ax.imshow(r.atencion[capa, h], aspect="auto", cmap="viridis", interpolation="nearest",
                  extent=[0, n_frames * r.seg_por_frame_enc, n_tokens - 0.5, -0.5])
        ax.set_title(f"cabeza {h + 1}", fontsize=10)
        ax.set_xlabel("s")
    ejes[0].set_yticks(range(n_tokens))
    ejes[0].set_yticklabels(r.tokens_generados, fontsize=7)
    fig.suptitle(f"Capa {capa + 1}: cada cabeza aprende un patrón de atención distinto")
    fig.tight_layout()
    return fig
