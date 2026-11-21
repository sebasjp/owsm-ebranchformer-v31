"""Interfaz Streamlit para inferencia con OWSM v3.1 (E-Branchformer).

Ejecutar:
    uv run streamlit run app.py
"""

import time
from datetime import datetime

import streamlit as st

from owsm_demo.modelo import (
    FRECUENCIA_MUESTREO,
    MODELO_HF,
    cargar_audio,
    cargar_modelo,
    dispositivo,
)
from owsm_demo.atencion import inferir_con_atencion
from owsm_demo.graficos import figura_atencion, figura_cabezas

DURACION_MAX_S = 30  # OWSM procesa ventanas fijas de 30 s

# Idiomas más comunes primero; el modelo soporta 151 (ver token_list del config.yaml).
IDIOMAS = {
    "Español": "<spa>",
    "Inglés": "<eng>",
    "Portugués": "<por>",
    "Francés": "<fra>",
    "Alemán": "<deu>",
    "Italiano": "<ita>",
    "Japonés": "<jpn>",
    "Chino mandarín": "<zho>",
}

# Tareas: transcripción (ASR) o traducción (ST) hacia un idioma destino.
TAREAS = {
    "Transcribir (ASR)": "<asr>",
    "Traducir → inglés": "<st_eng>",
    "Traducir → español": "<st_spa>",
    "Traducir → francés": "<st_fra>",
    "Traducir → alemán": "<st_deu>",
    "Traducir → portugués": "<st_por>",
}

st.set_page_config(page_title="OWSM v3.1 – Demo", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Cargando modelo (la primera vez se descarga de Hugging Face)…")
def obtener_modelo():
    """Carga el modelo una sola vez por sesión del servidor y lo reutiliza."""
    return cargar_modelo(MODELO_HF)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    idioma = st.selectbox("Idioma del audio", list(IDIOMAS), index=0)
    tarea = st.selectbox("Tarea", list(TAREAS), index=0)
    timestamps = st.checkbox("Predecir timestamps", value=False)
    st.caption(
        "La traducción en OWSM está entrenada principalmente entre inglés y otros "
        "idiomas; otras combinaciones pueden dar resultados pobres."
    )

    st.divider()
    st.subheader("🕒 Despliegue")
    st.write(f"**Fecha y hora:** {datetime.now():%Y-%m-%d %H:%M:%S}")
    st.write(f"**Dispositivo:** {dispositivo().upper()}")
    st.write(f"**Modelo:** `{MODELO_HF}` (101M)")

# ---------------------------------------------------------------- cuerpo principal
st.title("🎙️ OWSM v3.1 – Open Whisper-style Speech Model")
st.markdown(
    "Encoder **E-Branchformer** + decoder **Transformer**. "
    "Graba o sube un audio y observa la salida del modelo."
)

col_grabar, col_subir = st.columns(2)
with col_grabar:
    grabacion = st.audio_input("🎙️ Grabar desde el micrófono")
with col_subir:
    archivo = st.file_uploader("📁 O subir un archivo", type=["wav", "mp3", "flac", "ogg"])

fuente = grabacion or archivo
if fuente is None:
    st.info("Graba un audio o sube un archivo para comenzar.")
    st.stop()

# ---------------------------------------------------------------- ENTRADA
audio = cargar_audio(fuente)
duracion = len(audio) / FRECUENCIA_MUESTREO

st.subheader("📥 Entrada")
st.audio(audio, sample_rate=FRECUENCIA_MUESTREO)
st.write(
    f"Duración: **{duracion:.1f} s** · {FRECUENCIA_MUESTREO} Hz mono · "
    f"{len(audio):,} muestras"
)
if duracion > DURACION_MAX_S:
    st.warning(f"El audio dura más de {DURACION_MAX_S} s: solo se procesarán los primeros {DURACION_MAX_S} s.")

# Igual que Speech2Text._build_hyp_primer: <sos> <idioma> <tarea> [<notimestamps>].
# Con timestamps no se agrega nada más: el propio modelo genera los tokens <t.tt>.
prompt = f"<sos> {IDIOMAS[idioma]} {TAREAS[tarea]}" + ("" if timestamps else " <notimestamps>")
st.write("Tokens de control (prompt del decoder):")
st.code(prompt, language=None)

# ---------------------------------------------------------------- INFERENCIA
# El resultado se guarda en session_state: así, al cambiar de capa/cabeza en el
# mapa de atención, Streamlit re-ejecuta el script pero NO repite la inferencia.
clave = (fuente.file_id if hasattr(fuente, "file_id") else id(fuente), idioma, tarea, timestamps)
if st.button("▶️ Ejecutar inferencia", type="primary"):
    modelo = obtener_modelo()
    with st.spinner("Ejecutando inferencia (encoder + beam search + extracción de atención)…"):
        t0 = time.perf_counter()
        resultado = inferir_con_atencion(
            modelo, audio, idioma=IDIOMAS[idioma], tarea=TAREAS[tarea], timestamps=timestamps
        )
        st.session_state["resultado"] = (clave, resultado, time.perf_counter() - t0)

if "resultado" not in st.session_state or st.session_state["resultado"][0] != clave:
    st.stop()
_, r, t_inferencia = st.session_state["resultado"]

# ---------------------------------------------------------------- SALIDA
st.subheader("📤 Salida")
st.success(r.texto or "(sin texto)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tiempo de inferencia", f"{t_inferencia:.2f} s")
c2.metric("Tokens generados", len(r.tokens_generados))
c3.metric("Factor de tiempo real (RTF)", f"{t_inferencia / max(r.duracion, 1e-6):.2f}")
c4.metric("Confianza media por token", f"{r.probabilidades.mean():.0%}")

with st.expander("Texto crudo y tokens generados"):
    st.write("**Texto con tokens especiales:**")
    st.code(r.texto_crudo, language=None)
    st.write("**Tokens BPE generados (▁ marca inicio de palabra) y la probabilidad que les dio el decoder:**")
    st.dataframe(
        {"token": r.tokens_generados, "probabilidad": r.probabilidades.round(3)},
        width="stretch",
    )

# ---------------------------------------------------------------- CROSS-ATTENTION
st.subheader("🔍 Cross-attention: ¿a qué parte del audio mira cada token?")
st.markdown(
    "- **Q (queries)**: salen del **decoder**, una por token de texto → *¿qué sonido necesito para escribir el siguiente token?*\n"
    "- **K (keys)**: salen de la salida del **encoder**, una por frame de audio → *qué contiene cada trozo del audio*.\n"
    "- **V (values)**: también del **encoder** → *la información acústica que se le entrega al decoder*.\n\n"
    "Cada fila del mapa es `softmax(Q·Kᵀ/√d)` para un token: una **alineación suave texto ↔ audio**. "
    "En transcripción se espera un patrón aproximadamente **diagonal** (las palabras salen en el orden en que se dijeron)."
)

n_capas, n_cabezas = r.atencion.shape[:2]
col_capa, col_cab = st.columns(2)
op_capa = col_capa.selectbox("Capa del decoder", ["Promedio"] + [f"Capa {i + 1}" for i in range(n_capas)], index=n_capas)
op_cab = col_cab.selectbox("Cabeza de atención", ["Promedio"] + [f"Cabeza {i + 1}" for i in range(n_cabezas)])
capa = None if op_capa == "Promedio" else int(op_capa.split()[-1]) - 1
cabeza = None if op_cab == "Promedio" else int(op_cab.split()[-1]) - 1

st.pyplot(figura_atencion(r, capa, cabeza))
st.caption(
    "Solo se muestran los frames con audio real. El modelo rellena con silencio hasta 30 s y algunas "
    "cabezas ponen parte de su atención en ese relleno, por eso una fila puede no sumar 1 en lo visible."
)

if capa is not None:
    with st.expander(f"Ver las {n_cabezas} cabezas de la capa {capa + 1}"):
        st.pyplot(figura_cabezas(r, capa))
