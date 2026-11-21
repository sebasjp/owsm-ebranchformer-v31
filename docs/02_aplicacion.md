# Aplicación de inferencia OWSM v3.1 — Descripción

Aplicación web en **Streamlit** para hacer inferencia con el modelo
**OWSM v3.1 base** (`espnet/owsm_v3.1_ebf_base`, 101M parámetros) y visualizar
el mecanismo de **cross-attention** del decoder.

Responde a lo que pide la guía del curso: *"un método de visualización
interactiva que permita cargar datos y observar la inferencia del modelo"*.

---

## 1. Cómo ejecutarla

### Requisitos
- [uv](https://docs.astral.sh/uv/) instalado (gestiona Python y las dependencias).
- ≈1.5 GB de RAM libre y ≈400 MB de disco para el modelo.
- No requiere GPU: corre en CPU.

### Local
```bash
uv sync                          # crea .venv con Python 3.12 e instala dependencias (uv.lock)
uv run streamlit run app.py      # abre http://localhost:8501
```

La primera inferencia descarga el modelo desde Hugging Face Hub (≈393 MB) y lo
deja en caché. Las siguientes lo reutilizan.

### Prueba por consola (sin interfaz)
```bash
uv run python scripts/prueba_inferencia.py ruta/al/audio.wav --idioma "<spa>" --tarea "<asr>"
```
Imprime parámetros, tiempo de carga, tiempo de inferencia, RAM usada, tokens y texto.

### Caché del modelo
- **Windows:** se guarda en `<unidad del proyecto>:\hf_cache` (p. ej. `D:\hf_cache`).
  No puede ir dentro de la carpeta del proyecto: los archivos del modelo tienen
  rutas muy largas y superan el **límite de 260 caracteres** de Windows.
- **Linux / Colab:** se usa el caché por defecto (`~/.cache/huggingface`).
- Si se define la variable de entorno `HF_HOME`, se respeta.

---

## 2. Estructura del código

```
owsm-ebranchformer-v31/
├── pyproject.toml          # dependencias (gestionadas con uv)
├── uv.lock                 # versiones exactas, entorno reproducible
├── .python-version         # Python 3.12
├── app.py                  # interfaz Streamlit
├── owsm_demo/
│   ├── modelo.py           # descarga/carga del modelo, lectura de audio, transcripción
│   ├── atencion.py         # inferencia + extracción de la cross-attention
│   └── graficos.py         # log-Mel y mapas de calor de atención (matplotlib)
├── scripts/
│   └── prueba_inferencia.py  # prueba por consola con medición de tiempo y RAM
└── docs/
    ├── 01_resumen_conceptual.md
    └── 02_aplicacion.md    # este documento
```

| Módulo | Responsabilidad |
|---|---|
| `owsm_demo/modelo.py` | Configura el caché, carga `Speech2Text` de ESPnet (CPU/GPU automático, beam size 5, decodificación solo con atención: `ctc_weight=0`), lee audio y lo convierte a 16 kHz mono |
| `owsm_demo/atencion.py` | Ejecuta la inferencia y extrae los pesos de cross-attention de las 6 capas × 6 cabezas, el log-Mel de entrada y la probabilidad de cada token |
| `owsm_demo/graficos.py` | Dibuja el log-Mel y el mapa de atención con el mismo eje de tiempo, y la comparación de cabezas |
| `app.py` | Interfaz: entradas, ejecución, salidas y visualización |

---

## 3. Flujo de la aplicación

```
 Grabar (micrófono) ─┐
                     ├─► cargar_audio() ─► 16 kHz mono ─┐
 Subir archivo ──────┘                                  │
                                                        ▼
 Idioma + tarea + timestamps ─► prompt <sos> <spa> <asr> <notimestamps>
                                                        │
                                                        ▼
                                   inferir_con_atencion()
                                   1. beam search (texto)
                                   2. encoder → H
                                   3. decoder con teacher forcing → pesos de atención
                                                        │
                       ┌────────────────────────────────┼────────────────────────┐
                       ▼                                ▼                        ▼
                    Texto + métricas            Tokens + probabilidades   Mapa de cross-attention
```

---

## 4. Pantallas y funcionalidades

### Barra lateral
- **Idioma del audio:** español, inglés, portugués, francés, alemán, italiano, japonés, chino.
- **Tarea:** transcribir (`<asr>`) o traducir a inglés, español, francés, alemán o portugués (`<st_xxx>`).
- **Predecir timestamps:** quita `<notimestamps>` del prompt.
- **Despliegue:** fecha y hora, dispositivo (CPU/CUDA) y modelo. Sirve para el **pantallazo del entregable**.

### 📥 Entrada
- **🎙️ Grabar desde el micrófono** (`st.audio_input`) o **📁 subir un archivo** (wav, mp3, flac, ogg).
- Reproductor del audio, duración, frecuencia de muestreo y número de muestras.
- Aviso si el audio dura más de 30 s (solo se procesan los primeros 30 s).
- **Prompt de control** que recibe el decoder, p. ej. `<sos> <spa> <asr> <notimestamps>`.

### 📤 Salida
- Texto transcrito o traducido.
- Métricas:
  - **Tiempo de inferencia.**
  - **Tokens generados.**
  - **RTF** (*Real-Time Factor*): tiempo de inferencia ÷ duración del audio. Menor que 1 = más rápido que tiempo real.
  - **Confianza media por token.**
- Desplegable con el texto crudo (con tokens especiales) y la tabla de tokens BPE con la probabilidad que les asignó el decoder.

### 🔍 Cross-attention
- Explicación corta de Q, K y V para este problema.
- **Gráfico doble con el mismo eje de tiempo:**
  - Arriba: **log-Mel** real que entra al encoder (80 bandas, 10 ms por frame).
  - Abajo: **mapa de atención** `softmax(QKᵀ/√d)`. Filas = tokens generados (Q); columnas = frames del encoder de 40 ms (K/V).
- **Selectores de capa** (1–6 o promedio) y **cabeza** (1–6 o promedio). Por defecto: capa 6, promedio de cabezas.
- **Comparación de las 6 cabezas** de una capa lado a lado.
- Cambiar de capa o cabeza **no repite la inferencia**: el resultado queda en `st.session_state`.

---

## 5. Cómo se extrae la cross-attention

ESPnet no expone la atención directamente. El procedimiento en `owsm_demo/atencion.py`:

1. **Decodificación normal** con beam search → secuencia `[<sos>, <spa>, <asr>, <notimestamps>, …tokens…, <eos>]`.
2. **Encoder:** el audio se rellena a 30 s, pasa por el frontend (log-Mel) y por el encoder E-Branchformer → **H** (≈750 × 384).
3. **Decoder con teacher forcing:** se pasa la secuencia completa (sin `<eos>`) por el decoder junto con H. Cada capa de cross-attention (`decoder.decoders[i].src_attn`) guarda sus pesos softmax en el atributo `.attn`.
   - Antes se fuerza la implementación "manual" de la atención (`use_sdpa=False`, `use_flash_attn=False`), que es la única que guarda `.attn`. Después se restauran los valores originales.
4. Resultado: un tensor **(6 capas, 6 cabezas, tokens, frames)**.
   - La fila *i* corresponde a la predicción del token *i+1*. Se quitan las filas del prompt.
   - Se recortan los frames de relleno: solo se muestra el audio real.
5. De los logits del mismo paso se obtiene la **probabilidad de cada token generado**.

---

## 6. Desempeño medido (CPU, sin GPU, ≈7.4 GB RAM)

| Medida | Valor |
|---|---|
| Parámetros | 101.2M (coincide con el artículo) |
| Tamaño en disco | ≈393 MB |
| RAM tras cargar el modelo | ≈1.2 GB |
| Carga del modelo | ≈50 s (incluye la primera descarga) |
| Inferencia (audio de 4–5 s, beam 5) | ≈17–20 s |

El tiempo de inferencia casi no depende de la duración del audio (todo se rellena
a 30 s). La mayor parte se va en el beam search. Estas mediciones se hicieron con
audio sintético; falta medir con voz real.

---

## 7. Decisiones técnicas

| Decisión | Motivo |
|---|---|
| Solo modelo **base** | El medium (1.02B) necesita ≈5–6 GB de RAM y no cabe bien en un PC sin GPU |
| `ctc_weight=0` | Decodificar solo con el decoder (atención), que es el camino cuya cross-attention visualizamos |
| **uv** + `pyproject.toml` + `uv.lock` | Entorno reproducible para todo el equipo, sin `requirements.txt` |
| Python **3.12** | Coincide con la versión de Colab |
| Caché en `D:\hf_cache` | Límite de 260 caracteres de Windows y poco espacio en el disco C |
| Teacher forcing para la atención | Obtener la atención de la hipótesis final completa en una sola pasada |

---

## 8. Limitaciones conocidas

- Audios de **más de 30 s** se recortan (no hay decodificación por ventanas en la app).
- **Alucinaciones** con audio sin voz: el modelo siempre genera texto.
- La **traducción** funciona bien solo desde o hacia el inglés.
- `torch` (2.14.1) y `torchaudio` (2.11.1) quedaron en versiones distintas. Hasta ahora no ha causado errores.
- La app aún no se ha probado con **voz real** grabada desde el navegador.

---

## 9. Pendientes

- [ ] Probar la app con voz real (grabación y archivo).
- [ ] **Métricas de desempeño:** campo de texto de referencia para calcular **WER** y **CER** (requisito del README del curso).
- [ ] **Selector de beam size** (1 = greedy, rápido; 5 = mejor calidad).
- [ ] **Audios de ejemplo** propios en `audios/` como respaldo para la sustentación.
- [ ] (Opcional) Detección automática de idioma.
- [ ] (Opcional) Notebook de Colab para correr el modelo medium con GPU.
