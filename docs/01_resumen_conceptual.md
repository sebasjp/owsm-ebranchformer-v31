# OWSM v3.1 — Resumen conceptual para el equipo

> Documento de estudio con todo lo necesario para entender el artículo y poder
> explicarlo en la sustentación. Los datos marcados con **(config)** fueron
> verificados en el `config.yaml` del modelo base que usamos; los marcados con
> **(artículo)** vienen del paper y conviene citarlos desde allí.

**Artículo base:** Y. Peng et al., *"OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer"*, Interspeech 2024. arXiv: [2401.16658](https://arxiv.org/abs/2401.16658)
**Modelo usado:** [`espnet/owsm_v3.1_ebf_base`](https://huggingface.co/espnet/owsm_v3.1_ebf_base) (101M parámetros)
**Modelo principal del artículo:** [`espnet/owsm_v3.1_ebf`](https://huggingface.co/espnet/owsm_v3.1_ebf) (1.02B parámetros)

---

## 1. ¿De qué trata el artículo?

**OWSM** = *Open Whisper-style Speech Model*. Es un modelo de **voz → texto** desarrollado por CMU con el toolkit ESPnet, al estilo de **Whisper** de OpenAI.

- **Problema:** Whisper funciona muy bien, pero OpenAI no publicó los datos de entrenamiento ni el código para entrenarlo. Eso impide reproducirlo, auditarlo o estudiarlo científicamente.
- **Propuesta de OWSM:** reproducir el enfoque de Whisper usando **solo datos públicos** (≈180 mil horas, 151 idiomas **(artículo)**), y publicar código, pesos y registros de entrenamiento.
- **Qué aporta la v3.1:** usa los mismos datos que OWSM v3, pero **cambia el encoder Transformer por un encoder E-Branchformer**. El decoder sigue siendo un Transformer estándar. Resultado: mejor desempeño en la mayoría de las pruebas y hasta ≈25 % más rápido en inferencia **(artículo)**.
- **Tamaños publicados:** base (101M), small (367M), medium (1.02B).

Cumple el requisito del curso porque es un **Transformer encoder–decoder** aplicado a una secuencia compleja: audio → texto.

---

## 2. Arquitectura general

```
          AUDIO (16 kHz, mono, relleno o recortado a 30 s)
            │
   log-Mel: 80 bandas, un frame cada 10 ms  → 3000 frames para 30 s   (config)
            │
   conv2d: reduce la secuencia ×4 → un vector cada 40 ms → ≈750 frames
            │
   ┌──────────────────────────────┐
   │ ENCODER E-Branchformer       │  6 bloques, d = 384, 6 cabezas   (config)
   │ (solo self-attention audio↔audio + cgMLP)
   └──────────────────────────────┘
            │
            H = memoria acústica (≈750 × 384)
            │  (el MISMO H va a todas las capas del decoder)
            ▼
   ┌──────────────────────────────────────────────────┐
   │ DECODER Transformer   6 capas, 6 cabezas  (config)│
   │  cada capa: self-attn (texto) → CROSS-ATTN (H) → FFN
   └──────────────────────────────────────────────────┘
            │
   softmax sobre el vocabulario (≈50k tokens BPE) → siguiente token
```

Además hay una **cabeza CTC** sobre el encoder. El entrenamiento es **híbrido CTC + atención** (peso CTC 0.3 **(config)**). La cabeza CTC ayuda al encoder a alinear audio y texto, pero no decide la tarea.

### Configuración real del modelo base **(config)**

| Componente | Valor |
|---|---|
| Frontend | log-Mel, `n_fft=512`, `hop=160` (10 ms), 80 bandas, normalización `global_mvn` |
| Submuestreo | `conv2d` (×4 → 40 ms por frame) |
| Encoder | E-Branchformer, 6 bloques, d=384, 6 cabezas, cgMLP 1536 unidades, kernel conv 31, FFN macaron |
| Decoder | Transformer, 6 capas, 6 cabezas, FFN 1536 |
| Entrenamiento | CTC 0.3 + atención 0.7, label smoothing 0.1 |
| Vocabulario | BPE de 50k (sentencepiece) + tokens especiales |

---

## 3. La innovación: encoder E-Branchformer

Cada bloque del encoder procesa el audio con **dos ramas en paralelo**:

| Rama | Mecanismo | Qué captura |
|---|---|---|
| **Global** | Self-attention multi-cabeza | Relaciones lejanas en el audio (contexto de toda la frase) |
| **Local** | **cgMLP**: MLP con compuerta convolucional | Patrones cortos: fonemas, transiciones entre sonidos |

- Las dos ramas se **fusionan** con una convolución depthwise y una capa lineal.
- El bloque tiene **FFN tipo "macaron"**: una feed-forward antes y otra después.
- Diferencia con el **Conformer**, que combina atención y convolución **en secuencia**: el E-Branchformer lo hace **en paralelo** y luego fusiona.
- Por qué funciona para voz: el habla necesita **contexto global** (qué palabra tiene sentido en la frase) y **detalle local** (qué fonema suena en estos milisegundos).

En el encoder **solo hay self-attention**: Q, K y V salen todas del audio. **No hay cross-attention en el encoder.**

---

## 4. ENTRADAS del modelo

El modelo recibe **dos** entradas:

| Entrada | Va a | Qué es |
|---|---|---|
| **Audio** | Encoder | Lo que se dijo. Forma de onda mono a 16 kHz, relleno o recortado a 30 s, convertido a log-Mel |
| **Prompt de tokens** | Decoder | Qué hacer con el audio |

### El prompt de control

```
<sos> <idioma> <tarea> <notimestamps>
ej.: <sos> <spa> <asr>    <notimestamps>   → transcribir español
     <sos> <spa> <st_eng> <notimestamps>   → traducir español → inglés
```

- `<sos>`: inicio de la secuencia.
- `<spa>`, `<eng>`, …: idioma del audio (151 idiomas en el vocabulario).
- `<asr>` / `<st_xxx>`: tarea. Hay `<asr>` y 25 tokens de traducción `<st_eng>`, `<st_spa>`, …
- `<notimestamps>`: no generar marcas de tiempo. Si se quita, el modelo genera tokens de tiempo `<0.00>`, `<0.02>`, … (resolución de 20 ms).
- Opcional: un **texto previo** como contexto (para transcribir audios largos por ventanas).

El decoder **no empieza desde cero**: arranca con estos tokens ya escritos y desde ahí genera el resto.

---

## 5. SALIDAS del modelo

- Una **secuencia de tokens BPE** generada **de forma autorregresiva** (un token a la vez) hasta `<eos>`. El símbolo `▁` marca el inicio de una palabra.
- En cada paso el decoder produce una **distribución de probabilidad sobre el vocabulario**. El token se elige con **beam search**: se exploran varias hipótesis en paralelo y se queda la mejor.
- Al juntar los tokens se obtiene:
  - **ASR:** la transcripción en el mismo idioma.
  - **ST:** la traducción.
  - **Timestamps:** el texto intercalado con tokens de tiempo.
  - **Idioma:** si no se da el idioma, el primer token que genera es el idioma.

---

## 6. Un modelo, muchas tareas (multitarea)

La arquitectura está pensada para **varias tareas con un solo conjunto de pesos**, como Whisper. En el entrenamiento se mezclaron ejemplos de todas las tareas, y **la tarea se elige con los tokens del prompt**, no cambiando el modelo.

| Tarea | Cómo se pide |
|---|---|
| ASR (reconocimiento de voz) | `<idioma> <asr>` |
| ST (traducción de voz) | `<idioma> <st_xxx>` |
| LID (identificación de idioma) | solo `<sos>`; el modelo genera el idioma |
| Timestamps | quitar `<notimestamps>` |
| Transcripción larga | pasar el texto anterior como contexto |

- El **encoder** es el mismo para todas las tareas: solo "entiende el audio".
- Lo que cambia según la tarea es lo que genera el **decoder**, condicionado por el prompt.

### ¿Cómo sabe el modelo el idioma?

- El encoder **no** le dice al decoder el idioma de forma explícita. Su salida H es solo una matriz de vectores acústicos, sin etiquetas.
- **Opción 1 — se lo damos** (lo que hace nuestra app): el token `<spa>` es una instrucción en el prompt. Si se da un idioma equivocado, el modelo igual lo intenta y produce texto raro.
- **Opción 2 — lo predice:** en el entrenamiento el token de idioma era parte del texto objetivo, así que el decoder aprendió a predecirlo. Para hacerlo usa la cross-attention sobre H: la información del idioma está *implícita* en el audio codificado, y el decoder la convierte en un token.

---

## 7. Las tres atenciones del modelo

| Atención | Dónde | Q viene de | K, V vienen de |
|---|---|---|---|
| Self-attention del encoder | Rama global del E-Branchformer | Audio | Audio |
| Self-attention del decoder (con máscara causal) | Cada capa del decoder | Texto | Texto (solo tokens anteriores) |
| **Cross-attention** | Cada capa del decoder | **Texto** | **Audio (H)** |

La **cross-attention es el único puente** entre el texto y el audio.

---

## 8. Q, K y V en la cross-attention (para nuestro problema)

| | Sale de | Qué representa | Forma (base) |
|---|---|---|---|
| **Q** (query) | Estado del **decoder** en cada token: lo ya escrito + el prompt | La **pregunta**: *"¿qué parte del audio necesito para escribir el siguiente token?"* | tokens × 384 |
| **K** (key) | Salida del **encoder** H, un vector por cada 40 ms | La **etiqueta** de cada trozo de audio: *"aquí suena tal cosa"* | frames × 384 |
| **V** (value) | La misma H, con otra proyección | El **contenido** acústico que se entrega al decoder | frames × 384 |

`Q = W_Q · x_decoder`, `K = W_K · H`, `V = W_V · H`, con matrices aprendidas de 384 × 384.

### El cálculo

1. **Comparar:** `Q · Kᵀ / √d_k` da qué tanto responde cada trozo de audio a la pregunta.
2. **Normalizar:** `softmax` convierte los puntajes en **pesos que suman 1** sobre el audio.
3. **Extraer:** el promedio de **V** ponderado por esos pesos es la información del audio que usa el decoder.

### Dimensiones y cabezas (modelo base)

- d_model = **384** → se reparte en **6 cabezas de 64** (6 × 64 = 384). Escalado: √64 = 8.
- Cada cabeza tiene sus propias W_Q, W_K y W_V y puede aprender a buscar algo distinto.
- Las salidas de las 6 cabezas se concatenan de nuevo a 384 y pasan por una capa lineal.
- Ejemplo con 10 s de audio y 12 tokens: Q = 12 × 384; K, V ≈ 250 × 384 (o ≈750 con el relleno a 30 s); mapa por cabeza = 12 × 250.
- Total en el decoder: **6 capas × 6 cabezas = 36 mapas de cross-attention**.

### Interpretación

- Los pesos forman una **alineación suave texto ↔ audio**: dicen *en qué momento del audio se dijo cada palabra*.
- **ASR:** patrón **diagonal**, porque las palabras salen en el mismo orden en que se pronunciaron.
- **ST (traducción):** la diagonal se rompe donde el idioma destino **reordena** las palabras.
- **K y V no cambian con la tarea** (el audio es el mismo). Lo que cambia es la **pregunta Q**: con `<asr>` pregunta *"qué sonido es"*; con `<st_eng>`, *"qué significa"*.

> **En una frase:** Q es el texto que pregunta, K es el índice del audio que responde y V es el contenido del audio que se entrega. El resultado es un mapa de qué parte del audio se usó para escribir cada palabra.

---

## 9. Cómo leer el mapa de cross-attention

```
              tiempo del audio →
              0.0s   0.5s   1.0s   1.5s   2.0s
   ▁hola     [ ██                            ]   ← "hola" se dijo cerca de 0.2 s
   ▁cómo     [        ██                     ]
   ▁estás    [               ███             ]
   ?         [                     █         ]
   <eos>     [                          ░░░  ]
```

- **Fila** = un token generado (de ahí sale la Q).
- **Columna** = un trozo de 40 ms del audio (de ahí salen K y V).
- **Color** = cuánto miró el decoder ese trozo de audio para generar ese token.

### ¿Qué significa cada "capa"?

- Son las **6 capas del DECODER**, no del encoder.
- El encoder procesa el audio **una sola vez**. Todas las capas del decoder consultan **el mismo H**, cada una con su propia cross-attention.
- Entre capas cambia la **Q**, porque el decoder va refinando su representación:
  - **Capas bajas:** el decoder sabe poco (identidad y posición del token). La atención suele ser **difusa**.
  - **Capas medias y altas:** Q más precisas. Suele aparecer la **diagonal limpia**.
- Algunas **cabezas "descansan"**: ponen su atención en un punto fijo (el inicio o el relleno de silencio) cuando no necesitan información del audio.

Este es el comportamiento típico en modelos tipo Whisper. Qué hace cada capa en *este* modelo hay que verificarlo con nuestros propios audios.

---

## 10. Limitaciones (para el informe y la sustentación)

- **Ventana fija de 30 s:** todo audio se rellena o recorta a 30 s. Un audio de 3 s cuesta casi lo mismo que uno de 30 s, y lo que pase de 30 s se corta (salvo usando decodificación por ventanas).
- **Alucinaciones:** el modelo siempre intenta generar texto. Con un tono más ruido, sin voz, generó *"en la enfermedad."* con baja confianza.
- **Traducción limitada:** entrenada sobre todo desde o hacia el inglés. Otras combinaciones (p. ej. español → francés) funcionan mal.
- **Tamaño vs. calidad:** el modelo base (101M) es claramente peor que el medium (1.02B) en traducción e idiomas con pocos datos.
- **Costo computacional:** la decodificación autorregresiva con beam search es lenta en CPU (≈20 s por audio en nuestro equipo).
- **Datos públicos:** menos datos que Whisper, así que en varios idiomas no alcanza su calidad.

---

## 11. Glosario rápido

| Término | Significado |
|---|---|
| ASR | *Automatic Speech Recognition*: voz → texto en el mismo idioma |
| ST | *Speech Translation*: voz → texto en otro idioma |
| LID | *Language Identification*: detectar el idioma hablado |
| log-Mel | Espectrograma en escala Mel con logaritmo; la "imagen" del audio que entra al modelo |
| BPE | *Byte Pair Encoding*: tokens de subpalabras (`▁ho` + `la`) |
| CTC | *Connectionist Temporal Classification*: pérdida auxiliar que alinea audio y texto |
| Beam search | Búsqueda que mantiene las N mejores hipótesis en cada paso |
| Teacher forcing | Pasar al decoder la secuencia completa de una vez (la usamos para extraer la atención) |
| WER / CER | *Word/Character Error Rate*: porcentaje de palabras o caracteres mal reconocidos |
| RTF | *Real-Time Factor*: tiempo de inferencia ÷ duración del audio |
