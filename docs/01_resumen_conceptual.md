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

## 4. Entrenamiento híbrido CTC + atención

**Sí, OWSM v3.1 usa CTC en el entrenamiento.** Esta sección explica qué es CTC
desde cero, cómo se conecta con el resto del modelo y cómo lo verificamos.

### 4.1 El problema que resuelve CTC

Tenemos **muchos frames de audio** y **pocos tokens de texto**:

```
audio de 1 s  →  25 frames del encoder (uno cada 40 ms)
texto         →  "hola"  (pocos tokens)
```

Para entrenar haría falta saber **qué frame corresponde a qué letra o token**
(p. ej. que la "h" está en los frames 3–5). Pero los datos de entrenamiento
**no traen esa información**: solo traen el audio y su transcripción completa.
Nadie anotó en qué milisegundo empieza cada sonido.

**CTC (*Connectionist Temporal Classification*) permite entrenar sin conocer esa alineación.**

### 4.2 La idea: un símbolo por frame, más un símbolo "vacío"

La cabeza CTC es muy simple: **una sola capa lineal** (`ctc_lo`, 384 → 50002)
aplicada **a cada frame del encoder por separado**:

```
H (750 frames × 384)  →  ctc_lo  →  softmax  →  750 distribuciones sobre 50002 símbolos
```

Para cada trozo de 40 ms predice un símbolo. Los 50002 símbolos son los ~50k
tokens del vocabulario más uno especial: **`_` (blank)**, que significa
*"aquí no emito nada nuevo"*.

### 4.3 La regla para convertir frames en texto

1. **Juntar** repeticiones seguidas del mismo símbolo.
2. **Borrar** los blanks.

Ejemplo con 10 frames (con letras, para que sea fácil de ver):

```
frames:     1  2  3  4  5  6  7  8  9  10
predicción: _  h  h  o  _  _  l  a  a  _
paso 1:     _  h     o  _     l  a     _       (junta repetidos)
paso 2:        h     o        l  a             (borra blanks)
resultado:  "hola"
```

¿Para qué sirve el blank?
- Permite **"esperar"** en silencios o transiciones sin emitir nada.
- Permite **letras dobles**: sin blank, `l l` se juntaría en una sola. Para "ll" hace falta `l _ l`.

### 4.4 ¿Cómo se entrena sin saber la alineación?

Hay **muchísimas formas** de alinear "hola" en 10 frames, y todas producen el mismo texto:

```
_ h h o _ _ l a a _
h o o o l l l a _ _
_ _ h o l a _ _ _ _
...  (miles de caminos posibles)
```

CTC **no elige uno**: **suma la probabilidad de todos los caminos** que, tras
aplicar la regla, dan el texto correcto.

```
P("hola" | audio) = Σ P(camino)     sobre todos los caminos que colapsan a "hola"
L_CTC             = −log P("hola" | audio)
```

**La clave:** no necesitamos saber cuál es el camino correcto; **solo necesitamos
saber cuál es el texto correcto**, y ese sí lo tenemos (cada audio de
entrenamiento viene con su transcripción). Cualquier camino que colapse a ese
texto cuenta como válido: no hay un único camino "verdadero".

#### Ejemplo con números

Audio de **3 frames**, vocabulario de solo `{a, b, _}` y transcripción **"ab"**.
Lo que predice el modelo en cada frame (cada columna suma 1):

| | frame 1 | frame 2 | frame 3 |
|---|---|---|---|
| a | **0.6** | 0.3 | 0.1 |
| b | 0.1 | 0.3 | **0.7** |
| _ | 0.3 | 0.4 | 0.2 |

**Paso 1 — listar los caminos de 3 frames que colapsan a "ab"** (se pueden
enumerar solo con la regla, sin saber nada del audio). La probabilidad de un
camino es el producto de las probabilidades de cada frame:

| Camino | ¿Colapsa a "ab"? | Probabilidad |
|---|---|---|
| `a a b` | ✅ | 0.6 × 0.3 × 0.7 = 0.126 |
| `a b b` | ✅ | 0.6 × 0.3 × 0.7 = 0.126 |
| `a _ b` | ✅ | 0.6 × 0.4 × 0.7 = 0.168 |
| `_ a b` | ✅ | 0.3 × 0.3 × 0.7 = 0.063 |
| `a b _` | ✅ | 0.6 × 0.3 × 0.2 = 0.036 |
| `b a _`, `a b a`, `_ _ a`, … | ❌ dan otro texto | no se cuentan |

**Paso 2 — sumar los válidos:**

```
P("ab" | audio) = 0.126 + 0.126 + 0.168 + 0.063 + 0.036 = 0.519
L_CTC           = −log(0.519) ≈ 0.66
```

Nunca tuvimos que decir cuál de los 5 caminos era el verdadero.

#### ¿Cómo sabe si está bien o mal?

- "Bien" significa que **la suma sea alta (cercana a 1)**: casi toda la probabilidad del modelo cae en caminos que dan el texto correcto.
- Si el modelo predice cosas que llevan a otros textos ("ba", "aba"…), esa probabilidad **se desperdicia**, la suma baja y la pérdida sube.
- El entrenamiento ajusta los pesos para **subir la suma**. Como el softmax de cada frame reparte una cantidad fija, subir los caminos válidos baja automáticamente los inválidos.
- No es una clasificación de cada frame: es una evaluación de la **secuencia completa**. Las predicciones por frame son un paso intermedio que nadie supervisa directamente.

#### ¿Cómo aprende la alineación correcta si nadie se la enseña?

Por **repetición y consistencia** a lo largo de miles de ejemplos:

1. Al principio el modelo predice casi al azar: todos los caminos valen parecido y la corrección se reparte entre todos.
2. Pero en miles de audios, **cada vez que aparece la "h" en el texto, hay un sonido de "h"** en algún lugar del audio.
3. La forma de subir la suma **en todos los audios a la vez** es asociar la "h" con ese sonido. Cualquier otra asignación funciona en un audio y falla en otro.
4. Se refuerza sola: cuando el modelo empieza a acertar, los caminos con la "h" en el lugar correcto pesan más en la suma y reciben más corrección.

> **Analogía:** es como aprender un idioma con películas **subtituladas pero sin
> sincronizar**. Sabes qué se dijo en la escena, pero no en qué segundo. Después
> de muchas películas, igual descubres qué sonido corresponde a qué palabra.

#### ¿Y si hay millones de caminos?

Con 750 frames y frases reales hay una cantidad astronómica de caminos. CTC usa
**programación dinámica** (algoritmo *forward-backward*): avanza frame por frame
guardando sumas parciales (*"probabilidad de haber escrito ya 'a' hasta el frame t"*).
Obtiene **exactamente la misma suma** sin enumerar los caminos uno por uno.

### 4.5 ¿Contra qué se compara? El texto como índices de tokens

El texto de referencia se **tokeniza con el mismo BPE del modelo** y se convierte
en **índices** del vocabulario. Ejemplo real con el tokenizador del modelo base:

```
"hola cómo estás"
   │ tokenizador BPE (bpe.model del modelo)
   ▼
['▁hol', 'a', '▁cómo', '▁está', 's']
   │ token_list del config.yaml
   ▼
[13944, 1721, 18111, 5818, 1687]     ← esta es la "etiqueta" del audio
```

```
"procesamiento de datos secuenciales"
→ ['▁proces', 'amiento', '▁de', '▁dat', 'os', '▁se', 'cu', 'encia', 'les']
→ [22567, 28304, 1708, 2285, 2014, 1879, 2847, 6098, 2665]
```

- El *blank* es el **índice 0** y el vocabulario tiene **50002** entradas (blank incluido).
- Por eso la capa CTC tiene exactamente **50002 salidas**: una columna por cada índice posible.

#### Los índices dicen qué columna leer

La cabeza CTC produce una tabla de **750 frames × 50002 columnas**. **No se
compara texto con texto**: los índices de la etiqueta indican qué columnas importan.

```
                       columnas (índices de token)
                 0(_)   1687(s)  1721(a)  5818(▁está) 13944(▁hol) 18111(▁cómo)  …otras 49996
frame   1        0.90    …        …        …          0.02         …
frame   2        0.40    …        …        …          0.55         …
frame   3        0.10    …        0.80     …          …            …
…
frame 750        0.95    …
```

Para un camino como `_ ▁hol ▁hol a _ ▁cómo …` se toma en cada frame **el valor de
la columna que ese camino indica** (frame 1 → columna 0, frame 2 → columna 13944,
frame 3 → columna 13944, frame 4 → columna 1721…) y se multiplican. Las ~49996
columnas que no están en la etiqueta no se leen: su probabilidad es la que "se
desperdicia" y el entrenamiento la empuja hacia abajo. Es el mismo ejemplo de
`{a, b, _}`, pero con índices reales.

#### La misma etiqueta, dos formas de contrastar

| | **Pérdida CTC** | **Pérdida del decoder (atención)** |
|---|---|---|
| Salida del modelo | 750 frames × 50002 | 1 distribución por token × 50002 |
| Etiqueta | `[13944, 1721, 18111, 5818, 1687]` | prompt + texto: `<spa> <asr> <notimestamps> 13944 1721 …` |
| ¿Cuántas salidas para cuántos tokens? | **750 para 5**: no coinciden | **una por token**: coinciden |
| Cómo se contrasta | Suma de todos los caminos válidos (no se sabe qué frame es qué token) | **Posición a posición** (entropía cruzada): en el paso *i* se lee la columna del token correcto *i* |

El decoder **no necesita sumar caminos**: genera exactamente un token por posición,
así que la comparación es directa (*"en la posición 2 el token correcto es 1721,
¿qué probabilidad le diste?"*). CTC necesita el truco de la suma porque tiene
**muchos más frames que tokens**.

> Lo habitual en ESPnet es que la etiqueta de la rama CTC sea **solo el texto**,
> sin tokens de idioma, tarea ni timestamps (CTC no maneja el prompt).
> **No lo verificamos** en este modelo: presentarlo como "lo habitual".

### 4.6 CTC obliga a un orden

Por la regla del colapso, CTC **solo puede leer el texto en el mismo orden del
audio**: los símbolos salen de izquierda a derecha y nunca se puede volver atrás.

- Para **transcribir** es perfecto: lo que se dice primero, se escribe primero.
- Para **traducir** no sirve sola: *"the red car"* → *"el carro rojo"* requiere reordenar.
- Además, CTC decide **cada frame de forma independiente**: no mira lo que ya escribió (no tiene "modelo de lenguaje" interno).

### 4.7 CTC es una rama lateral: la salida del encoder NO es texto

Es fácil confundirse y pensar que el encoder colapsa el audio a texto antes de
pasarlo al decoder. **No es así.**

```
log-Mel:        3000 frames × 80     (un frame cada 10 ms)
   │ conv2d ×4   ← única reducción: comprime el TIEMPO, siguen siendo frames de audio
   ▼
H (encoder):    750 frames × 384     (un frame cada 40 ms)
```

- **H tiene una fila por cada trozo de 40 ms de audio**, siempre. Los bloques E-Branchformer no cambian la cantidad de frames: entran 750 y salen 750.
- **H se copia hacia dos destinos independientes:**

```
                                   ┌─► cabeza CTC: 750 × 50002 → [juntar + borrar blanks] → texto CTC
                                   │      (solo para su pérdida en entrenamiento,
                                   │       o para decodificar solo con CTC; NO se pasa a nadie)
audio → conv2d → ENCODER → H (750 × 384)
                                   │
                                   └─► DECODER: K = H·W_K (750 × 384)
                                                V = H·W_V (750 × 384)   ← los 750 frames completos
```

- "Juntar repetidos y borrar blanks" ocurre **solo dentro de la rama CTC**. Su resultado no vuelve al encoder ni pasa al decoder.
- Es el **decoder**, con la cross-attention, el que pasa de muchos frames a pocos tokens: para cada token "elige" qué frames usar.
- Por eso, en el mapa de cross-attention de la app, **el eje X es tiempo** (frames de 40 ms) y no palabras.

| Paso | Forma | ¿Qué es? |
|---|---|---|
| log-Mel | 3000 × 80 | audio, cada 10 ms |
| conv2d | 750 × 384 | audio comprimido, cada 40 ms |
| Encoder (H) | **750 × 384** | audio con contexto → **va al decoder como K y V** |
| Rama CTC | 750 × 50002 → texto | lateral: solo entrenamiento (o decodificación CTC opcional) |
| Decoder | tokens × 50002 | texto, un token a la vez |

### 4.8 ¿Cómo se relacionan CTC y el decoder?

**Hacia adelante no se pasan información**: los dos leen el mismo H, en paralelo.
Se relacionan de dos formas:

**a) En el entrenamiento, a través de la pérdida combinada y el encoder compartido**

```
pérdida total = 0.3 · L_CTC + 0.7 · L_atención
```

En la **retropropagación** (el paso en que se corrigen los pesos según el error),
el gradiente de cada pérdida viaja hacia atrás por su rama y **los dos llegan al encoder**:

```
            L_CTC (0.3)                    L_atención (0.7)
                │ gradiente                     │ gradiente
                ▼                               ▼
          cabeza CTC                         DECODER
                │                               │
                └───────────────┬───────────────┘
                                ▼
                    ENCODER (recibe la SUMA de ambos gradientes)
```

| Pesos de… | Los corrige… |
|---|---|
| Cabeza CTC | solo L_CTC |
| Decoder | solo L_atención |
| **Encoder** | **las dos a la vez** |

CTC no le "dice" nada al decoder directamente. Actúa como un **"profesor estricto"
del encoder**: lo obliga a que H tenga información **ordenada en el tiempo** y
bien localizada. El decoder se beneficia porque **consulta ese H** con su
cross-attention. La influencia es **indirecta, a través de los pesos compartidos**.
Por eso se dice que CTC **estabiliza y acelera** el entrenamiento, y ayuda a que
la cross-attention forme la diagonal.

**b) En la inferencia, combinando puntajes (opcional)**

ESPnet puede usar las dos ramas en el beam search. Cada hipótesis se califica con ambas:

```
puntaje = λ · log P_CTC(texto) + (1 − λ) · log P_atención(texto)
```

El decoder **propone** tokens y CTC **vota** si son coherentes con el orden del
audio (ayuda a frenar alucinaciones y repeticiones). Sigue sin haber intercambio
de vectores: solo se combinan puntajes.

| Momento | ¿Cómo se relacionan CTC y el decoder? |
|---|---|
| Hacia adelante | No se relacionan: ambos leen el mismo H, en paralelo |
| Entrenamiento | Por la **pérdida combinada**: sus gradientes se suman en el **encoder compartido** |
| Inferencia (opcional) | Combinando sus **puntajes** en el beam search |

### 4.9 CTC vs. decoder con atención

| | **Cabeza CTC** | **Decoder (cross-attention)** |
|---|---|---|
| Arquitectura | 1 capa lineal sobre cada frame | 6 capas Transformer |
| Cómo genera | Todos los frames a la vez (en paralelo) | Un token a la vez (autorregresivo) |
| Orden | Obligatoriamente **monótono** | Libre: puede reordenar |
| Mira el texto anterior | No | Sí (self-attention) |
| Multitarea (prompt) | No | Sí (`<spa> <asr>` …) |
| Velocidad | Muy rápida | Lenta (beam search) |
| Ventaja | Alineación robusta, no "alucina" texto largo | Flexible, fluido, más preciso |

Por eso OWSM entrena con **las dos**: CTC organiza el encoder y el decoder aporta
la flexibilidad (multitarea, traducción, timestamps). **CTC complementa, no reemplaza.**

### 4.10 Evidencias en los archivos del modelo base

| Fuente | Qué muestra |
|---|---|
| `config.yaml` | `model_conf.ctc_weight: 0.3` y un bloque `ctc_conf` (`ctc_type: builtin`) **(config)** |
| `train.log` (última época) | `loss_ctc=43.342`, `loss_att=49.292`, `loss=47.507` → **0.3 × 43.342 + 0.7 × 49.292 = 47.507** |
| `images/` | Gráficas `loss_ctc.png` y `cer_ctc.png`: curvas de entrenamiento de la cabeza CTC |
| Pesos `.pth` | Capa `ctc.ctc_lo` de forma **(50002 × 384)**: proyecta cada vector de 384 del encoder a una probabilidad sobre los 50k tokens + el símbolo *blank* de CTC |

En el artículo, OWSM v3.1 mantiene el esquema **híbrido CTC/atención** de OWSM v3, que es el estándar de ESPnet **(artículo — confirmar la cita exacta en la sección de entrenamiento)**.

### 4.11 CTC en nuestra app

- **Usamos `ctc_weight=0`**: el texto se genera **solo con el decoder**. Fue a propósito, para que la salida provenga de la cross-attention que visualizamos.
- La cabeza CTC sigue existiendo en el modelo, pero no se ejecuta para generar.
- ESPnet **no permite** combinar CTC con timestamps: si `ctc_weight > 0` y se piden timestamps, lanza el error `"CTC cannot predict timestamps"`.

### 4.12 Preguntas que nos surgieron (y sus respuestas)

- **¿La salida del encoder ya es texto?** No. H sigue siendo audio: 750 frames × 384. Solo la rama CTC convierte frames en texto, y ese texto no va a ningún lado.
- **¿Por qué K y V tienen 750 filas y no tantas como palabras?** Porque salen de H, que tiene una fila por cada 40 ms de audio.
- **¿Quién reduce 3000 frames a 750?** La `conv2d` al entrar al encoder. Es compresión temporal, no texto.
- **¿CTC y el decoder se pasan información?** No directamente. Se relacionan por la pérdida combinada (sus gradientes se suman en el encoder) y, opcionalmente, en la inferencia combinando puntajes.
- **¿Si CTC no se usa en la inferencia, para qué está?** Para entrenar mejor el encoder. Su efecto queda "guardado" en los pesos del encoder que sí usamos.
- **¿Cómo sabe CTC cuál es el camino correcto si no hay etiqueta por frame?** No lo necesita: solo necesita el texto correcto. Suma la probabilidad de **todos** los caminos que dan ese texto (ver el ejemplo numérico en 4.4).
- **¿Contra qué se compara la salida?** Contra los **índices de tokens** del texto de referencia (tokenizado con el mismo BPE). Esos índices dicen qué columnas de la salida de 50002 leer en cada frame (ver 4.5).

---

## 5. ENTRADAS del modelo

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

## 6. SALIDAS del modelo

- Una **secuencia de tokens BPE** generada **de forma autorregresiva** (un token a la vez) hasta `<eos>`. El símbolo `▁` marca el inicio de una palabra.
- En cada paso el decoder produce una **distribución de probabilidad sobre el vocabulario**. El token se elige con **beam search**: se exploran varias hipótesis en paralelo y se queda la mejor.
- Al juntar los tokens se obtiene:
  - **ASR:** la transcripción en el mismo idioma.
  - **ST:** la traducción.
  - **Timestamps:** el texto intercalado con tokens de tiempo.
  - **Idioma:** si no se da el idioma, el primer token que genera es el idioma.

---

## 7. Un modelo, muchas tareas (multitarea)

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

## 8. Las tres atenciones del modelo

| Atención | Dónde | Q viene de | K, V vienen de |
|---|---|---|---|
| Self-attention del encoder | Rama global del E-Branchformer | Audio | Audio |
| Self-attention del decoder (con máscara causal) | Cada capa del decoder | Texto | Texto (solo tokens anteriores) |
| **Cross-attention** | Cada capa del decoder | **Texto** | **Audio (H)** |

La **cross-attention es el único puente** entre el texto y el audio.

---

## 9. Q, K y V en la cross-attention (para nuestro problema)

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

## 10. Cómo leer el mapa de cross-attention

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

## 11. Limitaciones (para el informe y la sustentación)

- **Ventana fija de 30 s:** todo audio se rellena o recorta a 30 s. Un audio de 3 s cuesta casi lo mismo que uno de 30 s, y lo que pase de 30 s se corta (salvo usando decodificación por ventanas).
- **Alucinaciones:** el modelo siempre intenta generar texto. Con un tono más ruido, sin voz, generó *"en la enfermedad."* con baja confianza.
- **Traducción limitada:** entrenada sobre todo desde o hacia el inglés. Otras combinaciones (p. ej. español → francés) funcionan mal.
- **Tamaño vs. calidad:** el modelo base (101M) es claramente peor que el medium (1.02B) en traducción e idiomas con pocos datos.
- **Costo computacional:** la decodificación autorregresiva con beam search es lenta en CPU (≈20 s por audio en nuestro equipo).
- **Datos públicos:** menos datos que Whisper, así que en varios idiomas no alcanza su calidad.

---

## 12. Glosario rápido

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
