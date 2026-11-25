# Audios y textos de referencia de la sección 6

Aquí **no** se publican los audios (`*.wav`, voz del autor) ni el archivo real
`referencias.csv`, porque este último contiene fragmentos de la novela *Noches blancas*
usados como texto de referencia.

- `referencias.ejemplo.csv`: plantilla con el formato esperado y las frases propias
  (cotidianas, inglés, silencio). Para reproducir las pruebas:
  1. Copia la plantilla: `cp audios/referencias.ejemplo.csv audios/referencias.csv`.
  2. Cambia las filas de ejemplo (`corta01`, `larga`, `traduccion01`) por tus propias
     frases, y agrega las que quieras (una fila por audio).
  3. Graba con `scripts/grabar.py` y evalúa con `scripts/evaluar.py` (sección 6.11 del README).
- Los resultados numéricos de nuestra corrida están en `docs/resultados/`. El CSV por
  audio se publica sin las columnas de texto (`referencia` y `salida`).
