# Resultados de las pruebas propias (sección 6.10)

Corrida del 24 de noviembre de 2026: 44 audios, CPU, beam 5, `ctc_weight = 0`.

| Archivo | Contenido |
|---|---|
| `p1_resultados_sin_texto.csv` | Una fila por audio: duración, tiempo, RTF, tokens, confianza, WER y CER. Sin las columnas de texto (`referencia`, `salida`) |
| `p2_ruido.csv` | WER de las 36 frases con ruido blanco (SNR 20, 10 y 5 dB) |
| `p3_beam.csv` | WER y tiempo medio con beam 1 y beam 5 |
| `resumen.json` | Resumen de la corrida (incluye la mejor cabeza de atención) |

Las figuras están en `docs/figuras/`. Cómo reproducir todo: sección 6.11 del README.
