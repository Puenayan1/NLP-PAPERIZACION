# AbstractBot · interfaz web local

Interfaz web de la herramienta de paperización. Se conecta con el código de `src/paperizar`
(NER híbrido, generación con Qwen2.5-3B-Instruct y verificación) a través de `server.py`.
Todo corre en el computador: el texto nunca se envía a servicios externos.

## Instalación

Copia `server.py` y la carpeta `web/` en la raíz del repositorio `paperizar/`
(junto a `configs/`, `models/`, `prompts/` y `src/`). Después:

```bash
pip install fastapi uvicorn          # además de los requerimientos del proyecto (spacy, torch, transformers…)
python server.py                     # abre http://localhost:8000 en el navegador
```

Opciones útiles:

| Opción | Para qué sirve |
|---|---|
| `--port 8080` | Cambiar el puerto |
| `--no-nli` | No usar la capa NLI (evita descargar el modelo NLI) |
| `--preload` | Cargar el LLM al arrancar en vez de en la primera petición |
| `--ner-model models/ner_v1` | Probar otra variante del NER |

## Requisitos de hardware

El LLM (Qwen2.5-3B-Instruct, fp16) necesita una GPU con unos 8 GB de memoria para responder en segundos.
En CPU funciona, pero cada abstract puede tardar varios minutos.
La primera ejecución descarga el modelo de Hugging Face (unos 6 GB) y luego queda en caché.

## Cómo funciona la conexión

| Endpoint | Qué hace |
|---|---|
| `GET /` | Sirve `web/index.html` |
| `GET /api/health` | Estado del servidor y del modelo |
| `GET /api/examples` | Ejemplos de `web/examples.json` (borradores de `pairs.jsonl` y resúmenes del parquet eLife) |
| `POST /api/ner` | Entidades MEASURE / TECH del NER híbrido, para resaltar el borrador en vivo |
| `POST /api/paperizar` | Genera con `P-PAP-v1`, verifica con `verify_single_rewrite` y, si se perdió, inventó o cambió una entidad, reintenta con los términos protegidos en el prompt |

Si el servidor no responde, la página funciona en **modo demo**, con un NER y una reescritura
simulados en el navegador, y lo indica en la barra superior.

## Capas del semáforo

| Capa | Fuente | Verde | Amarillo |
|---|---|---|---|
| Entidades preservadas | NER híbrido + `entity_present` | cobertura ≥ 0,95 | ≥ 0,85 |
| Nada inventado | precisión de entidades | ≥ 0,95 | ≥ 0,85 |
| Unidades intactas | `detect_units_changed` y MEASURE perdidas | ninguna cambia | — |
| Similitud semántica | vectores del modelo spaCy | ≥ 0,90 | ≥ 0,80 |
| Implicación (NLI) | `cross-encoder/nli-deberta-v3-xsmall`, en ambos sentidos | ≥ 0,80 | ≥ 0,60 |
| Registro académico | Flesch (textstat) y `configs/flesch_bounds.json` | baja y queda en [p10, p90] | baja |
| No es una copia | ROUGE-L entrada contra salida | ≤ 0,60 | ≤ 0,80 |

Los umbrales están en `THRESHOLDS`, al inicio de `server.py`.

## Nota sobre el modelo NER descargado de Drive

Al descargar la carpeta en zip, Google Drive renombra los archivos `cfg` a `cfg.txt`, y spaCy no carga
el modelo. `server.py` lo detecta y restaura los nombres al arrancar.
