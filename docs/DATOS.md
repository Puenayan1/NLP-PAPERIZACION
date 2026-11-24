# Gestión de Datos y Particiones

## 1. Resumen de Conjuntos Generados (Semilla 42)
- **`corpus_A`**: 3300 abstracts filtrados de ingeniería en inglés (`train`: 2310, `val`: 495, `test`: 495).
- **`roundtrip`**: 100 abstracts recientes posteriores al `2025-06-01` (solapamiento con `corpus_A`: 0 registros).
- **`ner_anotado` (candidatos base)**: 80 abstracts extraídos exclusivamente de `corpus_A` (40 `train`, 20 `dev`, 20 `test`).

## 2. Distribución por Categoría de arXiv
| Categoría | Descripción | `corpus_A` | `roundtrip` |
|---|---|---:|---:|
| `eess.SP` | signal processing | 920 | 18 |
| `eess.SY` | systems and control | 170 | 13 |
| `eess.IV` | image and video processing | 510 | 16 |
| `cs.RO` | robotics | 716 | 15 |
| `cs.AR` | hardware architecture | 332 | 21 |
| `cs.NI` | networking and internet architecture | 652 | 17 |

## 3. Filtros Aplicados y Licencias
- Longitud entre 80 y 250 palabras, máximo 2 signos `$`, sin duplicados exactos ni normalizados, y mínimo 2 entidades numéricas con unidad o porcentaje.
- Se conserva el campo `license` individual declarado por registro en arXiv; históricos obtenidos vía `GFissore/arxiv-abstracts-2021` (CC0), `nick007x/arxiv-papers` y la API oficial de arXiv.
