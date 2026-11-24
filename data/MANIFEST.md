# Manifiesto de Modelos, Fechas de Corte y Plantillas

## 1. Modelos LLM Utilizados (fp16 en GPU T4, 4 familias independientes)
| Rol en el protocolo | Identificador Hugging Face | Familia | Parámetros | Fecha de corte |
|---|---|---|---:|---|
| **Herramienta (`P-PAP-v1` / `P-BASE-v1`)** | `Qwen/Qwen2.5-3B-Instruct` | Alibaba Qwen | 3.09B | 2024-09 |
| **Generador `G1` (Train/Val + Roundtrip)** | `microsoft/Phi-3.5-mini-instruct` | Microsoft Phi | 3.82B | 2023-10 |
| **Generador `G2` (Train/Val + Roundtrip)** | `ibm-granite/granite-3.1-2b-instruct` | IBM Granite | 2.53B | 2024-12 |
| **Generador `G3` (Reservado Evaluación)** | `tiiuae/Falcon3-3B-Instruct` | TII Falcon | 3.23B | 2024-12 |

## 2. Garantía de No Contaminación
- Todos los modelos tienen corte de entrenamiento $\le$ diciembre de 2024.
- El conjunto `roundtrip` (100 abstracts) contiene únicamente artículos publicados en arXiv desde **`2025-06-01`** en adelante.
- Términos de uso verificados: licencias abiertas permisivas (Apache-2.0, MIT y TII Falcon License) compatibles con investigación académica.

## 3. Plantillas Versionadas y Hashes SHA-256
- `P-INF-v1.txt`: `bae5eb227690`
- `P-PAP-v1.txt`: `359d4b18484a`
- `P-BASE-v1.txt`: `44b0bad21e7a`
