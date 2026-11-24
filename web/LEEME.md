# AbstractBot · Presentación Interactiva y Demo en Vivo

Entorno web de presentación académica y demostración interactiva de **Paperización Verificable de Resúmenes de Ingeniería en Inglés**. Se conecta en local con `src/paperizar` (NER híbrido `ner_hybrid_v1`, generación restringida con `Qwen2.5-3B-Instruct` y verificación multicapa con `DeBERTa-v3 NLI`) a través de `server.py`.

## Ejecución Rápida

En Windows PowerShell (automatizado):

```powershell
.\run_abstractbot.ps1
```

O manualmente desde la raíz del repositorio:

```bash
pip install fastapi uvicorn
python server.py                     # abre http://localhost:8000
```

Opciones útiles de `server.py`:

| Opción | Para qué sirve |
|---|---|
| `--port 8080` | Cambiar el puerto HTTP |
| `--no-nli` | Desactivar la capa NLI (evita cargar `nli-deberta-v3-xsmall`) |
| `--preload` | Cargar el LLM al arrancar el servidor en vez de en la primera petición |
| `--ner-model models/ner_v1` | Probar otra variante del reconocedor NER |

## Estructura de la Presentación Interactiva (6 Pestañas)

1. **Inicio y Problema (`#1-problema`)**: Portada del artículo, equipo investigador, KPIs principales, los 4 riesgos de usar LLMs genéricos en resúmenes científicos y premisa ética/arquitectónica.
2. **Arquitectura y Datos (`#2-arquitectura`)**: Ciclo de aseguramiento ML de Ashmore et al. en 4 etapas, diagrama de tubería experimental vs. despliegue local, Tabla I de conjuntos disjuntos y control de contaminación temporal (`>= 2025-06-01`).
3. **Metodología (`#3-metodologia`)**: Separación de 4 familias abiertas de LLM, hallazgo del piloto (`SmolLM2` → `Granite-3.1`), Tabla II de la compuerta sintética (`593/900`), arquitectura `ner_hybrid_v1` y contraste `P-PAP-v1` vs. `P-BASE-v1`.
4. **Caso COIL Paso a Paso (`#4-trazabilidad`)**: Recorrido interactivo del caso real `arXiv:2611.12223` (`rt_000_G1_L1` y `rt_000_G1_L3`) con botón directo para enviarlo al Demo en Vivo.
5. **Resultados Experimentales (`#5-resultados`)**: Sub-pestañas interactivas para Auditor NER (`ner_test`), Evaluación Sintética Pareada (`N = 593`), Borradores Humanos Reales (`gold_real`, `N = 12`), Evaluación Humana Ciega (`N = 40`) y Conclusiones / Trabajo Futuro.
6. **Demo en Vivo (`#6-demo`)**: Espacio interactivo conectado a `server.py` con selector rápido de 8 casos reales del estudio (`web/examples.json`), resaltado NER en tiempo real (`POST /api/ner`) y semáforo auditable de 7 capas (`POST /api/paperizar`).

### Controles para Exposición en Salón de Clase
- **Teclado**: Flechas `←` y `→` para cambiar entre pestañas (se desactivan automáticamente al escribir dentro del editor).
- **Modo Proyector**: Botón en la barra superior que amplía la escala tipográfica y el ancho útil para videobeam.
- **Pantalla Completa y Tema**: Alternancia entre modo claro/oscuro y pantalla completa nativa del navegador.
