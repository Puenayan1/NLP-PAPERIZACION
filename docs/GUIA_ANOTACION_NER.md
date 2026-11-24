# Guía de Anotación NER: Etiquetas `MEASURE` y `TECH`

## 1. Definición de Etiquetas
- **`MEASURE`**: Cantidades numéricas técnicas cuantificables que deben preservarse intactas en la reescritura.
  - *Incluye*: Cifra + unidad (`12 ms`, `2.4 GHz`, `50 mW.`, `1080 fps`, `3.2 dB`), porcentajes (`95.3%`, `12 %`), escalas en palabras (`3.4 million`, `10 thousand`) y decimales de métricas (`0.941`).
  - *Excluye*: Años (`2024`), numeración de secciones/pasos (`(1)`, `first`, `two stages`), o números de versión pegados a un nombre propio (`ResNet50` o `Raspberry Pi 4` van completos en `TECH`).
- **`TECH`**: Nombres propios o identificadores técnicos de ingeniería.
  - *Incluye*: Modelos y arquitecturas (`MobileNetV2`, `YOLOv8`, `Transformer`), datasets (`SDNET2018`, `ImageNet`), hardware/plataformas (`Raspberry Pi 4`, `RTX 2080 Ti`, `FPGA`), métricas con nombre (`F1-score`, `ROUGE-L`, `PSNR`, `BER`, `SNR`), siglas en mayúsculas (`CNN`, `OFDM`, `MIMO`, `PID`) y términos en `CamelCase`.
  - *Excluye*: Sustantivos genéricos en minúscula (`neural network`, `sensor`, `algorithm`, `controller`, `dataset`).

## 2. Reglas de Bordes (Casos Límite)
1. **Unidades al final de oración**: Si el token de unidad trae punto final pegado (`50 W.`), el tramo anotado excluye el punto final de la oración (`50 W`), salvo abreviaturas internas.
2. **Porcentajes partidos**: Tanto `95.3%` como `95.3 %` se etiquetan como un único tramo `MEASURE`.
3. **Nombres compuestos con números**: Si un número forma parte del nombre del modelo, estándar o placa (`IEEE 802.11ax`, `Raspberry Pi 4`, `SDNET2018`, `FP16`), toda la expresión es `TECH`, nunca se parte el número como `MEASURE`.
4. **Rangos**: En expresiones como `10 to 20 MHz`, si el primer número no lleva unidad explícita, se etiqueta `20 MHz` como `MEASURE` (o cada cifra con unidad si ambas la tienen: `10 MHz to 20 MHz`).
