
# Identificación y Segmentación de Cultivos de Caña de Azúcar

Este repositorio contiene el código utilizado para entrenar y evaluar un modelo de segmentación semántica que identifica automáticamente áreas de cultivo de caña de azúcar en imágenes aéreas multiespectrales (RGB + NDVI) de campos agrícolas en Guatemala.

El proyecto forma parte del trabajo de graduación **“Segmentación, Binarización y Morfología Aplicadas a la Detección de Zonas Cultivadas de Campos Agrícolas”** presentado en la Facultad de Ingeniería de la **Universidad del Valle de Guatemala**.

---

## Índice
* [Descripción general](#descripción-general)
* [Características principales](#características-principales)
* [Estructura del repositorio](#estructura-del-repositorio)
* [Instalación](#instalación)
* [Uso](#uso)
  * [1. Generar tiles y NDVI](#1-generar-tiles-y-ndvi)
  * [2. Inferencia en nuevas imágenes](#2-inferencia-en-nuevas-imágenes)
* [Conjunto de datos de entrenamiento](#conjunto-de-datos-de-entrenamiento)
  * [Descarga de datos](#descarga-de-datos)
* [Metodología](#metodología)
* [Resultados](#resultados)
  * [Validación cruzada (3 pliegues)](#validación-cruzada-3-pliegues)
  * [Conjunto de prueba final](#conjunto-de-prueba-final)
  * [Comparación de métodos de binarización](#comparación-de-métodos-de-binarización)
* [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
* [Ejemplo de subimagen](#ejemplo-de-subimagen)
* [Ejemplo de inferencia](#ejemplo-de-inferencia)
* [Demostración de funcionamiento](#demostración-de-funcionamiento)
* [Documentación](#documentación)
* [Cómo citar](#cómo-citar)
* [Autores](#autores)

---

## Descripción general

El objetivo de este proyecto es **detectar y segmentar automáticamente zonas cultivadas de caña de azúcar** a partir de imágenes aéreas de alta resolución. Para ello se combina:

* Segmentación semántica con **U-Net + ResNet34** sobre imágenes **RGB + NDVI**.
* **Métodos clásicos de binarización** (Otsu, umbralización adaptativa, umbral simple).
* **Operaciones de morfología matemática** (closing, dilatación) para limpiar y refinar las máscaras.

El resultado es un esquema de procesamiento que produce **máscaras binarias y polígonos** de área cultivada/no cultivada, útiles como insumo para análisis posteriores (detección de fallas, planificación, monitoreo agrícola).

---

## Características principales

* **U-Net con codificador ResNet34** (preentrenado en ImageNet, usado como backbone).
* **Entrada de 4 canales**: `R`, `G`, `B` + `NDVI` para aprovechar información espectral de vegetación.
* **Aumento de datos** durante el entrenamiento: flips, rotaciones, ajustes de brillo/contraste.
* **Validación cruzada con 3 pliegues** en dos escenarios:

  * Codificador ResNet34 congelado.
  * Codificador totalmente entrenable (fine-tuning completo).
* **Comparación de métodos de binarización** aplicados a la salida del modelo:

  * Otsu sobre NDVI.
  * Otsu sobre RGB.
  * Umbralización adaptativa local.
  * Umbral simple (referencia).
* **Postprocesamiento morfológico** (closing + dilatación) para:

  * Rellenar huecos pequeños.
  * Conectar regiones cultivadas fragmentadas.
  * Suavizar contornos de las áreas segmentadas.

---

## Estructura del repositorio

La lógica principal del proyecto está implementada en cuadernos Jupyter dentro de `src/`:

```text
.
├── src/
│   ├── tiles_ndvi.ipynb     # (ndvi_tiles) Preprocesamiento y generación de tiles RGB+NDVI
│   ├── u_net_allp.ipynb     # Definición del modelo U-Net, entrenamiento y validación cruzada
│   └── prediction.ipynb     # Inferencia, binarización comparativa y postprocesamiento morfológico
└── README.md
```

> **Nota:** El repositorio no incluye una carpeta `data/` con imágenes. Los datos se descargan desde un enlace de Google Drive (ver [Descarga de datos](#descarga-de-datos)) y luego se deben configurar las rutas locales en los cuadernos.

---

## Instalación

Se recomienda usar **Python 3.10+** y un entorno virtual (o Conda) para aislar las dependencias.

1. Clonar este repositorio:

```bash
git clone https://github.com/Andrea-gt/PG-2025-21874.git
cd PG-2025-21874
```

2. Crear y activar un entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3. Instalar dependencias principales:

```bash
pip install torch torchvision segmentation-models-pytorch numpy opencv-python scikit-image scikit-learn matplotlib rasterio
```

---

## Uso

> Todos los pasos se realizan ejecutando los cuadernos en `src/`. Una vez descargados los datos desde Google Drive (ver [Descarga de datos](#descarga-de-datos)), organízalos en tu máquina local.

### 1. Generar tiles y NDVI
El cuaderno `src/tiles_ndvi.ipynb` (referido también como `ndvi_tiles`) contiene el flujo para:

1. Registrar y alinear espacialmente las bandas.
2. Calcular el **NDVI**.
3. Apilar `[R, G, B, NDVI]` en un único archivo multicanal.
4. Generar subimágenes (tiles) de tamaño fijo con sus máscaras correspondientes.

> **Importante:** Para poder generar correctamente los tiles con este cuaderno es necesario **contar con el par NIR–NDVI correspondiente** para cada escena de entrada.
> El archivo `test_tiles.pkl` (incluido en el enlace de Google Drive) contiene una lista de tiles que ya tienen asociado dicho par NIR–NDVI y que se utilizaron para este trabajo, se recomienda utilizar este conjunto para testear las predicciones del modelo.

Ajusta las rutas de entrada/salida al inicio del cuaderno según la ubicación de tus imágenes originales.

### 2. Inferencia en nuevas imágenes
El cuaderno `src/prediction.ipynb` permite:

1. Cargar el modelo (`best_model_allp.pth`).
2. Aplicar la segmentación sobre tiles de prueba o nuevas imágenes.
3. Comparar diferentes **métodos de binarización** sobre la salida del modelo:
   * Otsu sobre NDVI.
   * Otsu sobre RGB.
   * Umbralización adaptativa local.
   * Umbral simple (referencia).
4. Aplicar **operaciones morfológicas** (closing, dilatación) para refinar la máscara.
5. Visualizar lado a lado:
   * Imagen RGB.
   * Ground truth.
   * Predicción y binarización + morfología.

---

## Conjunto de datos de entrenamiento

El conjunto de datos utilizado para entrenar y evaluar el modelo se compone por:

* Imágenes aéreas multiespectrales de campos de caña de azúcar en Guatemala.
* Bandas: **RGB** + **NIR**.
* Subimágenes (tiles) con máscaras binarias:

  * `1` = área de cultivo de caña de azúcar.
  * `0` = áreas no agrícolas (caminos, agua, vegetación no relacionada, construcciones, etc.).

Como referencia, la estructura implementada fue:

```text
data/
├── tiles/
│   ├── tile_001.tif  # Subimagen multicanal: R, G, B, NDVI
│   ├── tile_002.tif
│   └── ...
└── masks/
    ├── tile_001.png  # Máscara binaria: 0 = no cultivo, 1 = cultivo
    ├── tile_002.png
    └── ...
```

### Descarga de datos

Los datos necesarios para probar este trabajo (tiles y modelo final documentado) se encuentran en la siguiente carpeta de Google Drive:

> **Enlace a datos:**
> [https://drive.google.com/drive/folders/18io0rwlY4Vn2PuAqmL5s3tcBCDoLSN_D?usp=sharing](https://drive.google.com/drive/folders/18io0rwlY4Vn2PuAqmL5s3tcBCDoLSN_D?usp=sharing)

Dentro de esta carpeta se incluyen:

* `test_tiles.pkl` – archivo pickle con una lista de tiles que **cuentan con el par NIR–NDVI asociado** y que se utilizaron para las pruebas del trabajo.
* `best_model_allp.pth` – pesos del mejor modelo U-Net + ResNet34 entrenado.

> **Disclaimer:** Los datos pertenecen a una **empresa privada**. El enlace de Google Drive **no es de acceso completamente abierto**, por lo que es posible que se tenga que **solicitar acceso** al responsable del proyecto para poder descargar los archivos.
> Por esta razón, los datos brutos no se incluyen directamente dentro del repositorio.

Una vez descargados, se debe debe:

1. Colocar los archivos en la ubicación que prefiera en su máquina.
2. Ajustar las rutas en los cuadernos (`tiles_ndvi.ipynb`, `prediction.ipynb`) para que apunten a la estructura local.

---

## Metodología

A alto nivel, el pipeline implementado en los cuadernos sigue estas etapas:

1. **Adquisición de datos**
   Imágenes multiespectrales RGB + NIR de alta resolución de campos de caña de azúcar.

2. **Preprocesamiento y realce espectral**

   * Registro y alineación de bandas.
   * Cálculo de **NDVI**.
   * Construcción de mosaicos multicanal `[R, G, B, NDVI]`.

3. **Generación de tiles y máscaras**

   * División en subimágenes.
   * Anotación manual de máscaras binarias para separar áreas de **cultivo** vs **no agrícolas**.

4. **Segmentación con U-Net + ResNet34**

   * Uso de **transfer learning** con ResNet34 preentrenado en ImageNet como codificador.
   * Entrenamiento con **3-fold cross-validation** para comparar codificador congelado vs descongelado.

5. **Binarización y morfología matemática**

   * Comparación de métodos clásicos de binarización sobre la salida del modelo (Otsu, adaptativa, umbral simple).
   * Aplicación de **closing + dilatación** para mejorar conectividad de las regiones y limpiar ruido.

---

## Resultados

### Validación cruzada (3 pliegues)

| Configuración        | IoU             | Dice            | Precisión       | Recall          |
| -------------------- | --------------- | --------------- | --------------- | --------------- |
| Encoder congelado    | 0.9434 ± 0.0179 | 0.9707 ± 0.0095 | 0.9705 ± 0.0144 | 0.9712 ± 0.0108 |
| Encoder descongelado | 0.9410 ± 0.0180 | 0.9694 ± 0.0096 | 0.9652 ± 0.0129 | 0.9739 ± 0.0064 |

> En ambos casos se obtienen métricas muy altas, con un ligero cambio en el balance precisión/recall según cómo se entrene el codificador.

### Conjunto de prueba final

| Métrica   | Valor  |
| --------- | ------ |
| IoU       | 0.9738 |
| Dice      | 0.9867 |
| Precisión | 0.9843 |
| Recall    | 0.9892 |

El modelo final (configuración de **encoder descongelado**) logra una superposición muy alta entre la máscara predicha y la máscara de referencia, con un recall elevado (detecta casi todas las áreas de cultivo).

### Comparación de métodos de binarización

| Método      | Exactitud     | Precisión     | Recall        | F1            |
| ----------- | ------------- | ------------- | ------------- | ------------- |
| Otsu (NDVI) | 0.904 ± 0.130 | 0.853 ± 0.264 | 0.771 ± 0.238 | 0.738 ± 0.213 |
| Otsu (RGB)  | 0.791 ± 0.195 | 0.653 ± 0.304 | 0.415 ± 0.292 | 0.356 ± 0.168 |
| Adaptativa  | 0.716 ± 0.087 | 0.366 ± 0.340 | 0.833 ± 0.168 | 0.388 ± 0.233 |

---

## Consideraciones y limitaciones

**Limitaciones actuales**

* Dataset limitado a **regiones y condiciones climáticas específicas**.
* El modelo se entrenó únicamente para **caña de azúcar** (no se ha probado en otros cultivos).
* Sombras, variaciones de iluminación y diferencias fenológicas pueden afectar la calidad de la segmentación.
* Las máscaras de referencia se basan en **anotación manual**, por lo que pueden contener cierto ruido o subjetividad.

**Posibles líneas de trabajo futuro**
* Incluir **datos multitemporales** para capturar distintas etapas de crecimiento del cultivo.
* Explorar arquitecturas más recientes (p. ej. U-Net++ o modelos tipo Vision Transformer para segmentación).

---

## Ejemplo de subimagen

<img width="60%" height="auto" alt="tile_2" src="https://github.com/user-attachments/assets/95ba075a-ffdd-418f-a062-1f2ec190618a" />

---

## Ejemplo de inferencia

<img width="85%" height="auto" alt="pred_03_val" src="https://github.com/user-attachments/assets/61494bf3-d337-4697-b136-5ad3a1f2f3b9" />

# Demostración de funcionamiento
El video demostrativo se encuentra en [/demo/demo.mp4](demo/demo.mp4)

---

## Documentación

El informe final del proyecto está disponible en [/docs/Trabajo_de_Graduación_UVG___A__Ramírez_firmado.pdf](docs/Trabajo_de_Graduación_UVG___A__Ramírez_firmado.pdf)

---

## Autores

**Autora principal**

* **Nombre:** Andrea Ximena Ramírez Recinos
* **Carné:** 21874
* **GitHub:** [@Andrea-gt](https://github.com/Andrea-gt)
