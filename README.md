# Identificación y Segmentación de Cultivos de Caña de Azúcar
Este proyecto implementa un modelo **U-Net con codificador ResNet34** para identificar y segmentar automáticamente cultivos de caña de azúcar a partir de imágenes aéreas multiespectrales (RGB + NDVI) obtenidas en Guatemala.

## Índice
* [Características Principales](#características-principales)
* [Instalación](#instalación)
* [Estructura del conjunto de datos](#estructura-del-conjunto-de-datos)
* [Ejemplo de subimagen](#ejemplo-de-subimagen)
* [Resultados](#resultados)

  * [Validación Cruzada (3 pliegues)](#validación-cruzada-3-pliegues)
  * [Conjunto de Prueba Final](#conjunto-de-prueba-final)
  * [Comparación de Métodos de Binarización](#comparación-de-métodos-de-binarización)
* [Limitaciones](#limitaciones)
* [Ejemplo de Inferencia](#ejemplo-de-inferencia)

## Características principales

* **U-Net con codificador ResNet34** (preentrenado en ImageNet)
* **Entrada de 4 canales**: RGB + NDVI para capturar características de vegetación
* **Aumento de datos**: flips, rotaciones, ajustes de brillo/contraste
* **3-fold cross-validation** para evaluación robusta
* **Comparación de métodos de binarización**: Otsu (RGB/NDVI), adaptativa local, umbral simple
* **Postprocesamiento morfológico**: closing y dilatación

## Instalación

1. Instalar dependencias:

```bash
!pip install torch torchvision segmentation-models-pytorch numpy opencv-python scikit-image scikit-learn matplotlib rasterio
```

## Estructura del conjunto de datos

```
data/
├── tiles/
│   ├── tile_001.tif  (4 canales: R, G, B, NDVI)
│   ├── tile_002.tif
│   └── ...
└── masks/
    ├── tile_001.png  (binario: 0=no cultivo, 1=cultivo)
    ├── tile_002.png
    └── ...
```

## Ejemplo de subimagen

<img width="60%" height="auto" alt="tile_2" src="https://github.com/user-attachments/assets/95ba075a-ffdd-418f-a062-1f2ec190618a" />

## Resultados

### Validación cruzada (3 pliegues)

| Configuración        | IoU             | Dice            | Precisión       | Recall          |
| -------------------- | --------------- | --------------- | --------------- | --------------- |
| Encoder congelado    | 0.9434 ± 0.0179 | 0.9707 ± 0.0095 | 0.9705 ± 0.0144 | 0.9712 ± 0.0108 |
| Encoder descongelado | 0.9410 ± 0.0180 | 0.9694 ± 0.0096 | 0.9652 ± 0.0129 | 0.9739 ± 0.0064 |

### Conjunto de prueba final

| Métrica   | Valor  |
| --------- | ------ |
| IoU       | 0.9738 |
| Dice      | 0.9867 |
| Precisión | 0.9843 |
| Recall    | 0.9892 |

### Comparación de métodos de binarización

| Método      | Exactitud     | Precisión     | Recall        | F1            |
| ----------- | ------------- | ------------- | ------------- | ------------- |
| Otsu (NDVI) | 0.904 ± 0.130 | 0.853 ± 0.264 | 0.771 ± 0.238 | 0.738 ± 0.213 |
| Otsu (RGB)  | 0.791 ± 0.195 | 0.653 ± 0.304 | 0.415 ± 0.292 | 0.356 ± 0.168 |
| Adaptativa  | 0.716 ± 0.087 | 0.366 ± 0.340 | 0.833 ± 0.168 | 0.388 ± 0.233 |

## Limitaciones
* Dataset limitado a una región y un único momento en el tiempo.
* Sombras y variaciones de iluminación pueden afectar el rendimiento.
* Posibles errores en las anotaciones manuales.
* El modelo fue entrenado únicamente en caña de azúcar.

## Ejemplo de inferencia
<img width="85%" height="auto" alt="pred_03_val" src="https://github.com/user-attachments/assets/61494bf3-d337-4697-b136-5ad3a1f2f3b9" />
