# Resultados del Proyecto: Clasificador de Flores (Oxford 102 Flowers)
 
Registro completo de todos los experimentos realizados, con sus métricas. Para la interpretación y conclusiones, ver [`ANALYSIS.md`](./ANALYSIS.md).
 
## Entorno
 
- Python 3.14.7
- PyTorch 2.14.0 (build `+cu126`, CUDA 12.6)
- GPU: NVIDIA GeForce RTX 4060 Laptop
- Dataset: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (`torchvision.datasets.Flowers102`)
## Dataset
 
| Split | Nº de imágenes |
| --- | --- |
| train | 1020 |
| val | 1020 |
| test | 6149 |
 
102 clases, entre 40 y 258 imágenes por clase en total (dataset desbalanceado).
 
## Experimento 1 — CNN desde cero (`SimpleCNN`)
 
Arquitectura: 4 bloques convolucionales (32→64→128→256 filtros, kernel 3x3, BatchNorm, ReLU, MaxPool) + Dropout(0.5) + capa lineal final. Entrada 128x128. Optimizer Adam (`lr=0.001`).
 
| Configuración | Épocas | Val Acc máxima | Train Acc final | Gap Train-Val |
| --- | --- | --- | --- | --- |
| Baseline (solo `RandomHorizontalFlip`, sin BatchNorm/weight_decay) | 20 | ~25.0% | 84.4% | ~61 pts |
| + augmentation avanzado + weight_decay(1e-4) + BatchNorm | 20 | 33.1% | 62.2% | ~29 pts |
| + augmentation avanzado + weight_decay(1e-4) + BatchNorm | 30 | 37.65% | 74.9%–75.0% | ~29-40 pts |
 
## Experimento 2 — ResNet18 Feature Extraction
 
ResNet18 preentrenada en ImageNet, backbone completo congelado, solo se entrena la capa `fc` (102 clases). Entrada 224x224. Optimizer Adam sobre `model.fc.parameters()`.
 
| Configuración | Épocas | Parámetros entrenables | Val Acc máxima | Train Acc final |
| --- | --- | --- | --- | --- |
| `lr=0.001`, sin weight_decay | 20 | 52,326 (0.47%) | 79.22% | 98.24% |
| `lr=0.001`, weight_decay=1e-4 | 20 | 52,326 (0.47%) | 79.12% | 97.84% |
 
## Experimento 3 — ResNet18 Fine-tuning (`layer4` + `fc`)
 
Se descongela adicionalmente el bloque `layer4`. Optimizer Adam con dos grupos de learning rate: `layer4` a `1e-4`, `fc` a `1e-3`.
 
| Configuración | Épocas | Parámetros entrenables | Val Acc máxima | Train Acc final |
| --- | --- | --- | --- | --- |
| Sin weight_decay | 20 | 8,446,054 (~75%) | **90.78%** | 100% (desde época 8) |
| weight_decay=1e-4 | 20 | 8,446,054 (~75%) | 90.00% | 100% (desde época 10) |
 
**Modelo final seleccionado:** fine-tuning sin weight_decay (`best_resnet18_finetune.pth`), por ser la configuración con mayor Val Acc de todo el proyecto.
 
## Evaluación final sobre el split de `test`
 
Realizada una única vez, con el modelo `best_resnet18_finetune.pth`, sin ajustes posteriores.
 
| Métrica | Valor |
| --- | --- |
| Test Loss | 0.5144 |
| **Test Accuracy** | **87.87%** |
| Accuracy media por clase (no ponderada, 102 clases) | 89.51% |
 
## Top 10 confusiones más frecuentes (test set)
 
| Clase real | Clase predicha | Nº de veces |
| --- | --- | --- |
| petunia (50) | hibiscus (82) | 25 |
| petunia (50) | garden phlox (31) | 19 |
| petunia (50) | morning glory (75) | 11 |
| primula (52) | wallflower (45) | 10 |
| camellia (95) | mallow (96) | 9 |
| sweet william (29) | garden phlox (31) | 8 |
| snapdragon (10) | trumpet creeper (100) | 7 |
| passion flower (76) | great masterwort (37) | 7 |
| lotus lotus (77)* | water lily (72) | 7 |
| cyclamen (87) | lotus lotus (77)* | 7 |
 
*\*Nombre tal cual aparece en el archivo de mapeo `cat_to_name.json` (fuente no oficial de la comunidad); posible inconsistencia en ese archivo, ver nota de fiabilidad en `ANALYSIS.md`.*
 
## Las 15 clases más difíciles (menor accuracy en test)
 
| Clase | Accuracy | Imágenes en test |
| --- | --- | --- |
| sweet pea (3) | 44.44% | 36 |
| petunia (50) | 55.88% | 238 |
| canterbury bells (2) | 60.00% | 20 |
| canna lily (89) | 61.29% | 62 |
| sweet william (29) | 61.54% | 65 |
| primula (52) | 61.64% | 73 |
| camellia (95) | 61.97% | 71 |
| azalea (71) | 63.16% | 76 |
| hibiscus (82) | 65.77% | 111 |
| corn poppy (25) | 66.67% | 21 |
| columbine (83) | 66.67% | 66 |
| sword lily (42) | 67.27% | 110 |
| snapdragon (10) | 70.15% | 67 |
| clematis (81) | 73.91% | 92 |
| mexican petunia (97) | 79.03% | 62 |
 
## Resumen comparativo final
 
| Enfoque | Métrica reportada | Valor |
| --- | --- | --- |
| CNN desde cero (mejor config., 30 épocas) | Val Acc | 37.65% |
| ResNet18 Feature Extraction | Val Acc | 79.22% |
| ResNet18 Fine-tuning | Val Acc | 90.78% |
| **ResNet18 Fine-tuning** | **Test Acc (definitivo)** | **87.87%** |
 
Artefactos generados: `outputs/confusion_matrix.png` (matriz de confusión completa, 102x102), checkpoints `.pth` de todos los experimentos.
 