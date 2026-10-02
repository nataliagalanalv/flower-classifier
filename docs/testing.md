# Tests del Proyecto
 
Resumen de los tests automatizados del clasificador de flores: qué se decidió testear, por qué, qué herramientas se usaron y cómo evolucionaron junto con el resto del proyecto. Para los resultados del modelo en sí, ver [`RESULTS.md`](./RESULTS.md) y [`ANALYSIS.md`](./ANALYSIS.md).
 
## Objetivo
 
Los tests no evalúan si el modelo "acierta" clasificando flores — esa evaluación ya vive, de forma manual y documentada, en `evaluate.py` y `confusion_analysis.py`. Los tests automatizados protegen la **capa de ingeniería**: que la arquitectura de los modelos, la congelación/descongelación de capas en transfer learning, y la lógica auxiliar (como el mapeo de índice a nombre de flor) se comporten exactamente como están diseñadas, cada vez que alguien modifica el código.
 
## Filosofía: qué se testea y qué no
 
**No se testea el bucle de entrenamiento** (`train.py`, `train_feature_extract.py`, `train_finetune.py`). Motivos:
- Es lento (minutos u horas), así que nadie lo ejecutaría antes de cada commit.
- No es determinista: los pesos cambian en cada ejecución (no se fijó una semilla aleatoria en este proyecto, ver `ANALYSIS.md`).
- Depende de tener el dataset descargado y, idealmente, GPU disponible.
**Sí se testea:**
- **Arquitectura de los modelos** (forma de salida dada una forma de entrada).
- **Congelación/descongelación de parámetros** en transfer learning, porque un error aquí no lanza ninguna excepción — el modelo simplemente entrenaría mal, en silencio, y tardaríamos horas en darnos cuenta sin un test.
- **Lógica pura sin efectos secundarios** (el mapeo índice → nombre de flor), porque es determinista y fácil de aislar.
La idea general: un test suite no busca cobertura del 100%, busca proteger las partes donde un error sería caro o invisible a simple vista.
 
## Un refactor que nació de testear: `class_names.py`
 
La función `idx_to_name` (que traduce el índice numérico 0–101 de una clase a su nombre de flor) vivía originalmente **dentro de `confusion_analysis.py`**, mezclada con la carga del modelo y la inferencia sobre todo el test set.
 
Importar esa función para testearla habría disparado, como efecto secundario, la carga completa de un modelo de deep learning y la evaluación de miles de imágenes — justo lo que la filosofía de testing de este proyecto evita. Por eso se extrajo a su propio módulo, `class_names.py`, que solo contiene la carga del JSON de nombres y la función de mapeo. `confusion_analysis.py` pasó a importarla desde ahí.
 
Esta es una ilustración concreta de cómo el testing puede influir en el diseño del código, no solo verificarlo después.
 
## Un bug real, encontrado por un test
 
Al escribir `test_offset_is_applied_correctly`, se asumió que el índice de torchvision `49` correspondía a "petunia", porque así aparecía etiquetado en el análisis de resultados ("petunia, clase 50"). El test falló: `idx_to_name(49)` devolvía `"common dandelion"`, no `"petunia"`.
 
La causa no estaba en `idx_to_name` sino en el propio test: las etiquetas "clase 50" de `RESULTS.md` ya imprimían directamente el índice crudo de torchvision (sin ajuste adicional), y el ajuste de +1 para consultar el JSON (`cat_to_name["51"]`) ya ocurre **dentro** de `idx_to_name`. Es decir, para obtener "petunia" había que llamar a `idx_to_name(50)`, no `idx_to_name(49)`. Al corregir el valor de entrada del test, el suite completo pasó.
 
Este caso se documenta explícitamente porque ilustra el valor real de un test: no solo confirma que el código funciona, también obliga a verificar que las propias suposiciones sobre los datos son correctas.
 
## Los tests, archivo por archivo
 
### `tests/test_model.py` — `SimpleCNN` (CNN desde cero)
 
Verifica que, dado un tensor de entrada `[batch, 3, 128, 128]`, el modelo devuelve `[batch, num_classes]`:
 
| Test | Qué comprueba |
| --- | --- |
| `test_output_shape_matches_num_classes` | Caso por defecto: batch 1, 102 clases → `[1, 102]`. |
| `test_output_shape_with_different_batch_size` | Batch de 4 → `[4, 102]`; confirma que generaliza a cualquier tamaño de batch. |
| `test_output_shape_with_different_num_classes` | `num_classes=10` → `[1, 10]`; confirma que la capa final se adapta, no está fija a 102. |
 
### `tests/test_model_transfer.py` — ResNet18 (feature extraction y fine-tuning)
 
El archivo más importante del suite: protege la parte del código donde un fallo sería invisible sin un test.
 
| Test | Qué comprueba |
| --- | --- |
| `test_feature_extractor_only_trains_fc_layer` | En feature extraction, `requires_grad` es `True` solo para los parámetros de `fc.`, `False` para el resto. |
| `test_feature_extractor_trainable_param_count` | El número exacto de parámetros entrenables: 52.326. Un número específico a propósito — si cambia sin querer qué capas se congelan, este test lo detecta aunque el anterior no lo hiciera. |
| `test_finetune_unfreezes_layer4_and_fc_only` | En fine-tuning, `requires_grad` es `True` para `layer4.` y `fc.`, `False` para `conv1.`, `bn1.`, `layer1.`, `layer2.`, `layer3.`. Descongelar la capa equivocada no lanza ningún error — el entrenamiento seguiría funcionando, solo que peor, y esa diferencia solo se notaría comparando métricas horas después. |
| `test_output_shape_matches_num_classes` | Forma de salida del modelo de fine-tuning con entrada 224x224 → `[1, 102]`. |
 
> Nota técnica: este archivo necesita conexión a internet la primera vez que se ejecuta, porque `resnet18(weights=...)` descarga los pesos preentrenados de ImageNet si no están en caché local.
 
### `tests/test_class_names.py` — mapeo índice → nombre de flor
 
| Test | Qué comprueba |
| --- | --- |
| `test_known_index_returns_a_name` | Un índice válido devuelve un nombre real, no el valor de fallback. |
| `test_offset_is_applied_correctly` | El desplazamiento de índice (torchvision 0–101 → claves JSON "1"–"102") se aplica correctamente. Ver el caso del bug documentado arriba. |
| `test_unknown_index_falls_back_to_placeholder` | Un índice fuera de rango no lanza excepción, devuelve `"clase_<idx>"`. |
 
## Cómo se resuelven los imports
 
Los tests importan los módulos del proyecto como un paquete instalado: `from flower_classifier.model import SimpleCNN`, `from flower_classifier.model_transfer import ...`, `from flower_classifier.class_names import idx_to_name`.
 
Esto es resultado de un cambio posterior a la creación original de los tests: el proyecto se reestructuró como un paquete Python instalable (`src/flower_classifier/`, con `pyproject.toml`), instalado en modo editable:
 
```powershell
pip install -e .
```
 
## Cómo ejecutar los tests
 
Desde la raíz del proyecto, con el entorno virtual activado:
 
```powershell
pytest -v
```
 
Resultado esperado: 10 tests recogidos, los 10 en verde, en pocos segundos — sin GPU, sin descargar el dataset completo, sin entrenar nada.