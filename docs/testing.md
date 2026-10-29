# Tests del Proyecto
 
Resumen de los tests automatizados del clasificador de flores: qué se decidió testear, por qué, qué herramientas se usaron y cómo evolucionaron junto con el resto del proyecto. Para los resultados del modelo en sí, ver [results.md](./results.md) y [analysis_results.md](./analysis_results.md).
 
## Objetivo
 
Los tests no evalúan si el modelo "acierta" clasificando flores — esa evaluación ya vive, de forma manual y documentada, en `evaluate.py` y `confusion_analysis.py`. Los tests automatizados protegen la **capa de ingeniería**: que la arquitectura de los modelos, la congelación/descongelación de capas en transfer learning, los bucles de entrenamiento y evaluación, las transformaciones de imagen, la predicción de una imagen, la configuración de `train.py`, los endpoints de la API web y la lógica auxiliar (como el mapeo de índice a nombre de flor) se comporten exactamente como están diseñados, cada vez que alguien modifica el código.
 
## Filosofía: qué se testea y qué no
 
**No se entrena de verdad en los tests.** Entrenar un modelo con el dataset real:
 
- Es lento (minutos u horas), así que nadie lo ejecutaría antes de cada commit.
- No es determinista: los pesos cambian en cada ejecución (no se fijó una semilla aleatoria en este proyecto, ver `analysis_results.md`).
- Depende de tener el dataset descargado y, idealmente, GPU disponible.

Pero sí se testea **la lógica** del entrenamiento, con datos de juguete y piezas falsas que hacen los tests rápidos y deterministas: el bucle de una época y la evaluación (`engine.py`) y la conexión de la configuración con los argumentos y los checkpoints (`train.py`).
 
**Sí se testea:**
 
- **Arquitectura de los modelos** (forma de salida dada una forma de entrada).
- **Congelación/descongelación de parámetros** en transfer learning, porque un error aquí no lanza ninguna excepción — el modelo simplemente entrenaría mal, en silencio, y tardaríamos horas en darnos cuenta sin un test.
- **Los bucles de entrenamiento y evaluación**, porque errores como olvidar `optimizer.step()`, dejar el modelo en el modo equivocado (`train`/`eval`) o promediar mal la pérdida no lanzan ninguna excepción y solo se notan en las métricas.
- **Las transformaciones de imagen**, porque entrenar y evaluar con normalizaciones distintas degrada el modelo sin ningún aviso.
- **La predicción de una imagen** (`predict_top_k`), incluidas las imágenes que suben los usuarios (escala de grises, con transparencia).
- **La configuración de `train.py`**: registro de modelos, argumentos de línea de comandos y guardado de checkpoints.
- **La API web (`app/main.py`)**: qué responde cada endpoint, que rechaza con el código correcto lo que no es una imagen válida o un `top_k` fuera de rango, y que el servidor **no arranca** si el modelo no se puede cargar.
- **Lógica pura sin efectos secundarios** (el mapeo índice → nombre de flor), porque es determinista y fácil de aislar.

La idea general: un test suite no busca cobertura del 100%, busca proteger las partes donde un error sería caro o invisible a simple vista.
 
## Técnicas: datos de juguete y piezas falsas
 
Para probar código que normalmente depende de datos reales, GPU o entrenamientos largos, los tests sustituyen esas piezas por versiones diminutas cuyo comportamiento se conoce de antemano:
 
- **Datos de juguete:** 32 puntos de dos características, separables con una recta (la clase es el signo de la primera característica). Un bucle correcto tiene que aprenderlos. Se entregan con un `DataLoader` normal, igual que los datos reales.
- **Modelos falsos:** `FixedLogits` (devuelve siempre los mismos 102 valores, para probar `predict_top_k` sin cargar la red real) y un modelo lineal con pesos a cero cuya salida es constante (para calcular a mano la pérdida y la precisión esperadas).
- **Espías:** `GradModeSpy` es un modelo que anota, en cada `forward`, si autograd estaba activo. Así se comprueba que `evaluate` va con `@torch.no_grad()`.
- **`monkeypatch`:** fixture de pytest que sustituye algo (una función, un diccionario, `sys.argv`) solo durante un test y lo restaura al terminar. Permite ejecutar `main()` de `train.py` con datos falsos y pasos de entrenamiento falsos.
- **`tmp_path`:** fixture de pytest que da una carpeta temporal nueva por test, para guardar checkpoints sin ensuciar el proyecto.
- **`@pytest.mark.parametrize`:** ejecuta la misma función con varios valores; cada uno es un test distinto.
- **`skipif`:** salta los tests de GPU si no hay CUDA, sin darlos por fallidos.
- **`TestClient` de FastAPI:** hace peticiones a la aplicación sin levantar un servidor ni abrir puertos. Usado dentro de un `with`, ejecuta también el arranque y el apagado (`lifespan`), igual que lo hace `uvicorn`.
- **Imágenes creadas en memoria:** los tests de la API generan sus imágenes con Pillow y `io.BytesIO`, sin dejar archivos en el repositorio. Para la imagen corrupta se recorta a la mitad una imagen válida.
- **`monkeypatch.setenv` y `pytest.raises`:** `setenv` cambia la variable `CHECKPOINT_PATH` solo durante un test; `pytest.raises` comprueba que un bloque lanza una excepción concreta (y, con `match`, que el mensaje contiene un texto).

## Un refactor que nació de testear: `class_names.py`
 
La función `idx_to_name` (que traduce el índice numérico 0–101 de una clase a su nombre de flor) vivía originalmente **dentro de `confusion_analysis.py`**, mezclada con la carga del modelo y la inferencia sobre todo el test set.
 
Importar esa función para testearla habría disparado, como efecto secundario, la carga completa de un modelo de deep learning y la evaluación de miles de imágenes — justo lo que la filosofía de testing de este proyecto evita. Por eso se extrajo a su propio módulo, `class_names.py`, que solo contiene la carga del JSON de nombres y la función de mapeo. `confusion_analysis.py` pasó a importarla desde ahí.
 
Esta es una ilustración concreta de cómo el testing puede influir en el diseño del código, no solo verificarlo después.
 
## Un bug real, encontrado por un test
 
Al escribir `test_offset_is_applied_correctly`, se asumió que el índice de torchvision `49` correspondía a "petunia", porque así aparecía etiquetado en el análisis de resultados ("petunia, clase 50"). El test falló: `idx_to_name(49)` devolvía `"common dandelion"`, no `"petunia"`.
 
La causa no estaba en `idx_to_name` sino en el propio test: las etiquetas "clase 50" de `results.md` ya imprimían directamente el índice crudo de torchvision (sin ajuste adicional), y el ajuste de +1 para consultar el JSON (`cat_to_name["51"]`) ya ocurre **dentro** de `idx_to_name`. Es decir, para obtener "petunia" había que llamar a `idx_to_name(50)`, no `idx_to_name(49)`. Al corregir el valor de entrada del test, el suite completo pasó.
 
Este caso se documenta explícitamente porque ilustra el valor real de un test: no solo confirma que el código funciona, también obliga a verificar que las propias suposiciones sobre los datos son correctas.
 
## Un test que no comprobaba lo que decía
 
La primera versión del test de `evaluate` comprobaba que, tras llamarla, `p.grad is None` para todos los parámetros, y se presentaba como la prueba de que `evaluate` va con `@torch.no_grad()`. No lo era: `.grad` solo se rellena al llamar a `backward()`, así que ese test habría pasado igual aunque `evaluate` no tuviera `no_grad`, siempre que no llamara a `backward()`. Lo que comprobaba de verdad era que `evaluate` no acumula gradientes.
 
Se sustituyó por `test_evaluate_runs_without_autograd`, que usa un modelo "espía" (`GradModeSpy`) que anota si autograd estaba activo durante cada `forward`. Al quitar `@torch.no_grad()` de `evaluate`, el nuevo test falla con `[True, True, True, True]` en la lista del espía (ver la sección de más abajo), y el antiguo no habría fallado.
 
La lección es la misma que la del caso anterior: un test debe poder fallar por la razón correcta, y conviene comprobarlo.
 
## Una decisión de diseño que los tests fijan: fallar al arrancar

Si el checkpoint no existe o no se puede cargar, la aplicación **no arranca**: la excepción sale de `lifespan` y `uvicorn` termina con `Application startup failed`. Es deliberado. Un servicio que arranca sin modelo estaría vivo pero sería inservible, mientras que uno que falla al arrancar lo detecta enseguida cualquier plataforma de contenedores (que puede reiniciarlo o marcarlo como fallido).

Los dos tests de arranque lo fijan: comprueban, con `pytest.raises`, que entrar en el `TestClient` lanza `FileNotFoundError` (checkpoint inexistente) o el error del cargador (checkpoint que no se puede cargar). La excepción salta al entrar, antes de poder hacer ninguna petición, así que estos tests no usan la fixture `client`, que ya arranca la aplicación con un checkpoint válido. La comprobación de `model_loaded` en `/health` queda como red de seguridad.

## Los tests, archivo por archivo
 
### `tests/test_model.py` — `SimpleCNN` (CNN desde cero)
 
Verifica que, dado un tensor de entrada `[batch, 3, 128, 128]`, el modelo devuelve `[batch, num_classes]`:
 
| Test | Qué comprueba |
| --- | --- |
| `test_output_shape_matches_num_classes` | Caso por defecto: batch 1, 102 clases → `[1, 102]`. |
| `test_output_shape_with_different_batch_size` | Batch de 4 → `[4, 102]`; confirma que generaliza a cualquier tamaño de batch. |
| `test_output_shape_with_different_num_classes` | `num_classes=10` → `[1, 10]`; confirma que la capa final se adapta, no está fija a 102. |
 
### `tests/test_model_transfer.py` — ResNet18 (feature extraction y fine-tuning)
 
Protege la parte del código donde un fallo sería invisible sin un test.
 
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
 
### `tests/test_engine.py` — bucles de entrenamiento y evaluación
 
Usa el problema de juguete (32 puntos separables, un `nn.Linear(2, 2)`, SGD) y un modelo de pesos a cero para los valores exactos. No necesita dataset, GPU ni internet.
 
| Test | Qué comprueba |
| --- | --- |
| `test_train_one_epoch_updates_weights` | Tras una época, al menos un parámetro cambió. Si el entrenamiento no cambiara ninguno (por ejemplo, olvidando `optimizer.step()`), falla. |
| `test_evaluate_does_not_update_weights` | `evaluate` deja todos los parámetros idénticos. Si los tocara, estaría ajustando el modelo con datos de validación o de test. |
| `test_metrics_are_in_valid_range` | La pérdida es ≥ 0 y la precisión está entre 0 y 1. |
| `test_train_one_epoch_sets_train_mode` | Un modelo en modo `eval` pasa a `train` al llamar a `train_one_epoch`. Equivocarse con el modo no lanza error, pero cambia el comportamiento de Dropout y BatchNorm. |
| `test_evaluate_sets_eval_mode` | Un modelo en modo `train` pasa a `eval` al llamar a `evaluate`. |
| `test_model_can_overfit_toy_data` | Tras 30 épocas, la pérdida baja y la precisión llega a 1.0. Es la comprobación de conjunto: si el bucle, la pérdida y el optimizador están bien conectados, un problema tan fácil se aprende. |
| `test_evaluate_runs_without_autograd` | Con el modelo espía, todas las llamadas a `forward` de `evaluate` se hacen con autograd desactivado (`@torch.no_grad()`). |
| `test_evaluate_returns_exact_loss_and_accuracy` | Con un modelo de salida constante, 5 muestras en lotes de 2, 2 y 1, la pérdida y la precisión coinciden con los valores calculados a mano (≈ 0,5269 y 0,8). Los lotes desiguales detectan el error de promediar las pérdidas de cada lote en vez de calcular la media por muestra (que daría ≈ 0,4603). |
| `test_train_one_epoch_returns_exact_loss_and_accuracy` | Lo mismo para `train_one_epoch`, usando un optimizador con `lr=0.0` para que el modelo no cambie durante la época. |
 
### `tests/test_transforms.py` — transformaciones de imagen
 
| Test | Qué comprueba |
| --- | --- |
| `test_eval_transform_outputs_square_tensor` (128, 224) | A partir de una imagen no cuadrada (300×200), la salida es `[3, img_size, img_size]`. |
| `test_train_transform_outputs_square_tensor` (128, 224) | Lo mismo para la transformación de entrenamiento (con *augmentation*). |
| `test_eval_transform_is_deterministic` | La transformación de evaluación aplicada dos veces a la misma imagen da el mismo tensor. Si fuera aleatoria, los resultados de test no serían repetibles. |
| `test_train_transform_is_random` | Aplicada 10 veces, la transformación de entrenamiento da al menos un resultado distinto (se usa una imagen con ruido, porque en una imagen lisa los recortes y giros serían indistinguibles). |
| `test_eval_transform_applies_imagenet_normalization` (0, 255) | Una imagen toda negra o toda blanca da el valor `(valor − media) / desviación` por canal, usando las constantes `IMAGENET_MEAN` e `IMAGENET_STD` del propio código. |
| `test_eval_transform_matches_known_imagenet_values` (255, 0) | Lo mismo, pero con los valores escritos a mano (por ejemplo, 2,2489 / 2,4286 / 2,6400 para el blanco). Complementa al anterior: si alguien se equivocara al escribir una constante, el test anterior (que usa las mismas) no lo detectaría, y este sí. |
 
### `tests/test_predict.py` — `predict_top_k`
 
Usa `FixedLogits`, un modelo falso que siempre devuelve los mismos logits (clase 7 > clase 3 > clase 50), para probar la lógica sin cargar la red real ni el checkpoint.
 
| Test | Qué comprueba |
| --- | --- |
| `test_top_predictions_are_ordered_by_probability` | Con `k=3`, los nombres salen en el orden esperado (clases 7, 3, 50) y traducidos con `idx_to_name`. |
| `test_returns_k_results` (1, 5) | Devuelve exactamente `k` resultados. |
| `test_probabilities_are_valid_and_descending` | Las probabilidades están entre 0 y 1, ordenadas de mayor a menor, y su suma no supera 1. |
| `test_accepts_non_rgb_images` (`L`, `RGBA`) | Imágenes en escala de grises y con transparencia no fallan. Sin el `convert("RGB")` de `predict_top_k`, el número de canales no cuadra con las tres medias de la normalización. |
| `test_predict_runs_on_device` (`cpu`, `cuda`) | La predicción funciona en CPU y, si hay GPU, también en CUDA. |
 
### `tests/test_train.py` — configuración y conexión de `train.py`
 
No entrena nada. Comprueba la configuración y, con `main()` ejecutado sobre una configuración de juguete, que las piezas están bien conectadas.
 
| Test | Qué comprueba |
| --- | --- |
| `test_registry_contains_the_three_models` | El registro `MODELS` contiene `cnn`, `resnet18-fe` y `resnet18-ft`. |
| `test_image_size_matches_each_model` | La CNN usa 128 y los dos ResNet usan `RESNET_INPUT_SIZE`. |
| `test_checkpoint_names_are_unique` | Ningún nombre de checkpoint se repite entre configuraciones. Si dos modelos compartieran archivo, uno sobrescribiría al otro en silencio. |
| `test_finetune_uses_a_lower_lr_for_layer4_than_for_the_head` | En fine-tuning, `layer4` va con un `lr` diez veces menor que la cabeza `fc`, y cada grupo contiene exactamente sus parámetros. |
| `test_model_argument_is_required` | Sin `--model`, el programa termina con error de uso. |
| `test_unknown_model_is_rejected` | Un modelo que no está en el registro se rechaza. |
| `test_optional_arguments_default_to_none` | Sin opciones, `--epochs`, `--lr` y `--weight-decay` valen `None` (para que mande la configuración) y `--output-dir` vale `outputs`. |
| `test_arguments_are_parsed_with_the_right_types` | `--epochs 3` es un entero, `--lr 0.01` y `--weight-decay 0.0` son decimales. |
| `test_defaults_come_from_the_model_config` | Sin opciones, `main()` usa las épocas, el tamaño de imagen, el `lr` y el `weight_decay` de la configuración. |
| `test_cli_overrides_replace_the_defaults` | `--epochs 2 --lr 0.5` sustituyen a los valores por defecto. |
| `test_zero_weight_decay_overrides_the_default` | `--weight-decay 0.0` sustituye al valor por defecto (no se confunde con "no indicado"). Es la razón por la que el código usa `is not None` y no `or`. |
| `test_best_checkpoint_follows_validation_accuracy` | Con precisiones de validación 0,5 / 0,7 / 0,6, el "mejor" checkpoint es el de la época 2 y el final es el de la época 3. Los pesos falsos se rellenan con el número de época para poder identificar de cuál viene cada archivo. |
| `test_output_directory_is_created` | Si la carpeta de salida no existe, se crea y contiene los dos checkpoints. |
 
### `tests/test_api.py` — la API web (FastAPI)

Usa la fixture `client`: crea un checkpoint vacío en una carpeta temporal, apunta `CHECKPOINT_PATH` a él, sustituye `load_finetuned_model` por una función que devuelve `FixedLogits` y arranca la aplicación con `TestClient`. No necesita el checkpoint real, el dataset ni internet. Las imágenes de las peticiones se generan en memoria.

| Test | Qué comprueba |
| --- | --- |
| `test_health_reports_model_loaded` | `GET /health` responde 200 y `model_loaded` es `true`. |
| `test_get_index_returns_html` | `GET /` devuelve HTML que contiene el título de la página. |
| `test_get_static_app_js` | `GET /static/app.js` responde 200 y se sirve como JavaScript. Solo se comprueba que el tipo contenga `javascript`, porque el tipo exacto puede variar según el sistema. |
| `test_post_receives_valid_image` | `POST /predict` con una imagen válida devuelve 200 y 5 predicciones (el valor por defecto), cada una con nombre y una probabilidad entre 0 y 1, ordenadas de mayor a menor. |
| `test_top_k_parameter` | Con `top_k=3` devuelve exactamente 3 predicciones. |
| `test_check_image_format` | Un archivo que no es una imagen (texto) da 400 con el mensaje `Invalid image file`. |
| `test_check_corrupt_image` | Un JPEG recortado a la mitad (cabecera correcta, datos incompletos) da 400 con `Invalid image file`. `Image.open` solo lee la cabecera y el error aparece al decodificar, por eso el endpoint llama a `image.load()` dentro del `try`. |
| `test_top_k_out_of_bounds` (0, 103) | Un `top_k` fuera del rango 1–102 da 422. |
| `test_file_not_provided` | Una petición sin archivo da 422. |
| `test_startup_fails_when_checkpoint_missing` | Con `CHECKPOINT_PATH` apuntando a un archivo que no existe, arrancar la aplicación lanza `FileNotFoundError`. |
| `test_startup_fails_when_model_cannot_be_loaded` | Si el archivo existe pero el cargador falla, el arranque lanza ese error. |
| `test_predict_returns_503_when_model_missing` | Si se elimina el modelo de `app.state`, `POST /predict` da 503. |

### `tests/test_gpu.py` — CUDA
 
Todo el archivo se salta automáticamente si no hay una GPU disponible (`pytest.mark.skipif`).
 
| Test | Qué comprueba |
| --- | --- |
| `test_evaluate_gives_same_result_on_cpu_and_gpu` | `evaluate` da el mismo resultado (con tolerancia, porque la coma flotante no es idéntica bit a bit entre dispositivos) en CPU y GPU. |
| `test_train_one_epoch_runs_on_gpu` | Tras entrenar una época en GPU, el modelo sigue en la GPU y las métricas son válidas. Los datos del `DataLoader` están en CPU, así que también comprueba que `engine.py` los mueve al dispositivo. |
 
## Comprobar que los tests detectan fallos
 
Que un test pase no demuestra que sirva: un test que nunca puede fallar no protege nada. Para comprobarlo se rompió el código a propósito, se ejecutaron los tests y se restauró después:
 
| Cambio | Tests que fallan | Qué se observó |
| --- | --- | --- |
| Quitar `optimizer.step()` de `train_one_epoch` | `test_train_one_epoch_updates_weights`, `test_model_can_overfit_toy_data` | En el segundo, la pérdida final era idéntica a la inicial: el modelo no se movió. |
| Quitar `@torch.no_grad()` de `evaluate` | `test_evaluate_runs_without_autograd` | El espía registró `[True, True, True, True]`: autograd activo en todas las llamadas. |
| Quitar `.convert("RGB")` de `predict_top_k` | `test_accepts_non_rgb_images` (`L` y `RGBA`) | Errores de forma: 1 canal frente a 3 medias en escala de grises, y 4 frente a 3 con transparencia. |
| Quitar `ge=1` del parámetro `top_k` de `/predict` | `test_top_k_out_of_bounds[0]` | `assert 200 == 422`: sin el límite inferior, `top_k=0` se aceptaba. El caso `[103]` siguió pasando, porque `le=102` seguía en su sitio. |
| Cambiar `except OSError` por `except UnidentifiedImageError` en `/predict` | `test_check_corrupt_image` | `OSError: Truncated File Read`: la imagen truncada ya no se captura y el error sale del endpoint (`TestClient` lo vuelve a lanzar dentro del test). `UnidentifiedImageError` hereda de `OSError`, pero el fallo al decodificar una imagen truncada es un `OSError` a secas. |

El mismo principio se aplicó a las puertas de seguridad del CI (Trivy y Checkov): se comprobó que fallan cuando deben, con PR de prueba que no se fusionan. Ver [security.md](./security.md#pruebas-negativas-de-las-puertas).
 
## Qué no cubren los tests
 
- **El rendimiento real del modelo.** Lo miden `evaluate.py` y `confusion_analysis.py` de forma manual, y está documentado en `results.md`.
- **Un entrenamiento real con el dataset.** `train.py` se prueba con datos y pasos falsos, no con Oxford 102 Flowers.
- **Los scripts que dependen del dataset o de un checkpoint:** `evaluate.py`, `confusion_analysis.py`, `data_setup.py`, `explore_data.py` y el `main()` de `predict.py` (solo se prueba `predict_top_k`).
- **La GPU en la integración continua.** Los tests de CUDA solo se ejecutan en una máquina con GPU; en GitHub Actions se saltan.
- **La API con el modelo real.** Los tests de la API usan un modelo falso: comprueban el contrato (códigos de estado, formato de la respuesta, validación de la entrada), no que las predicciones sean acertadas.
- **La página web.** El JavaScript (`app.js`) no tiene tests automáticos; se comprobó a mano en el navegador.
- **El contenedor en ejecución.** El CI construye la imagen y la escanea (Trivy), pero no la arranca. Que el contenedor arranque, que `/health` responda y que el `HEALTHCHECK` lo marque como `healthy` se comprobó a mano con `docker run` y `docker ps`.

## Cómo se resuelven los imports
 
Los tests importan los módulos del proyecto como un paquete instalado: `from flower_classifier.model import SimpleCNN`, `from flower_classifier.model_transfer import ...`, `from flower_classifier.class_names import idx_to_name`.
 
Esto es resultado de un cambio posterior a la creación original de los tests: el proyecto se reestructuró como un paquete Python instalable (`src/flower_classifier/`, con `pyproject.toml`), instalado en modo editable:
 
```powershell
pip install -e ".[dev,api]"
```

Los tests de la API importan `app.main`, y `app/` no es un paquete instalado (está en la raíz del proyecto). Por eso `pyproject.toml` incluye una sección `[tool.pytest.ini_options]` con `pythonpath = ["."]`, que añade la raíz del proyecto al camino de búsqueda de pytest. Sin ella, `pytest` a secas no encontraría `app`.
 
## Cómo ejecutar los tests
 
Desde la raíz del proyecto, con el entorno virtual activado:
 
```powershell
pytest -v
```
 
Resultado esperado: 65 tests recogidos, en pocos segundos (unos 8 en la máquina de desarrollo, con GPU). En una máquina con GPU pasan los 65; sin GPU, 62 pasan y 3 se saltan (los de CUDA). Para ver por qué se salta cada uno, `pytest -v -rs`. Para ejecutar solo los de la API: `pytest tests/test_api.py -v`. No hace falta descargar el dataset ni entrenar nada; solo `test_model_transfer.py` necesita internet la primera vez, para descargar los pesos de ImageNet.
 
## Integración continua
 
Los mismos tests se ejecutan automáticamente en GitHub Actions (`.github/workflows/tests.yml`) en cada `push` a `main` y en cada Pull Request. El workflow instala el proyecto (`pip install -e ".[dev,api]"`) en una máquina Linux limpia, con PyTorch solo para CPU, y lanza `pytest -v`. Esto detecta problemas que en un entorno local pueden pasar desapercibidos, como una dependencia que falta en `pyproject.toml`. Para acelerar las ejecuciones, se guarda en caché la descarga de los pesos de ImageNet que necesita `test_model_transfer.py`.
 
Como esa máquina no tiene GPU, el resultado esperado allí es **62 pasados y 3 saltados**.

El workflow de tests solo recibe permisos de lectura sobre el repositorio (`permissions: contents: read`) y sus acciones están fijadas por hash de commit. Los análisis de seguridad no viven aquí: están en otro workflow, `security.yml` (Trivy sobre la imagen y Checkov sobre el `Dockerfile` y los workflows), que se ejecuta en los mismos eventos. Ver [security.md](./security.md).