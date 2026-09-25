# Análisis y Conclusiones
 
Interpretación de los resultados recogidos en [`RESULTS.md`](./RESULTS.md).
 
## 1. Transfer learning domina claramente con datos escasos
 
Con solo 1020 imágenes de entrenamiento para 102 clases, la CNN entrenada desde cero nunca superó el **~38% de Val Accuracy**, incluso después de aplicar las técnicas de regularización más comunes (data augmentation agresivo, weight_decay, BatchNorm). Introducir ResNet18 preentrenada en ImageNet, sin apenas modificar el resto del pipeline, disparó el resultado a **~79%** (feature extraction).
 
Esto confirma que el cuello de botella no era la arquitectura del clasificador en sí, sino la falta de conocimiento visual de base (bordes, texturas, formas) que una red aprende de forma natural al entrenarse sobre millones de imágenes — conocimiento que 1020 imágenes de flores no bastan para construir desde cero.
 
## 2. El fine-tuning selectivo añade una mejora sustancial, con riesgo controlado
 
Descongelar únicamente el último bloque residual (`layer4`) de ResNet18, combinado con un learning rate bajo y diferenciado para esa capa (`1e-4` frente a `1e-3` de la capa nueva), elevó el resultado de ~79% a **~91%** en validación. No hizo falta reentrenar el modelo completo ni ampliar el dataset.
 
La clave práctica: las primeras capas de una red preentrenada capturan patrones genéricos, útiles para cualquier tarea de visión; solo las últimas capas conviene ajustarlas al dominio específico (en este caso, texturas y formas de flores).
 
## 3. `weight_decay` no aportó mejora medible en transfer learning
 
Se probó en las dos fases de transfer learning (feature extraction y fine-tuning) y en ambas la diferencia frente a no usarlo fue mínima (<1 punto porcentual), dentro del margen de variabilidad aleatoria observado entre corridas idénticas. Esto contrasta con su efecto claro en la CNN desde cero, donde redujo a la mitad el gap entre accuracy de entrenamiento y de validación.
 
**Hipótesis razonable (no confirmada de forma aislada):** con relativamente pocos parámetros libres (52K en feature extraction) o con un mecanismo ya acotado por learning rates bajos y pocas épocas (8.4M en fine-tuning), el margen para que la regularización L2 aporte valor adicional es limitado. La lección general: las técnicas de regularización deben validarse empíricamente en cada configuración concreta, no asumirse por defecto solo porque funcionaron en otro contexto.
 
## 4. Memorizar el training set no implicó overfitting dañino
 
En fine-tuning, el modelo alcanzó 100% de accuracy en entrenamiento desde la época 8-10, pero la accuracy de validación **siguió mejorando** en ese mismo periodo — a diferencia de la CNN, donde ese mismo patrón sí coincidió con el estancamiento y posterior deterioro de la validación.
 
Conclusión práctica: el gap entre Train Acc y Val Acc, por sí solo, no es una señal suficiente de overfitting problemático. Lo relevante es si la métrica de validación deja de mejorar (o empeora), no si el modelo llega a memorizar perfectamente el conjunto de entrenamiento.
 
## 5. La validación repetida infla ligeramente el resultado reportado
 
El mejor modelo alcanzó 90.78% en el split de validación, pero solo **87.87%** en el split de test (nunca usado para tomar ninguna decisión durante el desarrollo). Esa caída de ~3 puntos es coherente y esperable: al usar `val` repetidamente para comparar configuraciones (con/sin weight_decay, número de épocas, arquitectura), el modelo termina ligeramente "ajustado" a ese conjunto concreto, aunque nunca se haya entrenado directamente con él.
 
**87.87% es la cifra que debe reportarse como resultado final y honesto del proyecto** — no el 90.78% de validación.
 
## 6. El accuracy global no cuenta toda la historia (dataset desbalanceado)
 
Dos métricas de test arrojan resultados distintos porque miden cosas distintas:
 
- **Test Accuracy (87.87%, ponderado por muestra):** las clases con más imágenes en test pesan más en el resultado.
- **Accuracy media por clase (89.51%, no ponderado):** todas las 102 clases cuentan igual, sin importar cuántas imágenes tengan.
En este proyecto salió **al revés de lo habitual**: el promedio por clase fue *más alto* que el accuracy global. La causa identificada: *petunia* es la clase con más imágenes en test (238) y, a la vez, una de las peor clasificadas (55.88% de acierto) — su bajo rendimiento, al pesar mucho por volumen, arrastra el accuracy global hacia abajo con más fuerza que en el promedio no ponderado.
 
**Implicación práctica:** si se buscara mejorar el modelo, *petunia* sería la prioridad número uno — combina bajo accuracy con alto impacto en el resultado global, más que cualquier otra clase del top de "clases difíciles".
 
## 7. Errores del modelo: no aleatorios, sino sistemáticos
 
El listado de confusiones más frecuentes no muestra ruido disperso, sino un patrón claro: *petunia* protagoniza 3 de las 10 confusiones más comunes del modelo (confundida con hibiscus, garden phlox y morning glory), sumando 55 errores solo desde esa clase. El resto de confusiones del top 10 son puntuales (7-10 casos cada una), sugiriendo pares de especies con parecido visual genuino más que fallos aislados.
 
Distinción útil hecha durante el análisis: una clase con pocas imágenes en test (ej. *sweet pea*, 36 imágenes, 44.44% de acierto) puede parecer muy "difícil" en porcentaje, pero cada fallo adicional mueve varios puntos de golpe — su resultado es menos fiable estadísticamente que el de una clase con mucho volumen como *petunia*.
 
## Limitaciones y notas de fiabilidad
 
- **`cat_to_name.json`** (mapeo de índice numérico a nombre de flor) es un archivo de la comunidad, no una fuente oficial de Oxford ni de `torchvision`. Se detectó al menos una inconsistencia (`"lotus lotus"`, nombre duplicado) — los nombres deben tratarse como aproximados; el índice numérico de clase es la referencia fiable.
- No se fijó una semilla aleatoria (`torch.manual_seed`) en los entrenamientos, por lo que se observó variabilidad entre corridas idénticas (ej. 36.37% vs 37.65% en dos ejecuciones del mismo experimento de CNN). Los números reportados corresponden a las corridas concretas documentadas, no a un promedio de varias repeticiones.
- El efecto de `weight_decay` se evaluó combinado con otras técnicas en la CNN (augmentation + BatchNorm simultáneos), por lo que no se puede aislar completamente su contribución individual en esa fase.

