# Seguridad

Qué comprobaciones de seguridad tiene el proyecto, para qué sirve cada una, qué **no** cubre y qué riesgos se aceptan a propósito. Describe un proyecto personal pequeño, no una auditoría formal ni una evaluación de cumplimiento normativo.

Las cifras actuales (hallazgos, comprobaciones superadas, fechas de los escaneos) están en el [README](../README.md#vulnerability-scanning). No se repiten aquí para que no puedan quedar desactualizadas en dos sitios a la vez.

## Resumen de controles

| Control | Qué comprueba | Dónde se ejecuta | Quién lo hace cumplir |
| --- | --- | --- | --- |
| Escaneo de vulnerabilidades de la imagen | Vulnerabilidades conocidas (HIGH y CRITICAL) en los paquetes del sistema operativo y de Python de la imagen construida | `security.yml`, job `trivy-image` | Trivy v0.76.1; el job falla con código de salida 1 si hay hallazgos (`--ignore-unfixed`, ver [Riesgos aceptados](#riesgos-aceptados)) |
| Escaneo de configuración | El `Dockerfile` y los ficheros de workflow frente a las reglas incluidas en Checkov | `security.yml`, job `checkov` | Checkov 3.4.0, imagen fijada por digest; el job falla si una regla falla |
| Endurecimiento del contenedor | Usuario sin privilegios, sin `pip` ni `setuptools` en la imagen final, modelo fuera de la imagen, `HEALTHCHECK`, contexto de construcción pequeño | `Dockerfile`, `.dockerignore` | El usuario sin privilegios y el `HEALTHCHECK` los comprueba Checkov; el resto es decisión de diseño y se refleja en los resultados de Trivy |
| Endurecimiento de los workflows | Token limitado a `contents: read`; acciones fijadas por hash de commit | `tests.yml`, `security.yml` | Los permisos los comprueba Checkov (`CKV2_GHA_1`); el fijado por hash se hace a mano |

Los dos jobs de seguridad se ejecutan en cada pull request y en cada push a `main`.

## Principios de diseño

- **Hacer fallar la construcción en lugar de solo informar.** Un escáner que solo imprime una tabla acaba ignorado.
- **Demostrar que la puerta puede fallar.** Un check en verde no dice nada hasta que se ha visto el mismo check en rojo (ver [Pruebas negativas de las puertas](#pruebas-negativas-de-las-puertas)).
- **Fijar lo que se ejecuta en el pipeline.** Las acciones se fijan por hash de commit, Trivy por versión y Checkov por digest de imagen. Una etiqueta (*tag*) puede moverse para apuntar a otro código; un hash de commit o un digest, no. No es teórico: las versiones y etiquetas de acciones del propio Trivy se vieron afectadas por un incidente de cadena de suministro en marzo de 2026 [enlace al aviso: pendiente de añadir tras verificarlo]. Fijar limita lo que puede cambiar bajo el pipeline sin una revisión, pero no hace segura por sí sola una versión fijada.
- **Mínimo privilegio.** El token del workflow es de solo lectura.
- **Leer qué se ha escaneado, no solo el color del check.** El log forma parte de la revisión.

## Pruebas negativas de las puertas

| Puerta | Prueba negativa | Resultado |
| --- | --- | --- |
| Trivy | [PR #9](https://github.com/nataliagalanalv/flower-classifier/pull/9) (cerrado, sin fusionar) fijó la versión vulnerable Pillow 12.3.0 | `security` falló con código de salida 1 y los hallazgos de Pillow; `tests` pasó |
| Checkov | [PR #13](https://github.com/nataliagalanalv/flower-classifier/pull/13) (cerrado, sin fusionar) quitó el `HEALTHCHECK` | El paso del Dockerfile falló con `CKV_DOCKER_2`; `tests` y `trivy-image` pasaron |

Las capturas están en `docs/images/`. Dos de estas pruebas no funcionaron a la primera, y las dos enseñaron algo:

- **La primera prueba de Trivy no era válida.** Usaba `pillow>=12.3.0`, que sigue resolviéndose a Pillow 13.0.0, así que no se instaló nada vulnerable y el check siguió en verde. Una prueba negativa tiene que crear de verdad la condición que dice probar, así que se cambió a `pillow==12.3.0`.
- **El primer job de Checkov solo escaneaba la mitad de sus objetivos.** Pasaba `--file` y `--directory` en un solo comando, y el log tenía un bloque `github_actions scan results` pero ninguno de `dockerfile scan results`. El Dockerfile nunca se comprobaba, así que quitar el `HEALTHCHECK` seguía en verde. Se corrigió en el [PR #14](https://github.com/nataliagalanalv/flower-classifier/pull/14) con un paso por objetivo. No se conoce la causa de que el comando único se comportara así; el log es la única evidencia.

## Riesgos aceptados

| Riesgo | Por qué se acepta | Revisión |
| --- | --- | --- |
| Hallazgos HIGH en paquetes de Debian de la imagen base sin versión corregida disponible | No se puede arreglar nada en este repositorio hasta que Debian publique una corrección. El CI los oculta con `--ignore-unfixed`; el comando local del README no lo usa, así que siguen visibles. **No** se ha evaluado si cada uno es alcanzable desde esta aplicación. | Volver a escanear cada vez que se reconstruya la imagen base y cuando la etiqueta de la imagen base pase a una versión más nueva |

Un riesgo aceptado es una decisión, no una corrección: se queda en esta lista hasta que los hallazgos desaparezcan o se evalúen uno a uno.

## Huecos conocidos

Todavía sin hacer, sin un orden concreto:

- **No hay un escaneo programado.** Los escaneos se ejecutan en push y en pull request. La base de datos de vulnerabilidades se actualiza a diario, así que una CVE nueva que afecte a una imagen sin cambios no se detecta hasta el siguiente cambio.
- **`main` no está protegida.** Los pull requests son una costumbre, no una regla: se puede subir directamente a `main` (y ha pasado con commits de documentación).
- **Las dependencias se declaran con `>=`.** Dos construcciones en días distintos pueden instalar versiones distintas. Falta fijar las versiones o usar un fichero de bloqueo.
- **La imagen base se referencia por etiqueta** (`python:3.14-slim`), no por digest. La etiqueta sigue recibiendo reconstrucciones con parches de seguridad, lo cual es útil, pero las construcciones no son reproducibles.
- **`runs-on: ubuntu-latest` no está fijado**, así que la imagen del ejecutor cambia con el tiempo. Algunas acciones están en versiones antiguas (el paso de checkout muestra un aviso de obsolescencia de Node 20); actualizarlas exige leer antes las notas de cada versión.
- **No hay escaneo de secretos en el CI.** El escáner de secretos de Trivy está desactivado (`--scanners vuln`) porque agotaba el tiempo con la imagen de 2 GB. Escanear los ficheros del repositorio en lugar de las capas de la imagen encaja mejor y está por evaluar.
- **No se analiza el código de la aplicación** (ni análisis estático ni dinámico).
- **La API no tiene autenticación ni límite de peticiones**, y no se ha verificado el límite de tamaño de las subidas. Hay que revisar `app/main.py` antes de afirmar nada sobre esto.
- **El contenedor se construye y se escanea en el CI, pero nunca se arranca.** Que arranca y pasa a `healthy` se comprobó a mano con `docker run` y `docker ps`.

## Lo que los escáneres no pueden decirte

- **Trivy** informa de vulnerabilidades *conocidas* en los paquetes instalados. No encuentra las desconocidas, y un escaneo limpio es una afirmación sobre su base de datos ese día.
- **Checkov** aplica las reglas con las que viene. Por ejemplo, en su salida no apareció ninguna regla sobre fijar acciones por hash, así que ninguna herramienta verifica ese control aquí.
- **Ninguno de los dos verifica el cumplimiento normativo**, tampoco el del RGPD.

## Privacidad (RGPD): pendiente de documentar

Ningún escáner comprueba esto y todavía no se ha escrito. Los puntos de abajo son preguntas por responder leyendo el código y el futuro despliegue, no afirmaciones sobre cómo se comporta la aplicación hoy:

- ¿Qué pasa con una imagen subida: se guarda, se registra en logs o solo se mantiene en memoria?
- ¿Las subidas pueden llevar datos personales (por ejemplo, coordenadas GPS en los metadatos EXIF)?
- ¿Se registran las direcciones IP de los clientes, dónde y durante cuánto tiempo?
- ¿En qué región estarán los logs y los datos almacenados (UE)?
- ¿Se recoge algo más de lo que necesita la predicción (minimización de datos)?

Esto es documentación de ingeniería, no asesoramiento jurídico. Está previsto para las últimas semanas del proyecto.

## Próximos pasos

Previstos, por orden de prioridad:

1. Proteger `main` (exigir pull requests y checks en verde).
2. Añadir una ejecución programada del workflow de seguridad.
3. Fijar las versiones de las dependencias y decidir cómo fijar la imagen base.
4. Añadir al CI una prueba de humo que arranque el contenedor y espere a que esté `healthy`.
5. Ejecutar Checkov sobre el código de Terraform cuando exista (la misma herramienta, un objetivo nuevo).
6. En AWS: roles de IAM con mínimo privilegio, GitHub Actions autenticándose por OIDC en lugar de claves de larga duración, y el modelo en un bucket privado. Son planes; cada uno debe contrastarse con la documentación actual de AWS cuando se implemente.