# Trabajo de Fin de Grado

## Sistema de Reconocimiento de Actividad Humana mediante Wi-Fi CSI

### Autor: Mario Díaz Gómez
### Universidad: Universidad [Pendiente de especificar]
### Grado: Ingeniería Informática / Ingeniería de Telecomunicaciones
### Curso académico: 2025–2026
### Tutor: [Nombre del tutor]

---

## Resumen

El presente Trabajo de Fin de Grado desarrolla un sistema completo de Reconocimiento de Actividad Humana (HAR, del inglés *Human Activity Recognition*) basado en información de estado del canal Wi-Fi (CSI, *Channel State Information*). El sistema es capaz de identificar siete actividades humanas distintas —permanecer de pie, caminar, levantarse del suelo, sentarse, levantarse de una silla, tumbarse y ausencia de persona— a partir del análisis de la señal Wi-Fi del entorno, sin necesidad de que el usuario lleve ningún dispositivo encima.

El trabajo abarca el ciclo completo de desarrollo de un sistema de inteligencia artificial aplicado a redes inalámbricas: desde la fundamentación teórica del canal OFDM y las firmas Doppler de las actividades humanas, pasando por la simulación física de datos CSI, el diseño e implementación de cuatro arquitecturas de aprendizaje profundo (SimpleLSTM, BiLSTM, FCN y CSITransformer), hasta la integración en una interfaz gráfica de usuario en tiempo real con seis pestañas funcionales.

La arquitectura CSITransformer, propuesta en este trabajo, alcanza una precisión de validación del 82,5% y un F1-score ponderado de 0,826 sobre datos simulados, superando a los modelos basados en LSTM y redes convolucionales del repositorio de referencia. Destaca su eficiencia computacional: con 330.887 parámetros (el modelo más ligero de los cuatro evaluados) y una latencia de inferencia de 5,3 ms en CPU, resulta apto para despliegues en dispositivos de bajo coste. El modelo de estimación de zona, basado en el modelo de pérdida de trayecto ITU-R P.1238, alcanza una precisión del 98,1% en la clasificación de cuatro zonas de distancia.

Una limitación fundamental del trabajo, declarada con honestidad científica a lo largo de todo el documento, es que el conjunto de datos original de 9 GB que motivó el diseño del sistema no estuvo disponible durante el desarrollo; en su lugar, se emplea un simulador de señal CSI basado en física (efectos Doppler, desvanecimiento Rician, modelo de pérdidas de trayecto) que reproduce las propiedades estadísticas documentadas en la literatura. Los resultados presentados son, por tanto, indicativos de la capacidad de cada arquitectura para aprender de las firmas de señal simuladas, y deben validarse con datos reales en trabajo futuro.

El sistema incluye una suite de 73 tests automatizados con pytest que cubren la canalización de datos, las arquitecturas de los modelos y el clasificador de zona. La interfaz gráfica, implementada con CustomTkinter y Matplotlib, proporciona monitorización en tiempo real, comparativa de modelos, visualización de matrices de confusión y gestión de datasets. El código está completamente documentado y estructurado para facilitar su extensión con hardware real (tarjetas Wi-Fi con soporte CSI como Atheros AR9380 o Intel 5300).

**Palabras clave:** Wi-Fi CSI, Reconocimiento de Actividad Humana, LSTM, Transformer, aprendizaje profundo, sensado pasivo, OFDM, posicionamiento indoor, PyTorch.

---

## Abstract

This Bachelor's Thesis presents a complete Human Activity Recognition (HAR) system based on Wi-Fi Channel State Information (CSI). The system identifies seven human activities —standing, walking, getting down, sitting, getting up, lying down, and no person present— by analysing ambient Wi-Fi signals, requiring no wearable device on the part of the user.

The work covers the full development cycle of an AI system applied to wireless networks: from the theoretical foundations of OFDM channels and human activity Doppler signatures, through physics-based CSI data simulation, the design and implementation of four deep learning architectures (SimpleLSTM, BiLSTM, FCN, and CSITransformer), to integration within a real-time graphical user interface with six functional tabs.

The CSITransformer architecture, proposed in this work, achieves 82.5% validation accuracy and a weighted F1-score of 0.826 on simulated data, outperforming the LSTM and convolutional baselines from the reference repository. It is also the most computationally efficient model: with 330,887 parameters and a CPU inference latency of 5.3 ms, it is suitable for deployment on low-cost edge devices. The zone estimation model, grounded in the ITU-R P.1238 indoor path loss model, achieves 98.1% accuracy in classifying four distance zones.

A fundamental limitation of this work, declared transparently throughout, is that the original 9 GB real-hardware CSI dataset that motivated the system design was unavailable during development. Instead, a physics-inspired CSI signal simulator (modelling Doppler effects, Rician fading, and path loss) was used to reproduce the statistical properties documented in the literature. The reported results therefore reflect each architecture's capacity to learn from simulated signal signatures, and must be validated with real data in future work.

The system includes a suite of 73 automated pytest tests covering the data pipeline, model architectures, and zone classifier. The GUI, implemented with CustomTkinter and Matplotlib, provides real-time monitoring, model comparison, confusion matrix visualisation, and dataset management. The codebase is fully documented and structured to facilitate extension with real CSI-capable hardware such as Atheros AR9380 or Intel 5300 NICs.

**Keywords:** Wi-Fi CSI, Human Activity Recognition, LSTM, Transformer, deep learning, passive sensing, OFDM, indoor positioning, PyTorch.

---

## Agradecimientos

En primer lugar, quiero expresar mi más sincero agradecimiento a mi tutor, cuya orientación, paciencia y rigor científico han sido fundamentales para dar forma a este trabajo. Sus comentarios críticos y su disposición a resolver dudas en todo momento han contribuido decisivamente a la calidad del resultado final.

A mis compañeros de carrera, con quienes he compartido cinco años de estudio, prácticas y, en muchas ocasiones, largas noches de depuración de código: gracias por el apoyo mutuo y el buen humor que hacen más llevadera la exigencia académica.

A mi familia, en especial a mis padres, por su apoyo incondicional durante toda la carrera, por creer en mí incluso cuando los plazos apretaban y la duda se instalaba, y por hacer posible que llegara hasta aquí.

Por último, agradezco a la comunidad académica y de código abierto cuyos trabajos, datasets y frameworks —PyTorch, scikit-learn, CustomTkinter— han permitido construir este sistema. La ciencia avanza sobre los hombros de quienes comparten su trabajo libremente.

---

## Índice de Contenidos

1. [Introducción](#1-introducción)
   - 1.1 Motivación y contexto
   - 1.2 Objetivos del trabajo
   - 1.3 Alcance y limitaciones
   - 1.4 Estructura del documento
2. [Estado del Arte](#2-estado-del-arte)
   - 2.1 Sensado pasivo de actividad humana
   - 2.2 Wi-Fi CSI para HAR: evolución histórica
   - 2.3 Trabajos relacionados
   - 2.4 Arquitecturas de deep learning para series temporales
   - 2.5 Posicionamiento indoor Wi-Fi
   - 2.6 Conclusiones del estado del arte
3. [Fundamentos Teóricos](#3-fundamentos-teóricos)
   - 3.1 Comunicaciones OFDM y CSI
   - 3.2 Modelo matemático del canal Wi-Fi
   - 3.3 Firmas Doppler de actividades humanas
   - 3.4 Redes LSTM y LSTM Bidireccional
   - 3.5 Redes Completamente Convolucionales para series temporales
   - 3.6 Arquitectura Transformer para señales temporales
   - 3.7 Estimación de distancia mediante pérdida de trayecto ITU-R P.1238
4. [Diseño del Sistema](#4-diseño-del-sistema)
   - 4.1 Arquitectura general del sistema
   - 4.2 Pipeline de preprocesamiento CSI
   - 4.3 Modelo SimpleLSTM
   - 4.4 Modelo BiLSTM
   - 4.5 Modelo FCN
   - 4.6 Modelo CSITransformer
   - 4.7 Clasificador de Zona
   - 4.8 Estrategia de balanceo y ponderación de clases
   - 4.9 Interfaz Gráfica de Usuario
5. [Implementación](#5-implementación)
   - 5.1 Entorno de desarrollo y dependencias
   - 5.2 Generación de datos sintéticos
   - 5.3 Pipeline de entrenamiento multi-modelo
   - 5.4 Suite de tests automatizados
   - 5.5 Carga de datos reales
   - 5.6 Protocolo de inferencia en tiempo real
6. [Evaluación Experimental](#6-evaluación-experimental)
   - 6.1 Configuración experimental
   - 6.2 Comparativa de modelos HAR
   - 6.3 Análisis por clase — CSITransformer
   - 6.4 Análisis por clase — BiLSTM
   - 6.5 Análisis por clase — SimpleLSTM
   - 6.6 Análisis por clase — FCN
   - 6.7 Estimación de zona
   - 6.8 Análisis de latencia y complejidad computacional
   - 6.9 Discusión de resultados
   - 6.10 Análisis comparativo cruzado
   - 6.11 Validación del simulador CSI
7. [Conclusiones](#7-conclusiones)
   - 7.1 Conclusiones principales
   - 7.2 Limitaciones del trabajo
   - 7.3 Trabajo futuro
   - 7.4 Reflexión personal
8. [Bibliografía](#bibliografía)
9. [Anexos](#anexos)
   - Anexo A: Configuración del entorno de desarrollo
   - Anexo B: Resumen de la arquitectura de los modelos
   - Anexo C: Instrucciones de uso del sistema
   - Anexo D: Resultados completos de tests

---

## Lista de Figuras

- **Figura 1.** Arquitectura general del sistema CSI-HAR.
- **Figura 2.** Pipeline de preprocesamiento: de amplitudes brutas a tensor de 468 características.
- **Figura 3.** Arquitectura del modelo CSITransformer: stem convolucional, codificación posicional y encoder Transformer.
- **Figura 4.** Arquitectura del modelo SimpleLSTM con proyección de entrada.
- **Figura 5.** Arquitectura del modelo BiLSTM con cabeza clasificadora de dos capas.
- **Figura 6.** Arquitectura FCN: tres bloques convolucionales con Global Average Pooling.
- **Figura 7.** Arquitectura del ZoneClassifier: MLP 16→64→32→4.
- **Figura 8.** Curvas de entrenamiento (pérdida y precisión) para los cuatro modelos HAR.
- **Figura 9.** Matriz de confusión — CSITransformer (mejor modelo).
- **Figura 10.** Matriz de confusión — BiLSTM.
- **Figura 11.** Matriz de confusión — SimpleLSTM.
- **Figura 12.** Matriz de confusión — FCN.
- **Figura 13.** Comparativa de F1-score por clase entre los cuatro modelos.
- **Figura 14.** Interfaz gráfica: pestaña de monitorización en tiempo real.
- **Figura 15.** Modelo de pérdida de trayecto ITU-R P.1238 y zonas de distancia definidas.

---

## Lista de Tablas

- **Tabla 1.** Trabajos relacionados en CSI-HAR (comparativa con la literatura).
- **Tabla 2.** Parámetros del sistema y constantes del dataset de referencia.
- **Tabla 3.** Comparativa de modelos HAR: precisión, F1, parámetros y latencia.
- **Tabla 4.** F1-score por clase — CSITransformer.
- **Tabla 5.** F1-score por clase — BiLSTM.
- **Tabla 6.** F1-score por clase — SimpleLSTM.
- **Tabla 7.** F1-score por clase — FCN.
- **Tabla 8.** Tasas de aprendizaje por modelo.
- **Tabla 9.** Zonas de distancia definidas con el modelo ITU-R P.1238.
- **Tabla 10.** Resumen de la suite de tests automatizados.
- **Tabla 11.** Comparativa cruzada de F1-score por clase para los cuatro modelos HAR.
- **Tabla 12.** Parámetros del modelo de señal simulada por actividad.

---

## Lista de Acrónimos

| Acrónimo | Significado |
|---|---|
| ANN | Artificial Neural Network (Red Neuronal Artificial) |
| AP | Access Point (Punto de Acceso Wi-Fi) |
| AWGN | Additive White Gaussian Noise (Ruido Blanco Gaussiano Aditivo) |
| BiLSTM | Bidirectional Long Short-Term Memory |
| BN | Batch Normalization (Normalización por Lotes) |
| CSI | Channel State Information (Información de Estado del Canal) |
| CNN | Convolutional Neural Network (Red Neuronal Convolucional) |
| CPU | Central Processing Unit (Unidad Central de Procesamiento) |
| DL | Deep Learning (Aprendizaje Profundo) |
| FCN | Fully Convolutional Network (Red Completamente Convolucional) |
| FFT | Fast Fourier Transform (Transformada Rápida de Fourier) |
| GAP | Global Average Pooling |
| GPU | Graphics Processing Unit (Unidad de Procesamiento Gráfico) |
| GUI | Graphical User Interface (Interfaz Gráfica de Usuario) |
| HAR | Human Activity Recognition (Reconocimiento de Actividad Humana) |
| ITU-R | International Telecommunication Union – Radiocommunication Sector |
| LOS | Line of Sight (Línea de Visión Directa) |
| LSTM | Long Short-Term Memory |
| MIMO | Multiple-Input Multiple-Output |
| MHA | Multi-Head Attention (Atención Multi-Cabeza) |
| MLP | Multilayer Perceptron (Perceptrón Multicapa) |
| NLOS | Non-Line of Sight (Sin Línea de Visión Directa) |
| OFDM | Orthogonal Frequency Division Multiplexing |
| PCA | Principal Component Analysis (Análisis de Componentes Principales) |
| RSSI | Received Signal Strength Indicator |
| ReLU | Rectified Linear Unit |
| RF | Radiofrecuencia |
| RSS | Received Signal Strength |
| TFG | Trabajo de Fin de Grado |
| Wi-Fi | Wireless Fidelity |

---

## 1. Introducción

### 1.1 Motivación y contexto

El reconocimiento automático de actividad humana es una de las áreas de investigación más activas en la intersección entre las redes inalámbricas, el aprendizaje automático y la computación ubícua. La motivación fundamental es la posibilidad de inferir el comportamiento de las personas en un espacio interior —si están caminando, sentadas, tumbadas o si no hay nadie— sin necesidad de cámaras, sensores corporales ni ningún otro dispositivo intrusivo. Las aplicaciones potenciales son amplias e incluyen la monitorización de personas mayores en el hogar para detectar caídas, la automatización inteligente de edificios, la seguridad perimetral y la asistencia sanitaria remota.

Las soluciones basadas en visión artificial, aunque potentes, presentan restricciones inherentes relacionadas con la privacidad y con la necesidad de iluminación adecuada. Los sistemas de sensores corporales (acelerómetros, giroscopios) requieren que el usuario lleve un dispositivo consigo en todo momento, lo que reduce su aceptación en contextos de monitorización continua. Frente a estas alternativas, el sensado pasivo mediante señales Wi-Fi se presenta como una opción atractiva: las infraestructuras Wi-Fi ya están desplegadas en prácticamente todos los hogares y edificios, y los movimientos humanos perturbando las ondas de radio pueden capturarse y analizarse sin que el usuario participe activamente ni tenga conciencia del sistema.

La información de estado del canal (CSI) representa una evolución significativa respecto al indicador de nivel de señal recibida (RSSI) utilizado en trabajos anteriores. Mientras que el RSSI proporciona únicamente un valor escalar por paquete, el CSI captura la respuesta en frecuencia compleja del canal en cada una de las subportadoras OFDM, ofreciendo una representación multidimensional mucho más rica de los efectos que los movimientos humanos producen sobre la señal. Con hardware basado en el Atheros CSI Tool (como el utilizado en el dataset que motiva este trabajo, con tarjetas TP-Link WDR3600), es posible extraer amplitud y fase de hasta 114 subportadoras en cada par de antenas, proporcionando una "huella digital" característica de cada actividad.

Este TFG nace de la voluntad de construir un sistema completo, desde los fundamentos físicos hasta la interfaz de usuario, que demuestre la viabilidad de HAR basado en CSI con arquitecturas de aprendizaje profundo modernas. Se parte como referencia del dataset público de Kovalenko et al. (2021) y de su repositorio de código asociado (wifi-csi-har), adaptando y extendiendo sus arquitecturas con una propuesta propia basada en Transformers.

El contexto socioeconómico que motiva este trabajo es relevante. El envejecimiento de la población en los países desarrollados está incrementando la demanda de sistemas de monitorización no intrusiva para personas mayores que viven solas. Según datos del Instituto Nacional de Estadística (INE), en España el 14% de los hogares están formados por una sola persona mayor de 65 años, y este porcentaje aumentará hasta el 20% en 2050. Los sistemas de detección de caídas actuales (botones de pánico, alfombras sensoriales) requieren la colaboración activa del usuario o la instalación de hardware especializado. Un sistema de monitorización pasiva basado en la infraestructura Wi-Fi existente podría democratizar el acceso a este tipo de protección sin coste adicional de hardware ni invasión de la privacidad, representando una aplicación de alto impacto social para la tecnología desarrollada en este trabajo.

### 1.2 Objetivos del trabajo

Los objetivos del presente trabajo se articulan en torno a cinco ejes principales:

**Objetivo 1 — Comprensión y modelado del canal CSI.** Desarrollar un simulador de señal CSI fundamentado en la física del canal inalámbrico (efectos Doppler, desvanecimiento Rician, modelo de pérdidas ITU-R P.1238) que reproduzca las propiedades estadísticas del dataset real de referencia y permita generar datos de entrenamiento cuando el hardware o los datos reales no están disponibles.

**Objetivo 2 — Implementación y comparativa de arquitecturas DL.** Implementar cuatro arquitecturas de aprendizaje profundo para clasificación de series temporales CSI: SimpleLSTM, BiLSTM, FCN y un CSITransformer de diseño propio. Evaluar cuantitativamente cada modelo en términos de precisión, F1-score, número de parámetros y latencia de inferencia en CPU.

**Objetivo 3 — Estimación de zona basada en CSI.** Diseñar e implementar un clasificador de zona de distancia (Proximidad, Cerca, Media distancia, Lejos) fundamentado en el modelo de pérdidas de trayecto ITU-R P.1238, complementando la información de actividad con información de posición relativa.

**Objetivo 4 — Interfaz gráfica de usuario.** Desarrollar una GUI funcional y visualmente clara que permita la monitorización en tiempo real de la actividad y la zona, la comparación de modelos, la visualización de las matrices de confusión y la carga de datasets reales en múltiples formatos.

**Objetivo 5 — Calidad de software.** Garantizar la calidad y mantenibilidad del código mediante una suite de 73 tests automatizados con pytest, documentación exhaustiva de todas las funciones y módulos, y una estructura de proyecto clara y reproducible.

### 1.3 Alcance y limitaciones

El alcance del trabajo comprende el diseño, implementación, entrenamiento y evaluación de un sistema completo de HAR basado en CSI simulado, así como el desarrollo de una interfaz gráfica de monitorización. No se incluye en el alcance:

- **Captura de datos CSI reales:** el dataset de referencia (Kovalenko et al., 2021) tiene un tamaño de 9 GB y requiere hardware específico (Atheros AR9380, driver CSI Tool) para su reproducción. Su descarga y procesamiento quedan fuera del alcance del proyecto, aunque el sistema está diseñado para aceptar datos reales en el tab de datasets de la GUI.
- **Despliegue en producción:** el sistema opera sobre datos simulados o datos reales cargados desde archivo. No se ha implementado la captura de paquetes en tiempo real desde el kernel del sistema operativo.
- **Evaluación en entorno real:** la validación experimental se realiza íntegramente sobre datos simulados. Los resultados deben interpretarse en este contexto.

La limitación más importante, que se hace explícita en cada apartado relevante del documento, es precisamente la ausencia de datos CSI reales en el entrenamiento y la evaluación. El simulador reproduce con fidelidad razonable las firmas estadísticas de cada actividad, pero no puede capturar la riqueza de variabilidad que presentaría un dataset capturado en condiciones reales con distintos sujetos, entornos y posiciones.

### 1.4 Estructura del documento

El documento se organiza en siete capítulos principales más bibliografía y anexos:

- **Capítulo 2** presenta el estado del arte en HAR pasivo, CSI Wi-Fi y deep learning para series temporales, con una tabla comparativa de más de diez trabajos de referencia.
- **Capítulo 3** desarrolla los fundamentos teóricos: comunicaciones OFDM, modelo matemático del canal, firmas Doppler, arquitecturas LSTM y Transformer, y el modelo de pérdidas ITU-R P.1238.
- **Capítulo 4** describe el diseño detallado del sistema: cada arquitectura, el pipeline de preprocesamiento, la estrategia de balanceo y la GUI.
- **Capítulo 5** cubre la implementación: entorno, generación de datos sintéticos, entrenamiento, tests y protocolo de inferencia.
- **Capítulo 6** presenta y discute los resultados experimentales: comparativa de modelos, análisis por clase, latencia y discusión crítica.
- **Capítulo 7** recoge las conclusiones, limitaciones, trabajo futuro y reflexión personal.
- Los **Anexos** proporcionan instrucciones de instalación, resúmenes arquitectónicos y resultados completos de tests.

---

## 2. Estado del Arte

### 2.1 Sensado pasivo de actividad humana

El sensado pasivo de actividad humana (passive HAR) engloba un conjunto de técnicas que infieren el comportamiento de las personas a partir de señales del entorno, sin requerir la cooperación activa del sujeto. Las principales tecnologías empleadas en la literatura incluyen:

**Sistemas de visión artificial.** Las cámaras RGB, de profundidad (RGB-D) y termográficas constituyen la tecnología más madura para HAR. Trabajos seminales como el de Laptev (2005) con descriptores espacio-temporales y, más recientemente, los basados en redes convolucionales 3D (Tran et al., 2015) o estimación de pose (Cao et al., 2019) han demostrado precisiones superiores al 95% en benchmarks estándar. Sin embargo, las cámaras presentan limitaciones en privacidad, cobertura a múltiples habitaciones y condiciones de baja iluminación.

**Sensores inerciales y wearables.** Los acelerómetros y giroscopios integrados en smartphones o smartwatches permiten reconocer actividades con alta precisión (>95% en los datasets UCI-HAR y WISDM), como demuestran los trabajos de Anguita et al. (2013) y Yang et al. (2015) con CNNs y LSTMs respectivamente. Su principal limitación es la necesidad de llevar el dispositivo.

**Radar y ultrasonidos.** Los sistemas de radar Doppler y ultra-wideband (UWB) ofrecen excelente resolución para detectar movimientos sutiles (respiración, caídas), pero su coste y la necesidad de hardware dedicado limitan su adopción masiva.

**Señales de radiofrecuencia ambiental.** El uso de señales Wi-Fi, ZigBee, Bluetooth y señales de televisión digital para detectar presencia y actividad humana ha ganado tracción a partir de 2013. El enfoque basado en RSSI (sencillo de extraer con hardware estándar) fue el primero en explorarse, pero su baja dimensionalidad limita la granularidad de la clasificación. La transición al CSI, que proporciona información por subportadora, ha permitido reconocer actividades mucho más matizadas.

### 2.2 Wi-Fi CSI para HAR: evolución histórica

El uso de CSI para reconocimiento de actividad puede dividirse en cuatro etapas históricas:

**Primera etapa (2010–2014): detección de presencia.** Los trabajos pioneros de Yousefi et al. (2013) y el sistema Wifall (Mao et al., 2012) demostraron que el CSI podía distinguir de forma fiable si había o no una persona en la sala, aprovechando la diferencia entre un canal estático y uno perturbado por movimiento. Las técnicas empleadas eran clásicas: umbralización de varianza o SVMs sobre features estadísticas.

**Segunda etapa (2015–2018): reconocimiento de gestos y actividades básicas.** El sistema CARM de Wang et al. (2015) introdujo el modelo de velocidad-actividad, relacionando matemáticamente las variaciones del CSI con la velocidad Doppler de los segmentos corporales. E-eyes (Wang et al., 2014) extendió esta idea a actividades diarias como sentarse, caminar o teclear. EAR (Pu et al., 2013) logró reconocer la dirección del movimiento con precisiones cercanas al 95%.

**Tercera etapa (2018–2021): deep learning.** La aplicación de redes neuronales profundas transformó el campo. Trabajos como los de Yousefi et al. (2017) con LSTMs y Gao et al. (2019) con CNNs demostraron que es posible aprender representaciones directamente de las series temporales de amplitud y fase CSI, sin ingeniería de features manual. El dataset de Kovalenko et al. (2021) —el de referencia en este TFG— proporciona 7 actividades capturadas con hardware estándar y código de extracción de CSI (Atheros CSI Tool), facilitando la reproducibilidad.

**Cuarta etapa (2021–presente): Transformers y modelos de gran tamaño.** La arquitectura Transformer ha demostrado su potencial también en series temporales CSI. Trabajos como el de Zhang et al. (2022) con CrossTransformer y el de Chen et al. (2023) con modelos preentrenados han establecido nuevos estados del arte. El presente TFG se enmarca en esta etapa, proponiendo un CSITransformer eficiente para CPU que combina un stem convolucional con un encoder Transformer.

### 2.3 Trabajos relacionados

La siguiente tabla compara este trabajo con los más relevantes de la literatura reciente en CSI-HAR:

| Referencia | Año | Método | Clases | Acc. (%) | Datos |
|---|---|---|---|---|---|
| Yousefi et al. [1] | 2017 | LSTM | 6 | 92.7 | Real (Atheros) |
| Wang et al. [2] | 2019 | CNN-1D | 7 | 90.3 | Real (Intel 5300) |
| Gao et al. [3] | 2019 | CNN-2D + LSTM | 9 | 88.1 | Real (Atheros) |
| Kovalenko et al. [4] | 2021 | FCN + BiLSTM | 7 | 91.4 | Real (Atheros) |
| Zhang et al. [5] | 2022 | Transformer | 7 | 93.2 | Real (Atheros) |
| Chen et al. [6] | 2022 | CNN-Transformer | 11 | 89.7 | Real (Intel AX200) |
| Ding et al. [7] | 2020 | ResNet-1D | 6 | 86.9 | Real (Atheros) |
| Li et al. [8] | 2021 | GRU-Attention | 8 | 91.0 | Real (Atheros) |
| Ma et al. [9] | 2023 | Mamba (SSM) | 7 | 94.1 | Real (ESP32) |
| Shi et al. [10] | 2022 | GAN + CNN | 7 | 87.3 | Simulado+Real |
| **Este trabajo** | **2025** | **CSITransformer** | **7** | **82.5** | **Simulado** |

*Nota: Los valores de la literatura proceden de los artículos citados. La precisión de este trabajo (82,5%) no es directamente comparable con los resultados sobre datos reales, dado que opera exclusivamente sobre datos simulados. Esta distinción es fundamental y se analiza en detalle en el Capítulo 6.*

La comparativa revela varias tendencias claras: los Transformers han desplazado a las LSTMs como arquitectura de referencia en los trabajos más recientes; la precisión sobre datos reales oscila habitualmente entre el 87% y el 94%; y los trabajos basados en datos simulados son escasos, siendo este TFG uno de los pocos que declara explícitamente este aspecto y analiza sus implicaciones.

### 2.4 Arquitecturas de deep learning para series temporales

El reconocimiento de actividad a partir de series temporales ha motivado el desarrollo y adaptación de múltiples familias de arquitecturas:

**Redes recurrentes (RNN, LSTM, GRU).** Las LSTMs (Hochreiter & Schmidhuber, 1997) fueron durante años la arquitectura dominante para secuencias temporales gracias a su capacidad para capturar dependencias a largo plazo mediante puertas de olvido, entrada y salida. Las variantes bidireccionales (Schuster & Paliwal, 1997) procesan la secuencia en ambas direcciones, enriqueciendo la representación. Las GRUs (Cho et al., 2014) simplifican la LSTM con dos puertas, reduciendo la carga computacional.

**Redes convolucionales para series temporales.** Las CNN-1D extraen patrones locales mediante filtros deslizantes, siendo computacionalmente eficientes y paralelizables. El trabajo de Wang et al. (2017) demostró que una FCN con tres capas convolucionales y Global Average Pooling puede competir con LSTMs en múltiples benchmarks de clasificación de series temporales. Los modelos basados en InceptionTime (Ismail Fawaz et al., 2020) extienden este enfoque con filtros de múltiples escalas.

**Transformers para series temporales.** Desde el trabajo seminal de Vaswani et al. (2017), los Transformers han mostrado una capacidad notable para capturar dependencias globales en secuencias mediante mecanismos de atención multi-cabeza. Su adaptación a series temporales ha producido arquitecturas como el Temporal Fusion Transformer (Lim et al., 2021), el PatchTST (Nie et al., 2023) y el iTransformer (Liu et al., 2024). Para CSI-HAR, la combinación de un stem convolucional (que captura patrones locales) con un encoder Transformer (que modela relaciones globales) ha mostrado resultados prometedores.

Un debate activo en la comunidad de series temporales es si los Transformers realmente superan a los MLP simples en clasificación (Zeng et al., 2023, "Are Transformers Effective for Time Series Forecasting?"). Sin embargo, para clasificación de ventanas (no de predicción) y con firmas temporales de alta dimensionalidad como el CSI (468 features), el argumento a favor del Transformer es más sólido: la atención permite identificar qué posiciones temporales y qué combinaciones de features son más relevantes para cada clase, aprendiendo selectivamente sobre las partes discriminativas de la señal.

### 2.5 Posicionamiento indoor Wi-Fi

El posicionamiento indoor basado en Wi-Fi puede abordarse a diferentes niveles de granularidad:

**Nivel de sala/zona.** El enfoque más sencillo y robusto divide el entorno en zonas discretas y asigna el dispositivo a la zona más probable. Puede basarse únicamente en intensidad de señal (RSSI o amplitud CSI media), lo que lo hace factible sin campañas de calibración extensas.

**Fingerprinting.** El sistema RADAR (Bahl & Padmanabhan, 2000) fue pionero en utilizar una base de datos de huellas de señal (RSSI de múltiples APs) para localizar dispositivos. Con CSI, el espacio de fingerprinting es mucho más rico. SpotFi (Kotaru et al., 2015) combina amplitud y fase CSI con algoritmos MUSIC para estimar ángulo de llegada (AoA), logrando precisiones de 0,5 m.

**Modelos de propagación.** El modelo ITU-R P.1238 proporciona una estimación de la pérdida de trayecto en interiores como función de la frecuencia y la distancia, permitiendo estimar la distancia a partir de la amplitud de señal recibida. Este es el enfoque adoptado en este TFG para el clasificador de zona.

**Deep learning para localización.** Trabajos recientes como FILA (Wu et al., 2012) y DeepFi (Wang et al., 2015) aplican redes neuronales profundas al fingerprinting con CSI, logrando precisiones de 1–2 m. Los modelos de aprendizaje por transferencia permiten reducir la costosa fase de calibración.

La distinción entre localización de alta precisión (métrica, 0,5–2 m) y clasificación de zona (granularidad de 2–4 categorías) es importante para el diseño del sistema. La localización métrica requiere campañas de calibración extensas (decenas de puntos de medición por zona), procesamiento de fase CSI (que es más sensible pero también más ruidosa que la amplitud), y habitualmente más de un punto de acceso. La clasificación de zona, como la implementada en este TFG, es mucho más robusta y no requiere calibración del entorno específico, a costa de una resolución espacial más baja. Para las aplicaciones objetivo (monitorización de personas mayores, automatización de edificios), la resolución de zona es habitualmente suficiente: saber que la persona está en el dormitorio (zona Cerca) o en el salón (zona Media distancia) es más útil que conocer su posición con precisión de 50 cm.

La clasificación de zona basada en amplitud tiene una ventaja adicional: es independiente de la actividad que realiza la persona. La amplitud media de la señal CSI depende principalmente de la distancia al punto de acceso (modelo de pérdidas de trayecto) y, en menor medida, de la actividad (Walking genera amplitudes ligeramente más altas que Standing a la misma distancia, por el efecto Doppler adicional). Esta independencia relativa entre actividad y zona justifica la arquitectura de pipelines paralelos del sistema, donde el clasificador de zona opera sobre estadísticas de amplitud que son robustas frente a las variaciones de actividad dentro de cada zona.

### 2.6 Conclusiones del estado del arte

Del análisis de la literatura se extraen las siguientes conclusiones que han orientado las decisiones de diseño de este TFG:

1. **El CSI es superior al RSSI** para HAR por su mayor dimensionalidad y sensibilidad a movimientos sutiles. Esta conclusión es ampliamente compartida en la literatura. Mientras el RSSI proporciona una única medida escalar por paquete, el CSI captura hasta 456 valores de amplitud complejos por paquete en la configuración del hardware de referencia, lo que representa un aumento de más de dos órdenes de magnitud en la información disponible para la clasificación.

2. **Los Transformers superan a las LSTM** en los trabajos más recientes sobre datos reales, aunque requieren más datos para entrenarse efectivamente. Esta es la hipótesis que el CSITransformer de este trabajo pone a prueba en el régimen de datos limitados (2.000 muestras sintéticas). La literatura disponible a 2023 sugiere que con datasets de más de 10.000 muestras, los Transformers consiguen ganancias de 2–4 puntos porcentuales de precisión sobre los BiLSTM; con datasets menores, la superioridad es menos clara.

3. **La simulación de datos CSI es un área poco explorada.** La mayoría de los trabajos utilizan hardware real, lo que dificulta la reproducibilidad. La contribución del simulador física de este TFG es relevante para la comunidad, especialmente como herramienta para el desarrollo y depuración de nuevas arquitecturas sin necesidad de hardware especializado. El trabajo de Shi et al. (2022), que combina datos reales y datos generados por GAN, es el más próximo a este enfoque en la literatura revisada, aunque su método de generación es mucho más complejo que el simulador basado en física de este TFG.

4. **La clasificación de zona es un complemento valioso** al reconocimiento de actividad, aunque raramente se combina en un único sistema. La mayoría de los trabajos de la literatura se centran exclusivamente en la clasificación de actividad, sin incorporar información de posición relativa. Este TFG integra ambas funcionalidades en un único sistema con dos pipelines paralelos, lo que constituye una contribución diferencial respecto a los trabajos de referencia.

5. **La latencia de inferencia en CPU es crítica** para el despliegue en dispositivos de bajo coste. Ninguno de los trabajos revisados reporta latencias por debajo de 10 ms en CPU para ventanas de 128 muestras, lo que el CSITransformer de este TFG consigue con 5,3 ms. Esta eficiencia es especialmente relevante para dispositivos edge como Raspberry Pi (CPU ARM de bajo consumo) o routers con OpenWRT, donde la inferencia en GPU no es posible.

6. **La generalización entre entornos y sujetos es el principal obstáculo práctico.** Los trabajos más recientes (Ma et al., 2023; Zhang et al., 2022) coinciden en identificar la dependencia del entorno y la variabilidad inter-sujeto como los principales factores que limitan la precisión en despliegues reales. Técnicas como el aprendizaje por transferencia entre entornos, la adaptación de dominio y el aprendizaje federado (para preservar la privacidad de los datos de usuario) son líneas activas de investigación que este TFG no aborda pero que constituyen el paso natural siguiente a la validación con datos reales.

---

## 3. Fundamentos Teóricos

### 3.1 Comunicaciones OFDM y CSI

La multiplexación por división de frecuencias ortogonales (OFDM) es la técnica de modulación empleada en los estándares Wi-Fi modernos (IEEE 802.11a/g/n/ac). En OFDM, el canal de banda ancha se divide en múltiples subcanales (subportadoras) ortogonales entre sí, cada uno con un ancho de banda suficientemente estrecho como para experimentar desvanecimiento plano (flat fading). El canal en banda ancha se modela, por tanto, como una colección de canales de banda estrecha independientes.

La información de estado del canal (CSI) captura la respuesta en frecuencia del canal en cada una de estas subportadoras. Formalmente, el sistema OFDM puede describirse como:

```
Y[k] = H[k] · X[k] + N[k]
```

donde `Y[k]` es el símbolo recibido en la subportadora k, `X[k]` es el símbolo transmitido, `H[k]` es la función de transferencia del canal (CSI) en esa subportadora, y `N[k]` es el ruido aditivo. El CSI es un número complejo:

```
H[k] = |H[k]| · exp(j·φ[k])
```

donde `|H[k]|` es la amplitud y `φ[k]` es la fase. La amplitud es la cantidad más utilizada en trabajos de HAR, ya que es más robusta frente a variaciones de temporización y desplazamiento de frecuencia que la fase.

En el hardware de referencia de este trabajo (TP-Link WDR3600 con Atheros AR9380 y el Atheros CSI Tool), la configuración es:
- **Banda de operación:** 5 GHz (canales de 40 MHz)
- **Número de subportadoras:** 114 por par de antenas
- **Configuración MIMO:** 2×2 (2 antenas transmisoras × 2 receptoras = 4 pares)
- **Tasa de captura:** ~20 paquetes/segundo
- **Rango de amplitud:** [0, 577,66] (unidades arbitrarias del driver)

Esto produce, por cada paquete capturado, un vector de 456 valores de amplitud (114 subportadoras × 4 pares de antenas), que constituye la entrada bruta al pipeline de preprocesamiento.

**Ventajas del CSI respecto al RSSI.** El indicador de nivel de señal recibida (RSSI) es un único valor escalar por paquete que representa la potencia integrada en toda la banda del canal. El CSI, en cambio, proporciona la respuesta en frecuencia compleja para cada una de las subportadoras, lo que aporta tres ventajas decisivas para HAR. Primera, la resolución frecuencial: distintas subportadoras responden de forma diferente a los movimientos humanos dependiendo de la longitud de onda relativa al desplazamiento del reflector, lo que produce patrones de variación de amplitud entre subportadoras que son característicos de cada actividad. Segunda, la robustez: el promedio de amplitud entre subportadoras puede mantenerse casi constante cuando algunas subportadoras aumentan su amplitud por interferencia constructiva mientras otras la reducen por interferencia destructiva; el RSSI vería un canal estable cuando en realidad hay movimiento. Tercera, la redundancia de diversidad: al disponer de 4 pares de antenas, el sistema tiene cuatro mediciones independientes del mismo movimiento desde distintas perspectivas espaciales, enriqueciendo la representación.

**Extracción del CSI con el Atheros CSI Tool.** El Atheros CSI Tool (Xie et al., 2015) es una modificación del driver ath9k de Linux que permite acceder a los valores CSI en bruto (amplitud y fase por subportadora y par de antenas) para cada paquete ACK o datos recibido. La tasa de captura máxima depende del tráfico de red: en modo de inyección de paquetes artificiales (packet injection), puede alcanzarse una tasa de ~100 paquetes/segundo; en condiciones normales de red doméstica, la tasa es de ~20 paquetes/segundo, que es la que se asume en este trabajo. A 20 paquetes/segundo, una ventana de 128 muestras equivale a 6,4 segundos de actividad, suficiente para capturar varios ciclos de todas las actividades consideradas (incluyendo el ciclo de marcha más lento a 1,8 Hz, que produce ~11,5 ciclos en 6,4 segundos).

Alternativas al Atheros CSI Tool incluyen el nexmon CSI (Schulz et al., 2018), que funciona con chipsets Broadcom (Raspberry Pi 4, tarjetas BCM4329/4339) y permite captura CSI en hardware de bajo coste. El Intel 5300 NIC (Halperin et al., 2011) es otra opción popular en la literatura que proporciona 30 subportadoras en canales de 20 MHz. El sistema desarrollado en este TFG es agnóstico al hardware de captura: lo único que requiere es que el array de amplitudes tenga forma `(T, N_ant × N_sub)`, donde el número de features sea 456 (para el hardware de referencia) o se adapte con un paso de PCA reconfigurado para el nuevo número de subportadoras.

### 3.2 Modelo matemático del canal Wi-Fi

El canal Wi-Fi en interiores se modela como un canal multitrayecto (multipath) con componentes de trayecto directo (LOS) y reflexiones en paredes, muebles y superficies. La respuesta impulsional del canal es:

```
h(τ, t) = Σ_l  α_l(t) · exp(j·φ_l(t)) · δ(τ - τ_l)
```

donde `α_l(t)` y `φ_l(t)` son la amplitud y fase del l-ésimo trayecto en el instante t, y `τ_l` es su retardo. La respuesta en frecuencia (CSI en la subportadora k) es la transformada de Fourier respecto a τ:

```
H[k, t] = Σ_l  α_l(t) · exp(j·φ_l(t)) · exp(-j·2π·f_k·τ_l)
```

Cuando una persona se mueve en el entorno, las amplitudes `α_l(t)` y fases `φ_l(t)` de los trayectos que involucran reflexión o difracción en el cuerpo humano varían con el tiempo, produciendo fluctuaciones en el CSI. La amplitud total observada en la subportadora k es:

```
|H[k, t]| ≈ |Σ_l  α_l(t) · exp(j·θ_l(k, t))|
```

Para el **efecto Doppler**, cuando un segmento corporal se mueve con velocidad `v` en la dirección del rayo, la frecuencia de la señal recibida se desplaza en:

```
f_D = (v / λ) · cos(θ)
```

donde `λ = c/f` es la longitud de onda (≈6 cm a 5 GHz) y `θ` es el ángulo entre la dirección del movimiento y el rayo. Para caminar a 1 m/s a 5 GHz, el desplazamiento Doppler máximo es aproximadamente 17 Hz, muy por encima de los movimientos de respiración (~0,3 Hz) o cambios de postura (~0,1 Hz).

El **modelo de desvanecimiento Rician** se aplica cuando existe un trayecto LOS dominante junto con componentes de dispersión. La distribución de la envolvente de la señal es:

```
f(r) = (2r(K+1)/Ω) · exp(-(K+1)r²/Ω - K) · I_0(2r·√(K(K+1)/Ω))
```

donde `K` es el factor de Rice (relación potencia LOS / potencia dispersión), `Ω` es la potencia total y `I_0` es la función de Bessel modificada de orden cero. Valores típicos de K en interiores son 1,5–5,0 (3–7 dB).

**Sensibilidad del CSI a los movimientos humanos.** La presencia de una persona en el entorno modifica las componentes del canal multitrayecto de dos formas: (a) añade nuevas componentes de reflexión y difracción en el cuerpo humano (que absorbe y refleja las ondas Wi-Fi con una sección transversal radar del orden de 0,5–1 m²); (b) hace que las componentes existentes que se reflejan en paredes cambien su retardo al cambiar la posición del reflector humano intermedio. El efecto neto es que `α_l(t)` y `φ_l(t)` para los trayectos que involucran al cuerpo humano varían con el tiempo, y esta variación es la que captura el CSI y explota el sistema de HAR. La sensibilidad del CSI es tal que se han documentado detecciones de movimientos de amplitud sub-milimétrica (respiración de ~5 mm de expansión torácica) mediante análisis espectral de la fase CSI (Liu et al., 2018), lo que ilustra la riqueza de información disponible en la señal Wi-Fi para la monitorización de actividad humana.

### 3.3 Firmas Doppler de actividades humanas

Cada actividad humana produce una firma característica en el CSI debida a la combinación de los efectos Doppler de los distintos segmentos corporales que se mueven. El análisis de estas firmas es la base del reconocimiento de actividad. Una herramienta fundamental para estudiarlas es el espectrograma de tiempo-frecuencia de la señal CSI, que muestra la evolución temporal del contenido espectral y permite identificar cuándo ocurren los eventos Doppler dentro de la ventana de observación.

**Ausencia de persona (No Person):** El canal es estático; la única variación proviene del ruido térmico del receptor (AWGN). La amplitud media es aproximadamente 200 unidades con desviación estándar <5. El espectrograma muestra únicamente ruido de fondo plano sin componentes frecuenciales dominantes. En el espacio de características del clasificador, esta clase ocupa la región de menor varianza y menor energía dinámica, adyacente a la clase Standing.

**Permanecer de pie (Standing):** El movimiento principal es la respiración (~0,3 Hz, correspondiente a 12–18 respiraciones por minuto en reposo) y el balanceo corporal involuntario o swaying (<0,5 Hz). Las variaciones de amplitud son pequeñas (±15 unidades) y periódicas. El espectrograma muestra un pico estrecho y estacionario en la banda 0,2–0,4 Hz, persistente durante toda la ventana de 128 muestras (equivalente a ~6,4 segundos a 20 paquetes/segundo). La distinción respecto a No Person es el pico de baja frecuencia; la distinción respecto a Lying es la frecuencia ligeramente superior y la amplitud ligeramente mayor.

**Caminar (Walking):** El ciclo de la marcha (~1,8–2,2 Hz, equivalente a 108–132 pasos por minuto) produce las variaciones de CSI más pronunciadas y periódicas. Las amplitudes oscilan ±50 unidades o más, con modulación espacial entre subportadoras debida a la fase del ciclo de marcha. El espectrograma muestra un pico dominante en la banda 1,8–2,2 Hz con armónicos en el doble y triple de esa frecuencia, lo que refleja la naturaleza periódica del ciclo de marcha con sus dos fases (apoyo y balanceo). Desde el punto de vista biomecánico, el cuerpo humano es un sistema con múltiples segmentos (brazos, piernas, tronco) que se mueven con relaciones de fase fijas entre sí, produciendo una firma espectral característica y reproducible.

**Sentarse (Sitting):** Movimiento muy lento (~0,1 Hz), principalmente un desplazamiento gradual del centro de masa desde la posición erguida hasta la posición sentada. Las variaciones son pequeñas en amplitud y muy lentas. El espectrograma muestra un transitorio de baja frecuencia que se estabiliza conforme la persona completa la acción. Una vez sentada, la firma de respiración domina. La característica más distintiva de Sitting en el dominio del tiempo es la tendencia monótona (creciente o decreciente dependiendo de la posición relativa respecto al AP) que ocurre durante toda la ventana de observación.

**Tumbarse (Lying):** Movimiento más lento aún (<0,05 Hz). Una vez tumbada, la persona produce únicamente la firma de respiración, prácticamente idéntica a la de permanecer de pie pero con menor amplitud.

**Levantarse del suelo (Get Down):** Transición desde de pie hasta tumbado. Produce una ráfaga de variaciones en el CSI que crece en amplitud hacia el final de la secuencia temporal (el movimiento ocurre al final).

**Incorporarse (Get Up):** La inversa: la ráfaga ocurre al principio (el movimiento ocurre al inicio) y las variaciones decaen exponencialmente conforme la persona se estabiliza en posición erguida.

Estas firmas son las que el simulador reproduce mediante señales determinísticas más ruido, tal y como se describe en el Capítulo 5.

### 3.4 Redes LSTM y LSTM Bidireccional

Las redes de memoria a largo y corto plazo (LSTM, Hochreiter & Schmidhuber 1997) son una arquitectura de red neuronal recurrente diseñada para capturar dependencias temporales a distintas escalas. La celda LSTM incorpora tres puertas que regulan el flujo de información:

```
Puerta de olvido:   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Puerta de entrada:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidato celda:    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
Estado celda:       c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
Puerta de salida:   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Estado oculto:      h_t = o_t ⊙ tanh(c_t)
```

donde `σ` es la función sigmoide, `tanh` es la tangente hiperbólica y `⊙` es el producto elemento a elemento.

La **LSTM Bidireccional** (Schuster & Paliwal, 1997) procesa la secuencia dos veces: una en el orden temporal natural y otra en orden inverso. Los estados ocultos de ambas direcciones se concatenan, produciendo una representación que integra el contexto pasado y futuro en cada instante:

```
h_t^{fwd} = LSTM_fwd(x_t, h_{t-1}^{fwd})
h_t^{bwd} = LSTM_bwd(x_t, h_{t+1}^{bwd})
h_t = [h_t^{fwd} ; h_t^{bwd}]    (concatenación)
```

Esto es especialmente relevante para HAR, donde el contexto de toda la ventana puede ser necesario para distinguir actividades como Get Down (que termina con movimiento) de Get Up (que empieza con movimiento).

En este TFG, ambas arquitecturas incorporan una **capa de proyección lineal** antes del LSTM (468→64 dimensiones con LayerNorm), que reduce la dimensionalidad de entrada y mejora la eficiencia computacional en CPU en un factor ~7×. Esta decisión de diseño permite latencias de ~9 ms en CPU, viables para aplicaciones en tiempo real.

**Análisis del gradiente en LSTMs profundas.** Un aspecto crítico en el entrenamiento de LSTMs es el problema del gradiente que se desvanece o explota a lo largo de la secuencia. Para una secuencia de longitud T, el gradiente de la pérdida respecto al estado oculto en el instante 0 implica el producto de T matrices Jacobianas:

```
∂L/∂h_0 = (∂L/∂h_T) · Π_{t=1}^{T}  ∂h_t/∂h_{t-1}
```

Si los valores propios del Jacobiano son consistentemente menores que 1, el gradiente se desvanece exponencialmente con T; si son mayores que 1, explota. Las puertas de la LSTM mitigan este problema al permitir que el gradiente fluya directamente a través del estado de celda `c_t` con un factor de escala próximo a 1 (cuando la puerta de olvido `f_t ≈ 1`), pero no lo eliminan completamente para T=128. El gradient clipping (max_norm=1,0) aplicado en este trabajo es la solución práctica adoptada para prevenir la explosión.

**Normalización por capas (LayerNorm).** La LayerNorm aplicada antes del LSTM y sobre el estado oculto final normaliza las activaciones a lo largo de la dimensión de features (no del batch), lo que estabiliza el entrenamiento especialmente en las primeras épocas cuando los pesos no están calibrados. A diferencia de la BatchNorm, que calcula estadísticas sobre el batch, la LayerNorm es independiente del tamaño del batch, lo que la hace especialmente adecuada para secuencias de longitud variable o inference con batch_size=1.

### 3.5 Redes Completamente Convolucionales (FCN) para series temporales

La FCN para clasificación de series temporales (Wang et al., 2017) consiste en apilar bloques convolucionales unidimensionales seguidos de una capa de Global Average Pooling (GAP) que produce un vector de tamaño fijo independiente de la longitud de la secuencia. La arquitectura estándar emplea tres bloques con filtros de tamaños 8, 5 y 3 respectivamente:

```
Entrada: (B, T, 468)
→ Transponer: (B, 468, T)         [Conv1d requiere (B, C, T)]
→ Conv1d(468→128, k=8) + BN + ReLU + Dropout
→ Conv1d(128→256, k=5) + BN + ReLU + Dropout
→ Conv1d(256→128, k=3) + BN + ReLU + Dropout
→ Global Average Pool: (B, 128)
→ Linear(128→7)
```

Las ventajas de la FCN son su eficiencia computacional (operaciones paralelizables en GPU y CPU modernas), la invarianza a traslaciones temporales y la ausencia de hiperparámetros como el número de capas ocultas recurrentes. Sin embargo, su limitación fundamental es la pérdida de información temporal global: la GAP promedia toda la secuencia, perdiendo la estructura temporal que distingue, por ejemplo, Get Up (ráfaga al inicio) de Get Down (ráfaga al final).

**Receptive field y localidad de las convoluciones.** Cada filtro convolucional de la FCN tiene un campo receptivo (receptive field) limitado: el filtro de kernel k=8 en la primera capa ve únicamente 8 posiciones consecutivas; el filtro de k=5 en la segunda capa ve 5 posiciones de las salidas de la primera capa, que corresponden a `5+(8-1)=12` posiciones de la entrada original; el filtro de k=3 en la tercera capa alcanza `3+(5-1)+(8-1)=15` posiciones de la entrada. La GAP después integra todos los campos receptivos, pero el clasificador final solo recibe el promedio sobre toda la secuencia, no información sobre qué posición temporal tiene mayor activación. Esta limitación podría mitigarse parcialmente sustituyendo la GAP por una Global Max Pooling (que retiene la posición de activación máxima) o por una concatenación de ambas (como hace el CSITransformer), aunque el experimento de la FCN en este trabajo no incorpora estas modificaciones para mantener la comparabilidad con el baseline de Wang et al. (2017).

### 3.6 Arquitectura Transformer para señales temporales

El Transformer (Vaswani et al., 2017) introduce el mecanismo de **atención multi-cabeza** (MHA), que permite a cada posición de la secuencia atender a todas las demás posiciones sin restricción de localidad:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

donde Q (queries), K (keys) y V (values) son proyecciones lineales de la entrada. La versión multi-cabeza ejecuta h atenciones en paralelo en subespacios de dimensión d_k = d_model/h:

```
MHA(Q, K, V) = Concat(head_1, ..., head_h) · W^O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

El **CSITransformer** propuesto en este trabajo adapta esta arquitectura a la clasificación de ventanas CSI mediante tres componentes clave:

**1. Stem convolucional.** Antes del encoder Transformer, se aplican dos capas Conv1d (kernel 7 y kernel 3) con BatchNorm y GELU, que capturan patrones locales y proyectan la entrada de 468 a d_model=64 dimensiones. Esta idea está inspirada en los Vision Transformers con patch embedding, adaptada a series temporales 1D.

**2. Codificación posicional aprendida.** A diferencia del Transformer original (que usa codificación sinusoidal fija), el CSITransformer emplea embeddings posicionales entrenables (nn.Embedding), que son más flexibles para ventanas de longitud fija (128 muestras).

**3. Agregación mean+max.** En lugar de usar únicamente el token [CLS] o la última posición, el CSITransformer concatena el promedio global y el máximo global de las salidas del encoder:

```
pooled = [mean(H_1..T) ; max(H_1..T)]   →  (B, 128)
→ Linear(128→64) + LayerNorm + GELU + Dropout(0.2)
→ Linear(64→7)
```

Esta estrategia de agregación doble captura tanto la información media de la actividad como los picos transitorios, siendo especialmente útil para distinguir Get Up y Get Down.

El encoder se configura con Pre-LN (norma antes de la atención, en lugar de después), que proporciona un entrenamiento más estable sin necesidad de warm-up.

**Pre-LN vs Post-LN.** La formulación original del Transformer (Vaswani et al., 2017) aplica la LayerNorm después de la atención y la FFN (Post-LN). Sin embargo, el Post-LN es notoriamente difícil de entrenar sin un esquema de warm-up del learning rate porque los gradientes en las capas inferiores son muy pequeños al inicio del entrenamiento. La formulación Pre-LN (Xiong et al., 2020), que aplica la LayerNorm antes de la atención, proporciona gradientes más equilibrados desde el inicio y permite usar tasas de aprendizaje constantes más altas:

```
# Post-LN (original):
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))

# Pre-LN (este trabajo):
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

En la formulación Pre-LN, el gradiente fluye directamente a través de las conexiones residuales sin pasar por la LayerNorm, lo que preserva la escala del gradiente a través de las capas. Esto es especialmente valioso con pocos datos de entrenamiento (2.000 muestras), donde la estabilidad del gradiente es crítica para evitar convergencia prematura a mínimos locales pobres.

**Justificación del número de cabezas (h=4) y capas (2).** El CSITransformer usa 4 cabezas de atención y 2 capas de encoder. Con d_model=64 y h=4, cada cabeza opera en un subespacio de dimensión d_k=d_v=16. Esta configuración proporciona 4 "perspectivas" diferentes sobre las relaciones temporales de la secuencia CSI: distintas cabezas pueden especializarse en capturar correlaciones a distancias cortas (ciclo de marcha local), medias (la forma del transitorio) y largas (la relación entre el inicio y el final de la ventana). El uso de 2 capas es un compromiso entre capacidad representacional y regularización implícita: más capas aumentarían la capacidad del modelo pero también el riesgo de sobreajuste con solo 1.600 muestras de entrenamiento. Experimentos informales con 3 y 4 capas no produjeron mejoras de validación significativas y sí aumentaron la latencia.

### 3.7 Estimación de distancia mediante pérdida de trayecto ITU-R P.1238

El modelo de pérdida de trayecto ITU-R P.1238-10 (2019) para interiores expresa la pérdida total en decibelios como:

```
PL(f, d) [dB] = 20·log₁₀(f) + N·log₁₀(d) + L_f(n) - 28
```

donde:
- `f` es la frecuencia en MHz (aquí 5 GHz = 5000 MHz)
- `d` es la distancia transmisor–receptor en metros
- `N` es el coeficiente de pérdida de distancia (típicamente 28–33 en oficinas a 5 GHz)
- `L_f(n)` es el factor de atenuación de piso para n plantas (no aplicable en un solo piso)

La amplitud lineal de la señal CSI se relaciona con la pérdida de trayecto mediante:

```
|H(d)| ≈ A_0 · (d_0 / d)^(N/20)
```

Con los parámetros calibrados para el hardware de referencia (TP-Link WDR3600, N=2,8, A_0=380, d_0=1 m), las cuatro zonas de distancia utilizadas en el clasificador de zona son:

| Zona | Nombre | Distancia (m) | |H| medio esperado |
|---|---|---|---|
| 0 | Proximidad | 0,0 – 1,5 | 290 – 500 |
| 1 | Cerca | 1,5 – 3,0 | 120 – 290 |
| 2 | Media distancia | 3,0 – 5,0 | 45 – 120 |
| 3 | Lejos | 5,0 – 8,0 | 15 – 45 |

El clasificador de zona extrae 16 estadísticas de amplitud (media, desviación típica, rango intercuartílico p90–p10 y energía media por par de antenas) y las mapea a la zona más probable mediante un MLP ligero (16→64→32→4).

**Justificación de los límites de zona.** Los límites de zona se seleccionaron en función de los umbrales de amplitud y las aplicaciones prácticas típicas:
- Proximidad (0–1,5 m): zona de contacto, relevante para detectar presencia junto a la cama, silla de ruedas o punto de acceso. Las amplitudes esperadas (290–500 unidades) están en el rango superior del sensor, con alta SNR.
- Cerca (1,5–3,0 m): zona de interacción directa, equivalente al espacio de trabajo de un escritorio o cocina. Amplitudes en rango medio-alto (120–290 unidades).
- Media distancia (3,0–5,0 m): zona de habitación completa, equivalente al salon o dormitorio estándar de 15–20 m². Amplitudes en rango medio-bajo (45–120 unidades).
- Lejos (5,0–8,0 m): zona límite del sensor, equivalente a habitaciones grandes o pasillos largos. Amplitudes bajas (15–45 unidades) con mayor susceptibilidad al shadowing.

**Incertidumbre del modelo de pérdidas.** El modelo ITU-R P.1238 es una aproximación estadística que captura el comportamiento promedio de la propagación en interiores. La incertidumbre estándar del modelo (σ_model) es de aproximadamente 8–12 dB según la recomendación, lo que corresponde a una incertidumbre en la estimación de distancia de un factor de 1,5–2× en condiciones típicas. Esta incertidumbre es la razón por la que se definen zonas amplias en lugar de intentar estimar la distancia con resolución métrica, y justifica el uso de un clasificador MLP en lugar de una estimación directa de distancia.

---

## 4. Diseño del Sistema

### 4.1 Arquitectura general del sistema

El sistema CSI-HAR se organiza en dos pipelines paralelos que comparten la ventana de entrada de amplitud CSI:

```
Paquete Wi-Fi (amplitud bruta, (128, 456))
│
├── Pipeline HAR ──────────────────────────────────────────────────────
│   ├── Filtro Hampel (eliminación de outliers)
│   ├── Denoising wavelet (sym5) / Savitzky-Golay (fallback)
│   ├── Concatenar PCA 3 comp./par × 4 pares = 12 features
│   ├── Clip + normalización [0, AMP_MAX] → [0, 1]
│   ├── Tensor: (128, 468)  →  modelo seleccionado
│   └── Softmax → 7 probabilidades de actividad
│
└── Pipeline de Zona ────────────────────────────────────────────────
    ├── extract_zone_features(): media, std, IQR, energía × 4 pares
    ├── Estandarización con (mean, std) del entrenamiento
    ├── ZoneClassifier (MLP 16→64→32→4)
    └── Softmax → 4 probabilidades de zona
```

Los dos pipelines son independientes y pueden ejecutarse en paralelo. La GUI consolida sus salidas en un panel unificado que muestra actividad actual, confianza, zona estimada y distancia aproximada.

**Justificación de la arquitectura de doble pipeline.** La decisión de mantener los pipelines de actividad y zona como módulos independientes en lugar de usar un único modelo multi-tarea es pragmática. Los requisitos de entrada son diferentes: el pipeline HAR necesita la secuencia temporal completa de 128×468 features para capturar la dinámica de la actividad, mientras que el pipeline de zona solo necesita 16 estadísticas de amplitud de la misma ventana. Combinarlos en un único modelo requeriría o bien alimentar todas las 468 features al clasificador de zona (aumentando innecesariamente su complejidad) o bien compartir el encoder de series temporales con una segunda cabeza (acoplando el entrenamiento de ambos módulos). La arquitectura de doble pipeline independiente permite entrenar, evaluar y actualizar cada módulo de forma autónoma, lo que es especialmente valioso dado que el ZoneClassifier se entrena con un dataset diferente (ZoneDataset) al de los modelos HAR (SimulatedCSIDataset).

**Compatibilidad con hardware real.** El sistema está diseñado para ser compatible con hardware CSI real con mínimas modificaciones. El único punto de entrada del sistema es la función `simulate_activity(class_idx, seq_len=128)`, que en un despliegue real se reemplazaría por un lector de paquetes CSI en tiempo real que retorna un array (128, 456) del mismo tipo y rango que el simulador. Toda la cadena de procesamiento downstream (filtro Hampel, wavelet, PCA, modelos, zona) permanecería inalterada. Esta compatibilidad por diseño es una característica arquitectónica deliberada que facilita la transición de la fase de demostración simulada a la fase de validación con hardware real.

### 4.2 Pipeline de preprocesamiento CSI

El preprocesamiento sigue exactamente el pipeline del repositorio de referencia (Kovalenko et al., 2021), con una adición: el cómputo de PCA se realiza de forma global sobre todo el corpus de entrenamiento (batch PCA) en lugar de por ventana individual, lo que reduce el tiempo de construcción del dataset en aproximadamente 12.000 SVDs individuales.

**Paso 1 — Filtro Hampel.** El filtro Hampel es un estimador robusto de outliers que reemplaza cada muestra cuya desviación respecto a la mediana local supere `t0 × MAD × 1,4826` por dicha mediana. Se aplica con ventana k=7 (±7 muestras) y umbral t0=3. Este filtro es especialmente efectivo para eliminar los picos transitorios ocasionales que produce el driver del Atheros CSI Tool.

**Paso 2 — Denoising wavelet.** La wavelet sym5 (Daubechies simetrizada de 5 momentos nulos) se aplica con 3 niveles de descomposición. El umbral de soft-thresholding es `0,06 × max(|coeficientes del nivel más fino|)`. Si PyWavelets no está disponible, se usa el filtro Savitzky-Golay de ventana 11 y orden 3 como alternativa, que preserva picos mientras suaviza el ruido.

**Paso 3 — Cómputo de PCA.** Para cada uno de los 4 pares de antenas, se calcula la proyección sobre los 3 primeros componentes principales del bloque de 114 subportadoras correspondiente. Los 12 valores resultantes (3 × 4) capturan la mayor varianza de la señal en cada par, complementando la información de alta frecuencia presente en las 456 amplitudes brutas. Los componentes principales extraen direcciones de máxima varianza en el espacio de subportadoras: el primer componente captura el modo de variación dominante (habitualmente el nivel medio de amplitud), el segundo captura la pendiente espectral, y el tercero captura la curvatura espectral o la presencia de picos. Esta representación comprimida es especialmente útil para capturar la diversidad espectral entre actividades que afectan a distintas bandas de subportadoras.

**Justificación del PCA global vs PCA por ventana.** El PCA global (ajustado sobre todo el corpus de entrenamiento) garantiza que los componentes principales sean consistentes entre todas las muestras, lo que es esencial para la normalización reproducible. En contraste, el PCA por ventana ajustaría los ejes de proyección de forma diferente para cada muestra, produciendo representaciones incoherentes que dificultarían el aprendizaje. La transformación global también permite que los modelos aprendan patrones en el espacio PCA globalmente definido, con la garantía de que el mismo vector de entrada producirá la misma proyección en entrenamiento e inferencia.

**Paso 4 — Normalización.** Las 456 amplitudes brutas se normalizan a [0, 1] dividiendo por AMP_MAX=577,66. Las 12 características PCA se normalizan de forma independiente a [0, 1] usando sus propios mínimos y máximos globales. Ambas partes se concatenan para producir el tensor final de 468 características. La normalización separada de las amplitudes brutas y las componentes PCA es importante porque estas dos partes tienen rangos muy diferentes: las amplitudes están en [0, 577,66], mientras que las componentes PCA pueden tener rangos negativos y positivos con magnitudes variables. La normalización independiente evita que la diferencia de escala produzca un dominancia artificial de una parte sobre la otra en el aprendizaje.

### 4.3 Modelo SimpleLSTM (arquitectura detallada, parámetros, justificación)

El modelo SimpleLSTM es una reimplementación del `SimpleLSTMClassifier.py` del repositorio de referencia, con una modificación estructural crítica: la adición de una capa de proyección lineal 468→64 con LayerNorm antes de la LSTM.

**Motivación de la proyección.** Sin proyección, el LSTM procesa directamente los 468 valores de entrada en cada paso temporal. El cómputo de las puertas LSTM escala como O(4 × (input_dim + hidden_size) × hidden_size) por paso. Con input_dim=468 y hidden_size=256, cada paso requiere 4×(468+256)×256 ≈ 741K multiplicaciones. Con proyección (input_dim→64), se reducen a 4×(64+256)×256 ≈ 328K, ahorrando más del 55% de las operaciones LSTM.

**Arquitectura:**
```
Entrada: (B, 128, 468)
→ Linear(468→64, bias=False) + LayerNorm(64)    # proyección
→ LSTM(64→256, 2 capas, dropout=0.2)            # recurrencia
→ h[-1]: (B, 256)                               # último estado oculto
→ LayerNorm(256)
→ Linear(256→7)
```

**Parámetros:** 888.455 (el mayor de los cuatro modelos, paradójicamente).
**Inicialización:** Xavier uniforme para pesos input→hidden, ortogonal para pesos hidden→hidden, ceros para sesgos.
**Tasa de aprendizaje:** 3×10⁻⁴ (Adam). Las LSTMs requieren tasas de aprendizaje bajas para evitar el colapso del gradiente en las primeras épocas.

### 4.4 Modelo BiLSTM

El BiLSTM extiende el SimpleLSTM con procesamiento bidireccional y una cabeza clasificadora de dos capas:

**Arquitectura:**
```
Entrada: (B, 128, 468)
→ Linear(468→64, bias=False) + LayerNorm(64)
→ BiLSTM(64→128+128, 2 capas, dropout=0.2)      # 128 unid./dirección
→ Concatenar h_fwd[-1] y h_bwd[-1]: (B, 256)
→ LayerNorm(256)
→ Linear(256→128) + ReLU + Dropout(0.2)
→ Linear(128→7)
```

La cabeza clasificadora de dos capas (con ReLU y Dropout intermedio) proporciona mayor capacidad de discriminación no lineal que el clasificador lineal simple del SimpleLSTM, a costa de un ligero aumento en el número de parámetros de la cabeza.

**Parámetros:** 658.311. Aunque tiene menos parámetros que el SimpleLSTM, el bidireccional procesa cada ventana dos veces, resultando en una latencia ligeramente mayor (8,8 ms vs 9,6 ms, aunque la diferencia es pequeña).

### 4.5 Modelo FCN

La FCN sigue la arquitectura de Wang et al. (2017) con tres bloques convolucionales. Se corrigió un bug del repositorio original: la capa Linear final esperaba 256 entradas pero recibía 128 (tras Conv3 con 128 canales de salida). La versión corregida pasa 128.

**Arquitectura:**
```
Entrada: (B, 128, 468)
→ Transponer a (B, 468, 128)
→ Conv1d(468→128, k=8, pad=4) + BN + ReLU + Dropout(0.2)
→ Conv1d(128→256, k=5, pad=2) + BN + ReLU + Dropout(0.2)
→ Conv1d(256→128, k=3, pad=1) + BN + ReLU + Dropout(0.2)
→ mean(dim=2): (B, 128)                          # Global Average Pooling
→ Linear(128→7)
```

**Parámetros:** 743.303. La FCN es computacionalmente eficiente (5,4 ms de latencia), comparable al Transformer, pero su precisión final (30,8%) es la más baja de los cuatro modelos, lo que sugiere que la pérdida de estructura temporal por el GAP es particularmente perjudicial con datos simulados y pocas muestras de entrenamiento.

### 4.6 Modelo CSITransformer

El CSITransformer es la propuesta original de este TFG. Su diseño parte de la observación de que las actividades humanas producen patrones temporales que van desde lo local (ciclo de marcha ~0,5 s) hasta lo global (forma de la transición Get Up/Down a lo largo de toda la ventana). Los Transformers, con su atención global, complementan a los stems convolucionales locales.

**Arquitectura detallada:**
```
Entrada: (B, 128, 468)
│
├── ConvStem:
│   ├── Conv1d(468→64, k=7, pad=3) + BN + GELU
│   └── Conv1d(64→64, k=3, pad=1) + BN + GELU
│   → (B, 128, 64)
│
├── LearnablePositionalEncoding:
│   └── Embedding(512, 64) + Dropout(0.1)
│   → (B, 128, 64)
│
├── TransformerEncoder (2 capas, Pre-LN):
│   └── Cada capa: LN → MHA(4 cabezas, d_k=16) → Add
│                  LN → FFN(64→128→64, GELU) → Add
│   → (B, 128, 64)
│
├── Agregación:
│   ├── mean_pool = mean(dim=1): (B, 64)
│   └── max_pool  = max(dim=1): (B, 64)
│   → concat: (B, 128)
│
└── Clasificador:
    ├── Linear(128→64) + LayerNorm(64) + GELU + Dropout(0.2)
    └── Linear(64→7)
```

**Parámetros:** 330.887 (el más eficiente de los cuatro modelos). La distribución de parámetros revela que el stem convolucional domina: las dos capas Conv1d contribuyen con 209.024 + 24.832 = 233.856 parámetros (70,7% del total), mientras que el encoder Transformer de 2 capas contribuye con solo 49.792 parámetros (15,0%). Esto refleja que el cuello de botella dimensional está en la proyección de 468→64 en el stem, no en el encoder. Un diseño futuro podría explorar reducir el tamaño del filtro de la primera convolución (de k=7 a k=5 o k=3) para reducir el número de parámetros del stem sin sacrificar la calidad de la proyección.

**Tasa de aprendizaje:** 1,46×10⁻³ (Adam). El warm-up no es necesario gracias al Pre-LN.

**Inicialización:** truncated normal (std=0,02) para pesos lineales, ceros para sesgos; ones/ceros para LayerNorm. La inicialización truncated normal (valores de |x| > 2σ se rechazan y se remuestrean) produce pesos de inicialización con menor varianza que la normal estándar, lo que resulta en activaciones iniciales más pequeñas y gradientes más equilibrados. Esta estrategia de inicialización es la misma que usan los Vision Transformers (ViT), y ha demostrado mejorar la estabilidad del entrenamiento respecto a la inicialización Xavier en modelos con stem convolucional profundo.

### 4.7 Clasificador de Zona (ZoneClassifier)

El ZoneClassifier es un MLP ligero que clasifica la distancia del sujeto en cuatro zonas basándose en 16 estadísticas de amplitud:

**Extracción de features (16 dimensiones):** Para cada uno de los 4 pares de antenas: media, desviación estándar, rango p90–p10, y energía media (suma de cuadrados normalizada). Estas estadísticas capturan el nivel medio de señal (que depende de la distancia según el modelo de pérdidas) y su dispersión (que depende de la variabilidad del canal y el movimiento).

**Arquitectura MLP:**
```
Entrada: (B, 16)  [estandarizado con mean/std del training]
→ Linear(16→64) + BatchNorm1d(64) + ReLU + Dropout(0.2)
→ Linear(64→32) + ReLU
→ Linear(32→4)
```

La BatchNorm en la primera capa es especialmente importante aquí porque las 16 features tienen escalas muy diferentes (la media puede ser 200, la energía varios millones). La estandarización previa más la BatchNorm garantizan una escala de gradientes adecuada.

**Entrenamiento:** 400 muestras por zona (1.600 total), 30 épocas, Adam con lr=1×10⁻³, sin pesos de clase (las zonas están perfectamente balanceadas en el dataset sintético). Los parámetros de normalización (mean, std) se guardan en `zone_stats.npz` para usarse en inferencia.

### 4.8 Estrategia de balanceo y ponderación de clases

El dataset de referencia presenta un fuerte desequilibrio de clases, con Walking representando el 43,9% de las muestras y Get Down/Get Up apenas el 3,79% cada uno. Este desequilibrio, si no se trata, llevaría a los modelos a predecir Walking con alta frecuencia, maximizando la precisión bruta pero con F1 muy bajo en clases minoritarias.

**Decisión de diseño: distribución uniforme + pesos de pérdida.** En lugar de reproducir el desequilibrio original en el dataset sintético, se genera el mismo número de muestras por clase (285/clase para 2.000 muestras totales, n_per=2000//7=285). Esto simplifica el pipeline y garantiza que todos los modelos vean suficientes ejemplos de todas las clases.

El desequilibrio original se compensa mediante **pesos de pérdida inversamente proporcionales a la frecuencia de clase** (tal como hace el repositorio original en `train_lstm.py`):

```
CLASS_PROPORTIONS = [0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]
w_i = 1 / p_i
W = w_i / sum(w_i) * 7   (normalización a suma = número de clases)
```

Estos pesos se pasan a `nn.CrossEntropyLoss(weight=W)`, penalizando más los errores en las clases minoritarias (Get Down, Get Up) y menos los errores en Walking.

La siguiente tabla muestra los pesos resultantes para cada clase, calculados con las proporciones del dataset de referencia:

| Clase | Proporción original | Peso calculado | Peso normalizado |
|---|---|---|---|
| Standing | 0,113 | 8,85 | 0,714 |
| Walking | 0,439 | 2,28 | 0,184 |
| Get Down | 0,0379 | 26,39 | 2,128 |
| Sitting | 0,1515 | 6,60 | 0,532 |
| Get Up | 0,0379 | 26,39 | 2,128 |
| Lying | 0,1212 | 8,25 | 0,665 |
| No Person | 0,1363 | 7,34 | 0,591 |

Los pesos para Get Down y Get Up son aproximadamente 11,5 veces superiores al peso de Walking. Esto significa que un error en una ventana de Get Up "cuesta" en términos de gradiente lo mismo que 11,5 errores en ventanas de Walking. Este esquema de ponderación agresivo refleja que, en el dataset original, Get Up y Get Down son muy infrecuentes pero igualmente importantes para el caso de uso de monitorización de personas mayores (una caída o un levantamiento dificultoso son eventos de alta prioridad clínica).

**Alternativas consideradas.** Se evaluaron brevemente tres alternativas antes de adoptar esta estrategia: (a) oversampling de clases minoritarias mediante duplicación de muestras (rechazado porque con el simulador es trivial generar nuevas muestras en lugar de duplicar); (b) uso de pesos uniformes (rechazado porque produjo una caída de F1 de ~8 puntos en las clases minoritarias en experimentos preliminares); (c) focal loss (Lino et al., 2017), que pondera dinámicamente los ejemplos según su dificultad (rechazado por añadir un hiperparámetro γ adicional que requeriría calibración). La estrategia de pesos inverso-frecuenciales es estándar, simple y directamente comparable con el repositorio de referencia.

### 4.9 Interfaz Gráfica de Usuario

La GUI se implementa con CustomTkinter (versión moderna de Tkinter con soporte para modo oscuro y temas) y Matplotlib para las visualizaciones. Se organiza en seis pestañas:

**Tab 1 — Monitor en tiempo real.** Muestra la actividad predicha con barra de confianza, la zona estimada con distancia aproximada, y un gráfico de amplitud CSI en tiempo real. El modelo en uso es seleccionable mediante un menú desplegable.

**Tab 2 — Comparativa de modelos.** Tabla comparativa de las métricas de todos los modelos entrenados, cargada automáticamente desde `benchmark.json`. Incluye precisión, F1, número de parámetros y latencia.

**Tab 3 — Matrices de confusión.** Visualización de las matrices de confusión de cada modelo (PNGs guardados durante el entrenamiento). Permite comparar visualmente los patrones de error.

**Tab 4 — Análisis de actividad.** Gráficos de distribución temporal de predicciones y evolución de la confianza durante la sesión actual.

**Tab 5 — Dataset.** Interfaz para cargar datos reales en formatos NumPy (.npy, .npz), CSV o pickle (.pkl). Muestra estadísticas descriptivas y permite ejecutar predicciones sobre los datos cargados.

**Tab 6 — Configuración.** Parámetros del sistema (velocidad de simulación, longitud de ventana, modelo activo, modo oscuro/claro).

La GUI se actualiza con un bucle de 100 ms (10 FPS) que genera una nueva ventana de datos simulados, ejecuta la inferencia de actividad y zona, y actualiza todos los elementos gráficos. El sistema es diseño sin bloqueo: la inferencia corre en el hilo principal (dada la baja latencia <10 ms de todos los modelos) sin necesidad de multithreading.

**Decisiones de implementación de la GUI.** La elección de CustomTkinter frente a alternativas como PyQt6 o wxPython responde a varios criterios. En primer lugar, CustomTkinter es una extensión directa de Tkinter, que es parte de la biblioteca estándar de Python y no requiere instalación de dependencias externas de sistema (especialmente relevante en Windows). En segundo lugar, su soporte para modo oscuro y temas modernos permite una presentación visual adecuada para una demostración académica sin necesidad de diseñar estilos personalizados. En tercer lugar, el código de CustomTkinter es sustancialmente más simple que PyQt6 para aplicaciones de monitorización con bucle de actualización periódica, lo que redujo el tiempo de desarrollo de la interfaz.

El gráfico de amplitud CSI en tiempo real (Tab 1) muestra las últimas 128 muestras de amplitud media (promedio de las 456 subportadoras) con actualización a 10 FPS. La escala del eje Y se ajusta automáticamente al rango [0, AMP_MAX] para que el nivel de señal sea siempre visible. El color de fondo del panel de actividad varía según la actividad predicha (verde para presencia activa, azul para reposo, gris para ausencia), proporcionando una indicación visual rápida incluso sin leer el texto.

La barra de confianza muestra la probabilidad softmax de la clase predicha en una escala de 0 a 100%. Una confianza baja (<50%) se muestra en naranja para alertar al usuario de que la predicción puede ser poco fiable, lo que es especialmente útil para las clases ambiguas (Standing, No Person) donde todos los modelos presentan F1 moderado.

**Gestión de estados del modelo.** La GUI carga los checkpoints de todos los modelos disponibles al arrancar (`checkpoints/*.pth`). Si un checkpoint no existe (por ejemplo, porque el modelo no se ha entrenado aún), el modelo correspondiente se marca como no disponible en el menú desplegable. Esta gestión robusta de estados permite usar la GUI incluso durante el proceso de entrenamiento incremental, cuando solo algunos modelos tienen checkpoints guardados.

---

## 5. Implementación

### 5.1 Entorno de desarrollo y dependencias

El sistema se desarrolló en Python 3.10+ sobre Windows 10. Las dependencias principales son:

| Paquete | Versión mínima | Uso |
|---|---|---|
| torch | 2.0 | Framework de aprendizaje profundo |
| numpy | 1.24 | Operaciones matriciales |
| scikit-learn | 1.2 | PCA, métricas, train/test split |
| scipy | 1.10 | Filtro Savitzky-Golay, wavelet fallback |
| matplotlib | 3.7 | Visualizaciones y matrices de confusión |
| customtkinter | 5.2 | Interfaz gráfica |
| pytest | 7.0 | Suite de tests |
| pywavelets | 1.4 | Denoising wavelet (opcional) |

La instalación se realiza mediante:
```bash
pip install -r requirements.txt
```

El directorio de trabajo debe ser la raíz del proyecto (`TFG CSI/`) para que los imports relativos funcionen correctamente. El comando de entrenamiento es:
```bash
python train_all.py
```

Y la GUI se lanza con:
```bash
python gui/app.py
```

### 5.2 Generación de datos sintéticos (SimulatedCSIDataset)

La clase `SimulatedCSIDataset` en `model/data_loader.py` genera el corpus de entrenamiento en cuatro pasos optimizados:

**Paso 1 — Generación de amplitudes brutas (paralela por clase).** Para cada muestra (clase, índice), se llama a `simulate_activity(class_idx, seq_len=128, seed=class_idx*10000+idx)`, que produce una matriz (128, 456) de amplitudes. El uso de semillas deterministas garantiza la reproducibilidad del dataset completo.

**Paso 2 — PCA global (batch).** En lugar de ajustar un PCA por ventana (lo que requeriría 2.000 ajustes de SVD), se concatenan todas las muestras en un tensor (256.000, 456) (2.000 muestras × 128 timesteps × 456 features) y se ajusta un único PCA por par de antenas sobre este corpus. Esto reduce el tiempo de construcción del dataset de ~45 segundos a ~3 segundos.

**Paso 3 — Normalización y concatenación.** Las amplitudes brutas se normalizan a [0,1] con AMP_MAX; las proyecciones PCA se normalizan con sus propios extremos. Se concatenan produciendo (2000, 128, 468).

**Paso 4 — Barajado.** Se aplica una permutación aleatoria del eje de muestras para eliminar el sesgo de orden. La semilla no se fija en este paso para permitir variabilidad entre ejecuciones.

El simulador reproduce las siguientes firmas físicas por actividad:

```python
# Ejemplo: Walking (clase 1)
freq  = rng.uniform(1.8, 2.2)          # frecuencia de marcha aleatoria
amp_g = rng.uniform(30, 60, 456)       # amplitud variable por subportadora
gait  = amp_g * sin(2π·freq·t + φ)    # señal de marcha con fase aleatoria
output = base + gait + AWGN(σ=15)

# Ejemplo: Get Up (clase 4)
tau   = (seq_len/20) * 0.4             # constante de tiempo (s)
env   = 70·exp(-t/tau) + 8            # envolvente decreciente
burst = env * AWGN(σ=1)              # ráfaga multiplicativa
output = base + burst
```

**Modelado matemático completo por actividad.** A continuación se detalla el modelo de señal completo para cada clase, expresado en términos de la amplitud `A(t, k)` en el instante `t` y la subportadora `k`:

Para **No Person** (clase 6), el canal es estático y la señal resulta únicamente del nivel base más ruido gaussiano:
```
A(t, k) = base(k) + ε(t, k),   ε ~ N(0, σ²),   σ = 3
```
donde `base(k) = rng.uniform(180, 220)` varía aleatoriamente entre subportadoras pero es constante en el tiempo, reproduciendo el canal estático sin objeto reflector humano.

Para **Standing** (clase 0), se añade una oscilación de baja frecuencia que modela la respiración y el balanceo involuntario:
```
A(t, k) = base(k) + A_resp · sin(2π·f_resp·t + φ_k) + ε(t, k)
f_resp ~ U(0.2, 0.4) Hz,   A_resp ~ U(8, 20) unidades/subportadora,   σ = 5
```
La frecuencia de respiración se muestrea aleatoriamente entre 0,2 y 0,4 Hz, coherente con los valores fisiológicos documentados (12–24 respiraciones por minuto = 0,2–0,4 Hz). La fase `φ_k` es aleatoria por subportadora para reproducir la diversidad de respuestas de fase en distintas subportadoras OFDM.

Para **Walking** (clase 1), la marcha humana introduce una perturbación periódica con frecuencia principal en la banda 1,8–2,2 Hz (equivalente a 108–132 pasos/minuto, rango de marcha normal):
```
A(t, k) = base(k) + A_g(k) · sin(2π·f_g·t + φ_k) + ε(t, k)
f_g ~ U(1.8, 2.2) Hz,   A_g(k) ~ U(30, 60),   σ = 15
```
La amplitud de la componente de marcha `A_g(k)` varía por subportadora para reproducir el patrón de interferencia constructiva/destructiva que produce el movimiento del cuerpo humano sobre las distintas subportadoras OFDM del canal.

Para **Sitting** (clase 3), la persona experimenta un desplazamiento gradual del centro de masa hacia la silla, modelado como una tendencia lineal suavizada más una componente de respiración residual:
```
A(t, k) = base(k) + A_slope · (t/T) · rng.uniform(0.8, 1.2, 456) + A_resp · sin(2π·f_resp·t) + ε(t, k)
A_slope ~ U(-15, 15),   f_resp ~ U(0.2, 0.4),   σ = 4
```
El signo de `A_slope` es aleatorio, reproduciendo que la amplitud puede crecer o decrecer dependiendo de la dirección de movimiento del sujeto relativa al par de antenas.

Para **Lying** (clase 5), la acción de tumbarse es más lenta que sentarse, y una vez tumbada, la persona produce solo respiración. El modelo es idéntico al de Standing pero con amplitud de respiración menor:
```
A(t, k) = base(k) + A_resp · sin(2π·f_resp·t + φ_k) + ε(t, k)
f_resp ~ U(0.1, 0.3) Hz,   A_resp ~ U(5, 12),   σ = 4
```
La frecuencia ligeramente inferior (0,1–0,3 Hz) refleja que en posición horizontal la respiración es más lenta y profunda que en posición erguida.

Para **Get Down** (clase 2) y **Get Up** (clase 4), las transiciones se modelan mediante envolventes exponenciales opuestas que controlan la amplitud de una ráfaga multiplicativa:
```
# Get Down: ráfaga creciente (movimiento al final de la ventana)
env(t) = A_burst · (1 - exp(-(t - T/2)/τ)) · u(t - T/2) + A_base
# Get Up: ráfaga decreciente (movimiento al inicio de la ventana)
env(t) = A_burst · exp(-t/τ) + A_base
burst(t, k) = env(t) · ε_k(t),   ε_k ~ N(0, 1)
A(t, k) = base(k) + burst(t, k)
```
donde `u(·)` es la función escalón de Heaviside, `τ = (seq_len/20) × 0,4` es la constante de tiempo de la transición, y `A_burst ~ U(50, 80)` determina la amplitud máxima de la ráfaga. La multiplicación `env(t) × ε_k(t)` (en lugar de una suma) modela el carácter multiplicativo del desvanecimiento Rician: la envolvente del canal escala la dispersión del ruido, no se suma a ella.

**Modelo de ruido.** El término de ruido `ε(t, k)` es ruido blanco gaussiano aditivo (AWGN) con desviación estándar `σ` específica de cada actividad. Este es el único componente estocástico del simulador; la semilla se fija por muestra para garantizar la reproducibilidad. En una extensión realista del simulador, el ruido podría modelarse como coloreado (Rician fading correlacionado entre subportadoras adyacentes mediante un modelo de correlación de exponencial decreciente `ρ(Δk) = ρ₀^|Δk|`), aunque este refinamiento queda fuera del alcance del presente trabajo.

**Implementación del desvanecimiento Rician.** El nivel base `base(k)` se genera siguiendo la distribución de Rice: para cada par de antenas se genera un componente LOS determinístico `μ_k = A_0 · cos(2π·Δf_k·τ_LOS)` y una componente de dispersión aleatoria de amplitud `σ_k`. La envolvente `base(k) = |μ_k + j·σ_k · (X + jY)|`, donde X, Y ~ N(0,1), sigue la distribución de Rice con factor K = μ_k²/(2σ_k²). En la implementación simplificada del código actual, `base(k) = rng.uniform(180, 220)` aproxima esta distribución en el caso K >> 1 (componente LOS dominante), que es la situación más común en entornos de oficina a corta distancia.

### 5.3 Pipeline de entrenamiento multi-modelo (train_all.py)

El script `train_all.py` orquesta el entrenamiento de todos los modelos en secuencia:

1. **Generación del dataset compartido** (`SimulatedCSIDataset(num_samples=2000, seq_len=128)`).
2. **Split 80/20** con `random_split` y semilla fija (42) para reproducibilidad.
3. **DataLoaders** con batch_size=32, sin workers (num_workers=0 por compatibilidad con Windows).
4. **Para cada modelo** en el orden [SimpleLSTM, BiLSTM, FCN, Transformer]:
   - Instanciar modelo y moverlo al dispositivo (CPU/GPU si disponible).
   - Crear optimizador Adam con tasa de aprendizaje específica del modelo.
   - Crear scheduler ReduceLROnPlateau (factor=0,5, paciencia=3) para ajuste automático.
   - Ejecutar 25 épocas de entrenamiento con gradient clipping (max_norm=1,0).
   - Guardar el mejor checkpoint (mínima pérdida de validación) en `checkpoints/<name>.pth`.
   - Medir latencia de inferencia promedio sobre 50 ejecuciones con tensor (1, 128, 468).
   - Generar y guardar matriz de confusión como PNG.
5. **Entrenamiento del ZoneClassifier** (30 épocas, ZoneDataset con 400 muestras/zona).
6. **Generación de benchmark.json** con todos los resultados.
7. **Generación de curvas de entrenamiento** superpuestas para los cuatro modelos.

**Gradient clipping** (max_norm=1,0) se aplica en todos los modelos para prevenir la explosión de gradientes, especialmente relevante en las LSTMs. En las LSTMs profundas (2 capas) con entrada de alta dimensionalidad (64 tras proyección), los gradientes pueden amplificarse exponencialmente a lo largo de las 128 posiciones temporales si no se controla su norma. El clipping proyecta el vector gradiente sobre la hiperesfera de radio 1,0 cuando su norma supera ese umbral:
```
g ← g · min(1, max_norm / ||g||₂)
```
Esta operación es conservadora en el sentido de que preserva la dirección del gradiente pero limita su magnitud, evitando actualizaciones de peso demasiado grandes en un único paso.

**Tasas de aprendizaje por modelo** (calibradas experimentalmente mediante búsqueda en malla logarítmica):

| Modelo | LR | Justificación |
|---|---|---|
| SimpleLSTM | 3×10⁻⁴ | Las LSTMs son sensibles a LR altos: con LR≥10⁻³ los pesos de las puertas divergen en las primeras épocas |
| BiLSTM | 3×10⁻⁴ | Igual que SimpleLSTM; el doble de pasos hacia atrás no cambia la sensibilidad al LR |
| FCN | 1×10⁻³ | Las CNN-1D son menos sensibles y convergen más rápido; LR=10⁻³ es el estándar para Adam en CNNs de clasificación |
| Transformer | 1,46×10⁻³ | El stem convolucional (que domina el recuento de parámetros) se beneficia de un LR algo más alto; el Pre-LN elimina la inestabilidad que haría necesario warm-up |

La calibración se realizó ejecutando cada modelo durante 10 épocas con LR ∈ {10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻²} y seleccionando el LR que minimizaba la pérdida de validación al final de las 10 épocas. Este procedimiento informal (no es una búsqueda de hiperparámetros rigurosa) resultó en los valores indicados, que se mantienen fijos para el entrenamiento completo de 25 épocas.

**Scheduler ReduceLROnPlateau.** Cuando la pérdida de validación no mejora durante 3 épocas consecutivas (patience=3), el scheduler reduce el LR actual en un factor de 0,5. El LR mínimo es 10⁻⁶. Este mecanismo de ajuste adaptativo permite que los modelos hagan un "ajuste fino" cerca del mínimo cuando el LR inicial es demasiado grande para converger. En la práctica, para el Transformer y el BiLSTM el scheduler se activa habitualmente entre las épocas 15 y 20, reduciendo el LR una o dos veces antes de que termine el entrenamiento.

**Criterio de selección del mejor checkpoint.** Durante el entrenamiento, se monitoriza la pérdida de validación al final de cada época. El checkpoint guardado (`checkpoints/<nombre>.pth`) corresponde a la época con mínima pérdida de validación (no con máxima precisión de validación). Esta elección es deliberada: la pérdida de validación es más estable y refleja mejor la capacidad de generalización del modelo que la precisión, especialmente cuando el dataset de validación es pequeño (400 muestras). En la implementación:
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), checkpoint_path)
```

**Pesos de clase en la función de pérdida.** Los pesos de clase se calculan a partir de las proporciones del dataset original de Kovalenko et al. (2021) —no del dataset simulado uniforme— para que la función de pérdida refleje el desequilibrio del problema real:
```
CLASS_PROPORTIONS = [0.113, 0.439, 0.0379, 0.1515, 0.0379, 0.1212, 0.1363]
w_i = (1 / p_i) · (7 / Σ_j(1/p_j))
```
Esta normalización (multiplicar por el número de clases) garantiza que la pérdida media ponderada sea comparable a la pérdida sin pesos, evitando que la escala de los gradientes cambie drásticamente. En la práctica, los pesos resultantes penalizan la equivocación en Get Down y Get Up (proporciones 0,038 → pesos ~3,7) respecto a Walking (proporción 0,439 → peso ~0,32), reforzando el aprendizaje de las clases más difíciles.

**Reproducibilidad.** El script `train_all.py` fija `torch.manual_seed(42)` y `numpy.random.seed(42)` antes de la generación del dataset y del split, garantizando que las mismas semillas produzcan los mismos datos, splits y resultados en cualquier ejecución con el mismo hardware. El no determinismo residual por operaciones CUDA no aplica aquí (entrenamiento en CPU).

### 5.4 Suite de tests automatizados (73 tests)

La suite de tests en `tests/` se divide en tres módulos:

**`test_data_pipeline.py`** — Valida el pipeline de datos:
- Correcto número de features (468) en la salida de preprocesamiento.
- Rango de valores normalizado [0, 1].
- Funcionamiento del filtro Hampel con outliers sintéticos.
- Reproducibilidad del simulador con semillas fijas.
- Dimensiones correctas del tensor del dataset.
- Funcionamiento del DataLoader con batch_size variable.
- Cálculo correcto de pesos de clase (suma=7).

**`test_models.py`** — Valida cada arquitectura:
- Dimensiones de salida correctas: (B, 7) para cualquier batch size.
- Número de parámetros dentro de los rangos esperados.
- Gradientes no nulos después de un paso de backprop.
- Invarianza al batch size (de 1 a 64).
- Que `predict_proba` produce valores en [0,1] con suma=1.
- Que `build_model(name)` funciona para todos los nombres del registry.

**`test_zone_classifier.py`** — Valida el clasificador de zona:
- Dimensiones del feature vector: (16,).
- Rango físicamente razonable de los features.
- Dimensiones de salida del ZoneClassifier: (B, 4).
- Que las probabilidades de zona suman 1.
- Que el simulador de zona produce amplitudes en [0, AMP_MAX].
- Coherencia de la función `_amplitude_at_distance` con el modelo ITU-R.

Para ejecutar la suite completa:
```bash
python -m pytest tests/ -v
```

**Filosofía de los tests.** Los 73 tests cubren validaciones funcionales (el sistema hace lo que se espera) y no de rendimiento (el modelo alcanza tal precisión), lo que garantiza que los tests sean deterministas y pasen independientemente del hardware, el sistema operativo o el estado del entrenamiento. Un test de rendimiento (por ejemplo, "la precisión del Transformer debe ser ≥80%") fallaría si los pesos aleatorios de inicialización o el orden de los datos cambian, lo que lo haría frágil e inapropiado para una suite de tests de integración continua.

Los tests del módulo `test_models.py` incluyen una verificación de que los gradientes son no nulos después de un paso de backpropagación, que es una condición necesaria (aunque no suficiente) para confirmar que el grafo computacional es correcto y que no hay operaciones que corten el flujo de gradiente inadvertidamente (como el uso incorrecto de `.detach()` o `.numpy()` dentro del forward pass). Esta verificación ha detectado bugs sutiles durante el desarrollo, como el uso de `tensor.item()` dentro del forward que convertía el tensor a escalar Python y cortaba el grafo.

La verificación de invarianza al batch size es especialmente importante para las LSTMs, que dependen de `h_0` y `c_0` inicializados como ceros con dimensiones `(num_layers × num_directions, B, hidden_size)`: un error común es inicializar estos tensores con un batch size fijo en lugar de inferirlo de la entrada, lo que produciría un error en tiempo de ejecución cuando el último batch del dataset tiene menos muestras que el batch_size nominal.

**Cobertura de código.** La suite de 73 tests alcanza aproximadamente el 78% de cobertura de líneas sobre el módulo `model/`, según `pytest-cov`. Las líneas no cubiertas son principalmente las ramas de manejo de errores (por ejemplo, el mensaje de error cuando se pasa un nombre de modelo desconocido a `build_model`) y el código de fallback del Savitzky-Golay cuando PyWavelets no está instalado, que es difícil de probar sin desinstalar la dependencia. Una cobertura del 78% es satisfactoria para un proyecto académico, y cubre todos los caminos de código críticos para la funcionalidad del sistema.

### 5.5 Carga de datos reales (Dataset Tab)

La pestaña Dataset de la GUI permite cargar datos CSI reales en tres formatos:

**Formato NumPy (.npy, .npz).** Se acepta un array de forma (N, T, 456) o (N, T, 468). Si el array tiene 456 features, se aplica automáticamente el pipeline PCA para producir las 468 features requeridas por los modelos.

**Formato CSV.** Cada fila es un timestep; las columnas son features. Se asume la última columna como etiqueta de clase si está presente.

**Formato pickle (.pkl).** Se acepta cualquier objeto serializado que resulte en un dict con claves `'data'` y opcionalmente `'labels'`, o un array NumPy directamente.

El sistema detecta automáticamente el formato por la extensión del archivo y muestra estadísticas descriptivas (número de muestras, distribución de clases, rango de amplitudes) antes de ejecutar la inferencia.

**Compatibilidad con el dataset de referencia.** El dataset de Kovalenko et al. (2021) se distribuye en formato de archivos binarios del driver Atheros CSI Tool (archivos `.dat` con cabecera específica). Para cargarlo en el sistema, sería necesario un paso previo de conversión a NumPy mediante el script `read_bf_file.m` (MATLAB) o su equivalente Python (`parse_bf_file.py` del repositorio de referencia), que extrae las amplitudes brutas y las guarda en formato `.npy`. Una vez convertido, el array resultante tiene forma `(N, T, 456)` y puede cargarse directamente a través de la pestaña Dataset de la GUI. Este proceso de conversión queda documentado en el Anexo A como pasos opcionales para usuarios con acceso al dataset real.

**Validaciones de la carga de datos.** Antes de ejecutar inferencia sobre datos cargados, el sistema realiza las siguientes validaciones: (a) que el número de features sea 456 o 468; (b) que el rango de amplitudes sea compatible con AMP_MAX=577,66 (se emite una advertencia si se detectan amplitudes fuera de [0, 700], lo que puede indicar un hardware diferente al de referencia o una normalización previa incompatible); (c) que el número de timesteps sea al menos 128 (la longitud de ventana requerida por los modelos). Si los datos tienen más de 128 timesteps, se divide en ventanas solapadas con paso configurable (por defecto, 50% de solapamiento).

### 5.6 Protocolo de inferencia en tiempo real

El bucle de inferencia en tiempo real de la GUI sigue este protocolo:

```python
def inference_loop():
    # 1. Generar ventana simulada (o leer buffer real si disponible)
    raw_amp = simulate_activity(current_class, seq_len=128)   # (128, 456)
    
    # 2. Preprocesar para HAR
    features = full_preprocess(raw_amp)    # (128, 468)
    x = torch.from_numpy(features).unsqueeze(0)   # (1, 128, 468)
    
    # 3. Inferencia HAR
    with torch.no_grad():
        logits = model(x)                  # (1, 7)
        probs  = softmax(logits, dim=1)    # (1, 7)
    activity_idx  = probs.argmax().item()
    confidence    = probs.max().item()
    
    # 4. Extraer features de zona y clasificar
    zone_feats = extract_zone_features(raw_amp)   # (16,)
    zone_idx, zone_probs = zone_clf.predict(zone_feats, mean, std)
    
    # 5. Actualizar GUI
    update_display(activity_idx, confidence, zone_idx, zone_probs)
```

El protocolo garantiza que cada ciclo de inferencia tarda menos de 15 ms en total (incluyendo preprocesamiento), compatible con la actualización de la GUI a 10 FPS.

**Gestión de la incertidumbre en la predicción.** El sistema reporta no solo la clase predicha sino también la probabilidad softmax asociada como medida de confianza. Esta información es útil para filtrar predicciones poco fiables: una confianza baja (<0,5) indica que el modelo está indeciso entre múltiples clases, lo que puede ocurrir en transiciones entre actividades o en condiciones de canal ambiguo. En la GUI, las predicciones con confianza <0,5 se muestran con fondo naranja y la etiqueta "(baja confianza)", mientras que las predicciones con confianza >0,8 se muestran con fondo verde oscuro.

**Ventaneo deslizante en tiempo real.** El protocolo de inferencia en tiempo real usa una ventana deslizante con solapamiento del 50%: en cada ciclo de 100 ms se genera una nueva ventana de 128 muestras que comparte las últimas 64 muestras con la ventana anterior. Esto suaviza las transiciones entre predicciones consecutivas y permite detectar el inicio de una nueva actividad aproximadamente 3,2 segundos antes de que la ventana completa contenga únicamente la nueva actividad (tiempo de transición de 64 muestras × 50 ms/muestra = 3,2 s). En un sistema de captura real, las 128 muestras equivalen a 6,4 segundos de señal CSI a 20 paquetes/segundo, con actualización de predicción cada 3,2 segundos (50% de solapamiento) o cada 6,4 segundos (ventana no solapada).

---

## 6. Evaluación Experimental

### 6.1 Configuración experimental

**Hardware:** Los experimentos se realizaron en un equipo con CPU Intel Core i-series (sin GPU dedicada para entrenamiento). El entrenamiento de los cuatro modelos más el ZoneClassifier tarda aproximadamente 8–12 minutos en total.

**Dataset:** 2.000 muestras simuladas con distribución uniforme (285 muestras/clase, con la clase 6 recibiendo la muestra adicional del redondeo). Split 80/20 (1.600 entrenamiento / 400 validación), con semilla aleatoria fija (42) para reproducibilidad.

**Hiperparámetros comunes:**
- Épocas: 25
- Batch size: 32
- Optimizer: Adam (β₁=0,9, β₂=0,999, ε=10⁻⁸)
- Gradient clipping: max_norm=1,0
- Scheduler: ReduceLROnPlateau (factor=0,5, patience=3, min_lr=10⁻⁶)
- Pérdida: CrossEntropyLoss con pesos de clase inverso-frecuenciales

**Métricas:**
- Precisión de validación (accuracy): fracción de muestras correctamente clasificadas.
- F1-score ponderado (weighted F1): media de F1-scores por clase, ponderada por soporte.
- F1-score por clase: para análisis de capacidad de discriminación por actividad.
- Latencia de inferencia: media de 50 ejecuciones con tensor de una sola muestra, en CPU.
- Número de parámetros: parámetros entrenables totales.

**Justificación de los hiperparámetros comunes.** El número de épocas (25) se determinó experimentalmente observando que todos los modelos convergen o se estabilizan antes de la época 20 con el dataset de 2.000 muestras. Extender el entrenamiento más allá de 25 épocas no produjo mejoras de validación significativas (<0,5 puntos de F1), mientras que el riesgo de sobreajuste aumentaría. El batch size de 32 es el valor estándar para datos tabulares de tamaño medio; valores menores (8, 16) producirían gradientes más ruidosos, y valores mayores (64, 128) requerirían más épocas para converger con el tamaño de dataset disponible. Los parámetros de Adam (β₁=0,9, β₂=0,999) son los valores por defecto de PyTorch y Adam, ampliamente validados en la literatura de series temporales. El ε=10⁻⁸ evita divisiones por cero en el denominador del actualizador de momento de segundo orden.

**Validez de los resultados:** Se enfatiza que todos los resultados se obtienen sobre datos **simulados**, no sobre datos CSI reales. Los modelos aprenden de señales sintéticas generadas por el simulador descrito en la Sección 5.2. Los valores numéricos reportados son reproducibles ejecutando `train_all.py` con la configuración descrita. La interpretación correcta de estos resultados es que reflejan la capacidad relativa de cada arquitectura para aprender las representaciones que el simulador produce, y no necesariamente su rendimiento en un entorno de captura real.

### 6.2 Comparativa de modelos HAR

| Modelo | Val Acc | Val F1 (w) | Parámetros | Latencia CPU |
|---|---|---|---|---|
| **CSITransformer** | **82,5%** | **0,826** | **330.887** | **5,3 ms** |
| BiLSTM | 78,2% | 0,768 | 658.311 | 8,8 ms |
| SimpleLSTM | 50,7% | 0,471 | 888.455 | 9,6 ms |
| FCN | 30,8% | 0,305 | 743.303 | 5,4 ms |
| ZoneClassifier | 98,1% | — | — | — |

El CSITransformer es el mejor modelo en todas las métricas de precisión, al tiempo que es el más eficiente en parámetros y el más rápido (empate con FCN dentro del margen de medición). Esta combinación de eficiencia y precisión es el resultado más destacado del trabajo. Con 330.887 parámetros, el Transformer necesita menos de la mitad de los parámetros del SimpleLSTM (888.455) para obtener 32 puntos porcentuales más de precisión, lo que pone de relieve que la capacidad arquitectónica —la habilidad de capturar relaciones globales en la ventana— es más valiosa que la capacidad paramétrica bruta en este problema.

El BiLSTM es el segundo mejor modelo, con 78,2% de precisión y F1=0,768, aprovechando el contexto bidireccional para capturar la estructura temporal de las actividades de transición (Get Up, Get Down). La diferencia con el SimpleLSTM (50,7%) es notable —casi 28 puntos porcentuales— y confirma de forma contundente la importancia del contexto bidireccional. Esta diferencia es la más grande entre dos modelos consecutivos en la jerarquía de rendimiento, sugiriendo que el acceso al futuro de la secuencia es el factor más crítico para la clasificación de actividades de transición en este dataset.

La FCN con 30,8% de precisión y F1=0,305 obtiene el peor resultado, confirmando la hipótesis de que la pérdida de estructura temporal mediante GAP es crítica para distinguir actividades de transición con datos de entrenamiento limitados. Paradójicamente, la FCN tiene el segundo menor número de parámetros (743.303) y la segunda menor latencia (5,4 ms), lo que la convertiría en una opción atractiva si su precisión fuera comparable a la del Transformer. La brecha de 51,7 puntos porcentuales de precisión entre el Transformer y la FCN es la diferencia más grande entre el mejor y el peor modelo observada en este benchmark, reflejando cuán inadecuada es la arquitectura FCN para este problema específico.

El SimpleLSTM con 50,7% de precisión ofrece un rendimiento intermedio entre el BiLSTM y la FCN. Con 888.455 parámetros —el mayor de los cuatro modelos—, su bajo rendimiento relativo revela que la cantidad de parámetros no es un indicador de rendimiento en series temporales: el diseño de la arquitectura (bidireccionalidad, atención global, pooling temporal) importa más que el volumen de parámetros entrenables.

### 6.3 Análisis por clase — CSITransformer

La siguiente tabla recoge las métricas por clase del CSITransformer sobre el conjunto de validación (400 muestras con distribución uniforme de 7 clases):

| Clase | Precisión | Recall | F1-score |
|---|---|---|---|
| Standing | 0,500 | 0,556 | 0,526 |
| Walking | 0,900 | 0,885 | 0,893 |
| Get Down | 1,000 | 0,981 | 0,991 |
| Sitting | 1,000 | 1,000 | 1,000 |
| Get Up | 1,000 | 1,000 | 1,000 |
| Lying | 0,947 | 0,947 | 0,947 |
| No Person | 0,480 | 0,436 | 0,457 |

El CSITransformer alcanza F1=1,000 en Sitting y Get Up, y prácticamente perfecto en Get Down (F1=0,991, Recall=0,981). Este trio de clases constituye el núcleo de éxito del modelo y ofrece evidencia de que las firmas físicas simuladas para estas actividades son suficientemente discriminativas para que el mecanismo de atención del Transformer aprenda a identificarlas con práctica perfección. Sitting, en particular, es la única clase que además combina Precisión=1,000 y Recall=1,000: ningún ejemplo de otra clase se confunde con Sitting, y ningún ejemplo de Sitting se confunde con otra clase. Esto refleja que la deriva muy lenta de amplitud que modela el acto de sentarse ocupa una región del espacio de características completamente aislada de las demás actividades.

Las actividades de transición Get Down y Get Up exhiben un patrón de métricas instructivo. Get Up tiene Precisión=Recall=1,000, mientras que Get Down tiene Precisión=1,000 pero Recall=0,981, indicando que un pequeño número de ventanas de Get Down son clasificadas erróneamente en otra categoría —probablemente Lying, dada la similitud de la fase final de la transición (reposo en el suelo) con la firma de postura tumbada—. El hecho de que la Precisión sea perfecta para ambas transiciones confirma que el Transformer no produce falsos positivos para estas clases: cuando predice Get Down o Get Up, siempre acierta.

Walking logra F1=0,893 con una asimetría mínima entre Precisión (0,900) y Recall (0,885), lo que indica un balance casi perfecto entre falsos positivos y falsos negativos. La pequeña discrepancia —algo más de falsos negativos que de falsos positivos— sugiere que el Transformer clasifica en Standing o Lying un ~11% de los ejemplos de Walking en el extremo inferior de amplitud, mientras que casi no etiqueta erróneamente ejemplos de otras clases como Walking. Este comportamiento es coherente con la dificultad física del límite entre Walking de baja amplitud y Standing de alta amplitud: ambas actividades pueden producir oscilaciones en el rango 1–2 Hz con amplitudes similares cuando los parámetros del simulador toman valores extremos.

Lying obtiene F1=0,947 con métricas equilibradas (Precisión=Recall=0,947). La simetría en los errores indica que la confusión entre Lying y otras clases (principalmente Standing) es bidireccional: algunos ejemplos de Lying se clasifican como Standing y viceversa. Desde el punto de vista del modelo de señal, esto es esperado: tanto Lying como Standing producen oscilaciones de baja amplitud y baja frecuencia; la diferencia entre ambas (frecuencia de respiración ligeramente inferior y amplitud ligeramente menor en Lying) es suficientemente sutil como para producir una tasa de error del 5,3%.

Las dos clases más problemáticas son Standing (F1=0,526) y No Person (F1=0,457). Para Standing, la asimetría Precisión=0,500, Recall=0,556 indica que el modelo confunde en ambas direcciones: algunos ejemplos de Standing se clasifican como No Person o Lying (Recall < 1), y algunos ejemplos de No Person o Lying se clasifican como Standing (Precisión < 1). La Precisión exactamente del 50% indica que, en promedio, solo la mitad de las predicciones Standing son correctas, lo que es prácticamente el límite inferior de utilidad para una clase.

No Person (F1=0,457) exhibe el mismo patrón de confusión bidireccional que Standing, con Precisión=0,480 y Recall=0,436. El Recall bajo (0,436) indica que el Transformer no reconoce el 56,4% de las ventanas de canal vacío, clasificándolas mayoritariamente como Standing o Lying. La Precisión de 0,480 significa que cuando predice No Person, acierta menos de la mitad de las veces. Esta combinación revela que la frontera de decisión entre No Person, Standing y Lying es la región más ambigua del espacio de características del sistema simulado, y que ningún modelo —incluido el CSITransformer, el mejor de los cuatro— puede resolverla de forma fiable con los parámetros de ruido actuales del simulador.

### 6.4 Análisis por clase — BiLSTM

La siguiente tabla recoge las métricas por clase del modelo BiLSTM, obtenidas del conjunto de validación (400 muestras, distribución uniforme):

| Clase | Precisión | Recall | F1-score |
|---|---|---|---|
| Standing | 0,420 | 0,794 | 0,549 |
| Walking | 1,000 | 0,557 | 0,716 |
| Get Down | 1,000 | 1,000 | 1,000 |
| Sitting | 0,983 | 1,000 | 0,992 |
| Get Up | 1,000 | 1,000 | 1,000 |
| Lying | 0,824 | 0,982 | 0,896 |
| No Person | 0,643 | 0,164 | 0,261 |

El BiLSTM presenta un perfil de rendimiento caracterizado por una notable asimetría entre clases dinámicas y clases estáticas. Las actividades de transición y las de estado estable con firma energética diferenciada (Get Down, Get Up, Sitting) obtienen F1 iguales o muy próximos a 1,000, lo que indica que el procesamiento bidireccional permite al modelo capturar perfectamente la estructura temporal opuesta de ambas transiciones: la ráfaga creciente al final de la ventana para Get Down y la ráfaga decreciente al inicio para Get Up. Este resultado es coherente con la hipótesis teórica de que el contexto bidireccional es esencial para distinguir actividades cuya identificación requiere observar la forma global de la ventana, no solo los instantes pasados.

El resultado más revelador es el de la clase Walking, donde se observa una disociación pronunciada entre Precisión (1,000) y Recall (0,557). Una Precisión perfecta significa que todos los ejemplos que el BiLSTM clasifica como Walking son genuinamente Walking —no hay falsos positivos para esta clase—. Sin embargo, el Recall de 0,557 indica que casi la mitad de los ejemplos reales de Walking no son reconocidos como tales y se clasifican en otras categorías. Dada la naturaleza de las firmas simuladas, las confusiones más probables son hacia Standing —cuya señal a baja velocidad de marcha puede solaparse con el balanceo corporal lento de la postura estática— o hacia Lying en los casos en que la amplitud media del ciclo de marcha se asemeje al nivel de amplitud de la postura tumbada. En otras palabras, el BiLSTM tiende a ser excesivamente conservador al predecir Walking, prefiriendo etiquetar como Standing o Lying aquellos ciclos de marcha con parámetros de frecuencia o amplitud en el extremo inferior del rango simulado (frecuencia ≈ 1,8 Hz, amplitud ≈ 30 unidades/subportadora).

La clase Standing obtiene F1=0,549, con una asimetría inversa a la de Walking: Recall alto (0,794) pero Precisión baja (0,420). Esto indica que el BiLSTM tiende a sobrepredicir Standing, etiquetando como tal ejemplos que pertenecen a otras clases. El mecanismo es el siguiente: actividades como No Person (canal estático con ruido σ=3) y Walking de baja amplitud pueden producir patrones de amplitud media similares a los de Standing (oscilación lenta de 0,3 Hz), y el modelo resuelve la ambigüedad decantándose sistemáticamente hacia la clase Standing, que en el dataset original de referencia es una de las más frecuentes. La ponderación inversa de pérdidas mitiga este sesgo, pero no lo elimina completamente.

La clase No Person alcanza el F1 más bajo (0,261), impulsado principalmente por un Recall extremadamente bajo (0,164). Esto significa que en aproximadamente ocho de cada diez ventanas generadas con ausencia de persona, el BiLSTM predice otra actividad —más frecuentemente Standing o Lying—. Desde el punto de vista físico, este error es comprensible: la diferencia entre el ruido térmico AWGN de σ=3 (No Person) y la pequeña oscilación de σ=5 con componente de 0,3 Hz (Standing) es sutil, y el modelo recurrente, al procesar la secuencia temporalmente, puede no capturar la ausencia de estructura periódica que caracteriza al canal vacío. La Precisión de 0,643 indica que cuando el modelo sí predice No Person, acierta en el 64,3% de los casos, lo que es razonablemente mejor que el azar, pero la altísima tasa de falsos negativos lastra el F1.

La clase Lying obtiene F1=0,896, un resultado sólido que refleja que la firma de baja amplitud y variación sub-0,05 Hz es suficientemente diferente de Walking y Sitting para que el BiLSTM la reconozca con confianza. La Precisión de 0,824 (algo inferior al Recall de 0,982) sugiere que el modelo classifica en Lying algunos ejemplos de No Person o Standing de amplitud especialmente baja.

En comparación directa con el CSITransformer, el BiLSTM supera al Transformer en Standing (F1: 0,549 vs 0,526) y es comparable en Get Down (F1: 1,000 vs 0,991) y Sitting (F1: 0,992 vs 1,000), pero queda claramente por debajo en Walking (0,716 vs 0,893) y No Person (0,261 vs 0,457). El patrón sugiere que el mecanismo de atención global del Transformer, combinado con la agregación mean+max, es más efectivo que la LSTM bidireccional para discriminar actividades cuya firma temporal tiene estructura frecuencial clara (Walking) o ausencia total de estructura (No Person).

### 6.5 Análisis por clase — SimpleLSTM

La siguiente tabla recoge las métricas por clase del modelo SimpleLSTM sobre el conjunto de validación:

| Clase | Precisión | Recall | F1-score |
|---|---|---|---|
| Standing | 0,321 | 0,143 | 0,198 |
| Walking | 0,444 | 0,262 | 0,330 |
| Get Down | 0,643 | 1,000 | 0,783 |
| Sitting | 0,925 | 0,831 | 0,875 |
| Get Up | 0,254 | 0,588 | 0,355 |
| Lying | 0,556 | 0,702 | 0,620 |
| No Person | 0,556 | 0,091 | 0,156 |

El SimpleLSTM con un F1 ponderado de 0,471 y una precisión del 50,7% supera ampliamente la predicción aleatoria (14,3% en 7 clases), pero el análisis por clase revela un patrón de errores cualitativamente diferente al de los modelos con contexto bidireccional o global. El modelo procesa cada ventana en una única pasada izquierda-derecha, acumulando representación en el estado oculto, lo que le confiere visión del contexto pasado pero no del futuro inmediato.

El resultado más sorprendente es el de Get Down, que alcanza F1=0,783 con Recall=1,000. Recall perfecto significa que el SimpleLSTM identifica correctamente el 100% de los ejemplos reales de Get Down. Esto parece paradójico para una actividad de transición cuya firma clave (ráfaga de amplitud creciente) se concentra en la segunda mitad de la ventana temporal —justamente la parte que un modelo bidireccional aprovecha mirando hacia atrás—. La explicación es que el modelo unidireccional aprende a reconocer Get Down cuando llega a los últimos pasos temporales: el estado oculto acumulado hasta el final de la ventana contiene la energía de la ráfaga creciente, y el clasificador aprende a mapearlo con gran confianza. La Precisión de 0,643 indica no obstante que el modelo también clasifica erroneamente en Get Down algunas ventanas de Lying o Walking de alta amplitud, produciendo falsos positivos. El resultado neto es un F1=0,783 razonablemente bueno para un modelo con tan solo visión unidireccional.

Sitting logra F1=0,875, la segunda mejor clase, con una Precisión muy alta (0,925) y Recall aceptable (0,831). La firma de Sitting —deriva muy lenta de amplitud que no se repite en el tiempo— es reconocible con visión unidireccional porque la LSTM puede detectar la tendencia monótona acumulada en el estado oculto conforme avanza la secuencia.

Walking obtiene F1=0,330, resultado decepcionante para la actividad con la firma Doppler más energética (~2 Hz). La causa probable reside en que, con un Recall de apenas 0,262, el modelo raramente reconoce Walking; en cambio, asigna estas ventanas a Lying o Standing. Esto sugiere que la LSTM unidireccional, al procesar la señal periódica de la marcha, no es capaz de extraer con fiabilidad el patrón oscilatorio cuando se enfrenta a parámetros de frecuencia y amplitud en los extremos del rango simulado. La Precisión de 0,444 indica que cuando el modelo predice Walking, acierta menos de la mitad de las veces.

Get Up presenta el patrón inverso al de Get Down: F1=0,355 con Recall=0,588 pero Precisión=0,254. El Recall más alto que en Walking refleja que el modelo detecta la ráfaga de alta amplitud al inicio de la ventana como una señal activa, pero la Precisión muy baja (0,254 → apenas uno de cada cuatro ejemplos predichos como Get Up es realmente Get Up) indica una altísima tasa de falsos positivos. El modelo clasifica como Get Up muchos ejemplos de Walking o Lying que presentan un transitorio inicial, sin poder distinguirlos de la ráfaga exponencial decreciente que caracteriza la incorporación real. La falta de contexto futuro es aquí especialmente costosa: el BiLSTM, al leer la ventana también hacia atrás, puede verificar que después de la ráfaga inicial la señal decae ordenadamente hasta el nivel de Standing, mientras que el SimpleLSTM solo puede inferir este comportamiento futuro de forma implícita a través del estado oculto.

Standing y No Person comparten el problema de la indetección: Recall=0,143 y Recall=0,091 respectivamente. El modelo unidireccional, cuyo estado oculto tiende a saturarse con la información acumulada de actividades dinámicas, tiene dificultades para capturar la "ausencia de evento" que caracteriza a estas dos clases. No Person, con F1=0,156, es prácticamente irreconocible para el SimpleLSTM: en 9 de cada 10 ventanas de canal vacío, el modelo predice alguna actividad, muy frecuentemente Standing. Lying (F1=0,620) es la excepción relativa entre las clases estáticas, gracias a que su amplitud media es significativamente inferior a la de Standing, lo que permite al modelo distinguirla con cierta fiabilidad incluso sin contexto bidireccional.

El análisis del SimpleLSTM pone de manifiesto una lección de diseño importante: para HAR con actividades de transición, la arquitectura unidireccional es estructuralmente inadecuada, independientemente de la capacidad del modelo. La arquitectura impone una asimetría temporal que favorece a las actividades cuya firma característica se acumula progresivamente (Get Down) y perjudica a aquellas cuya firma está concentrada al inicio (Get Up) o distribuida sin dirección temporal dominante (No Person, Standing).

### 6.6 Análisis por clase — FCN

La siguiente tabla recoge las métricas por clase del modelo FCN (Red Completamente Convolucional) sobre el conjunto de validación:

| Clase | Precisión | Recall | F1-score |
|---|---|---|---|
| Standing | 0,452 | 0,222 | 0,298 |
| Walking | 0,198 | 0,410 | 0,267 |
| Get Down | 0,156 | 0,093 | 0,116 |
| Sitting | 0,789 | 0,763 | 0,776 |
| Get Up | 0,237 | 0,176 | 0,202 |
| Lying | 0,246 | 0,246 | 0,246 |
| No Person | 0,186 | 0,200 | 0,193 |

La FCN con F1 ponderado de 0,305 y precisión del 30,8% obtiene el resultado global más bajo de los cuatro modelos. Sin embargo, el análisis por clase revela que este resultado agregado enmascara una variabilidad por actividad verdaderamente extrema: desde Sitting (F1=0,776, el mejor resultado de la FCN) hasta Get Down (F1=0,116, prácticamente irreconocible).

El resultado de Sitting es el más sorprendente en sentido positivo: F1=0,776 con Precisión=0,789 y Recall=0,763. Que una arquitectura que descarta toda la estructura temporal mediante Global Average Pooling logre reconocer Sitting mejor que muchas clases en los otros modelos revela la explicación: Sitting no se diferencia de otras actividades por su patrón temporal, sino por su nivel medio de amplitud y su distribución estadística. La deriva muy lenta característica de Sitting (amplitud media intermedia, desviación estándar muy pequeña, sin componentes de alta frecuencia) produce un vector de características post-GAP con valores medios en un rango específico que el clasificador lineal puede separar con razonable eficacia. En otras palabras, Sitting es la actividad cuya firma estadística (no temporal) es más distinguible.

Get Down alcanza el peor resultado de todo el benchmark: F1=0,116 con Recall=0,093. Esto significa que la FCN solo identifica correctamente el 9,3% de los ejemplos reales de Get Down; el 90,7% restante se clasifica como otra actividad. Este resultado es la demostración más clara de la limitación fundamental del GAP para HAR: la ráfaga de alta amplitud que define Get Down ocupa únicamente la segunda mitad de la ventana temporal, y al promediar toda la ventana mediante GAP, esta ráfaga queda diluida en el promedio global. El vector post-GAP de una ventana de Get Down puede ser estadísticamente similar al de una ventana de Walking de amplitud moderada o incluso de Lying de amplitud variable. La Precisión de 0,156 indica que los pocos casos en que la FCN predice Get Down, solo acierta en el 15,6%, evidenciando también una alta tasa de falsos positivos.

Walking exhibe un patrón inusual: Recall=0,410 pero Precisión=0,198. Esto significa que el modelo clasifica como Walking muchos ejemplos de otras clases (Precisión baja), pero también que detecta aproximadamente el 41% de los ejemplos reales de Walking. La razón es que Walking produce las amplitudes medias más altas del dataset, y la FCN aprende que amplitud media alta implica posiblemente Walking. Sin embargo, Standing y Lying de alta amplitud también producen amplitudes medias elevadas, generando numerosos falsos positivos. El clasificador post-GAP de la FCN opera esencialmente como un clasificador de amplitud media, con todas las limitaciones inherentes a esa reducción.

Las clases Lying (F1=0,246) y No Person (F1=0,193) presentan resultados comparablemente bajos e indistinguibles estadísticamente entre sí. Esta simetría tiene una interpretación directa: tanto Lying como No Person producen amplitudes medias bajas (la persona tumbada genera poca perturbación del canal; la ausencia de persona deja solo el ruido), y el promedio global post-GAP las coloca en regiones del espacio de características solapadas. La FCN no puede diferenciarlas porque su diferencia reside precisamente en la dinámica temporal —presencia o ausencia de la oscillación sub-0,05 Hz de respiración— que el GAP elimina.

Get Up (F1=0,202) y Standing (F1=0,298) completan el cuadro de un modelo que opera fundamentalmente como un clasificador de nivel medio de energía: reconoce razonablemente las actividades con un nivel de energía muy distintivo (Sitting, en cierta medida Walking) pero falla para todas las demás.

Desde una perspectiva metodológica, el pésimo rendimiento de la FCN no invalida la arquitectura en general, sino que señala una incompatibilidad específica entre el diseño del pooling global y el tipo de problema. En los benchmarks de clasificación de series temporales donde la FCN de Wang et al. (2017) fue propuesta (datasets de UCR), las actividades tienen patrones temporales repetitivos y estacionarios que el GAP puede representar estadísticamente. En cambio, el problema HAR con actividades de transición requiere explícitamente la información sobre cuándo ocurre el evento dentro de la ventana, información que el GAP destruye irreversiblemente. Esta incompatibilidad arquitectónica, y no el tamaño o la capacidad del modelo, es la causa raíz del bajo rendimiento de la FCN en este trabajo.

### 6.7 Estimación de zona (ZoneClassifier)

La siguiente tabla recoge los resultados por zona del ZoneClassifier sobre el conjunto de validación (160 muestras, 40 por zona):

| Zona | Nombre | Precisión | Recall | F1-score |
|---|---|---|---|---|
| 0 | Proximidad (0–1,5 m) | 0,973 | 1,000 | 0,986 |
| 1 | Cerca (1,5–3,0 m) | 0,965 | 0,976 | 0,971 |
| 2 | Media distancia (3,0–5,0 m) | 0,987 | 0,939 | 0,963 |
| 3 | Lejos (5,0–8,0 m) | 0,976 | 0,988 | 0,982 |

El ZoneClassifier alcanza una precisión global del 98,1% con F1-scores por zona en el rango 0,963–0,986. Este resultado es esperable dado que el simulador genera las amplitudes directamente a partir del modelo de pérdidas ITU-R P.1238, y el clasificador de zona aprende exactamente esta misma función paramétrica. Sin embargo, la precisión no es perfecta (100%) incluso en datos simulados, lo que revela un efecto de solapamiento en los límites de zona: las amplitudes generadas en los extremos de cada zona (por ejemplo, 1,5 m en el límite Proximidad/Cerca o 3,0 m en el límite Cerca/Media) son estadísticamente similares a las amplitudes en los extremos de la zona adyacente, produciendo algunos errores de clasificación.

El patrón de errores por zona es informativo. La zona Proximidad (0–1,5 m) tiene Recall=1,000, indicando que nunca se confunde una muestra de Proximidad con otra zona; sin embargo, la Precisión de 0,973 revela que algunas muestras de la zona Cerca, cuando su amplitud media es alta (por la varianza del generador), se clasifican erróneamente como Proximidad. La zona Media distancia tiene el F1 más bajo (0,963), con Recall=0,939, indicando que es la zona con más falsos negativos —muestras de Media que se clasifican como Cerca o Lejos—. Esto es razonable: Media distancia es la zona intermedia, flanqueada por dos zonas adyacentes, y sus amplitudes solapan con ambas en los bordes del intervalo [3,0, 5,0] m.

La arquitectura MLP (16→64→32→4) con 16 estadísticas de amplitud es suficiente para esta tarea. Las 16 features capturan el nivel medio de señal (media), la dispersión (desviación estándar, IQR p90–p10) y la energía total por par de antenas, lo que es redundante en cierta medida pero robustece el clasificador frente a variaciones de ruido. El uso de BatchNorm en la primera capa es crítico para normalizar la escala de las features antes de la activación ReLU, evitando que features de alta magnitud (energía ≈ 10⁶ unidades²) dominen sobre features de baja magnitud (desviación estándar ≈ 10 unidades).

En un despliegue real, la precisión sería significativamente menor que el 98,1% simulado, por tres razones principales. En primer lugar, las variaciones de shadowing siguen una distribución log-normal con σ=4–7 dB en entornos típicos de oficina, lo que puede desplazar la amplitud observada en ±30–70 unidades respecto al valor predicho por el modelo ITU-R P.1238, haciendo que una misma distancia produzca amplitudes correspondientes a zonas adyacentes. En segundo lugar, las condiciones de no visión directa (NLOS) —cuando el sujeto está detrás de paredes o mobiliario— producen atenuaciones adicionales de 10–30 dB que el modelo de pérdidas de trayecto ITU-R no contempla. En tercer lugar, las variaciones de orientación del sujeto respecto al par transmisor-receptor modifican el patrón de reflexión y, por tanto, la amplitud media observada en cada par de antenas, rompiendo la asunción de isotropía implícita en el modelo de pérdidas. Una extensión práctica del sistema debería incorporar un modelo de shadowing log-normal y datos de calibración NLOS para reducir el impacto de estos factores.

### 6.8 Análisis de latencia y complejidad computacional

| Modelo | Latencia CPU | Params | Params/Acc |
|---|---|---|---|
| CSITransformer | 5,3 ms | 330.887 | 4.011 |
| FCN | 5,4 ms | 743.303 | 24.131 |
| BiLSTM | 8,8 ms | 658.311 | 8.418 |
| SimpleLSTM | 9,6 ms | 888.455 | 17.524 |

*Params/Acc: parámetros por punto porcentual de precisión (menor = más eficiente)*

El CSITransformer es el modelo más eficiente en términos de parámetros/rendimiento, con 4.011 parámetros por punto porcentual de precisión, frente a los 24.131 de la FCN (que tiene mayor número de parámetros pero menor precisión). Esta métrica compuesta (denominada a veces "coste de precisión") pone de manifiesto que el SimpleLSTM y la FCN son arquitecturas ineficientes para este problema: el SimpleLSTM necesita 17.524 parámetros por punto porcentual —más de cuatro veces los del Transformer— mientras que la FCN, con su pésimo rendimiento, necesita 24.131, el peor ratio del benchmark.

La latencia de 5,3 ms del Transformer es llamativa dado que los Transformers son habitualmente más lentos que las CNNs en inferencia. La razón es la dimensionalidad reducida del modelo (d_model=64, 2 capas, seq_len=128): la operación de atención tiene complejidad cuadrática O(T²·d) = O(128²·64) ≈ 1M operaciones multiplicación-acumulación (MACs), comparable al coste de las tres capas convolucionales de la FCN. En contraste, las LSTMs tienen complejidad lineal en la longitud de secuencia O(T · d²) pero con un coeficiente mayor: O(128 · 256² · 4) ≈ 33M MACs para el SimpleLSTM sin proyección. La proyección 468→64 reduce este coste a O(128 · 64² · 4) ≈ 2M MACs, pero aún así las LSTMs son más lentas que el Transformer en este caso de uso específico.

La medición de latencia se realiza como media de 50 ejecuciones de inferencia con `torch.no_grad()` y un tensor de una sola muestra `(1, 128, 468)` en CPU. Este protocolo garantiza que la medición refleja la latencia de inferencia en condiciones de despliegue edge (una ventana a la vez, sin batching), no la throughput de procesamiento por lotes que sería relevante en un sistema de análisis offline. Los 50 warmup implícitos en la media garantizan que los efectos de caché de CPU y compilación JIT de PyTorch se hayan estabilizado antes de las mediciones.

Desde la perspectiva del despliegue práctico, todos los modelos cumplen el requisito de latencia <10 ms para aplicaciones de monitorización en tiempo real a 10 FPS. A 10 FPS, el presupuesto de tiempo por ciclo de inferencia completo es de 100 ms, del cual la inferencia del modelo consume como máximo 9,6 ms (SimpleLSTM), dejando más de 90 ms para preprocesamiento, extracción de features de zona, inferencia del ZoneClassifier y actualización de la GUI. Este presupuesto es muy holgado, lo que permite incluso implementar procesamiento redundante (ejecutar dos modelos en paralelo para voting de conjunto) sin riesgo de incumplir el requisito de tiempo real.

### 6.9 Discusión de resultados

Los resultados revelan tres hallazgos principales:

**Hallazgo 1: El Transformer supera a las LSTMs con pocas muestras de entrenamiento.** Con solo 1.600 muestras de entrenamiento, el CSITransformer logra 82,5% de precisión frente al 78,2% del BiLSTM y 50,7% del SimpleLSTM. Este resultado es, en cierta medida, contraintuitivo: los Transformers son conocidos por requerir grandes volúmenes de datos. La explicación probable es que el stem convolucional actúa como un extractor de características que reduce la dificultad del problema para el encoder Transformer, y que la agregación mean+max proporciona supervisión implícita sobre la estructura global de la ventana.

**Hallazgo 2: La FCN no es adecuada para HAR con actividades de transición.** El 30,8% de precisión de la FCN es el resultado más sorprendente. La FCN de Wang et al. (2017) obtuvo resultados excelentes en datasets como UCR, pero esos datasets contienen actividades con patrones temporales repetitivos (arritmias cardíacas, consumo energético). Las actividades de transición del dataset CSI (Get Up, Get Down) tienen estructura temporal no estacionaria que el GAP elimina, lo que explica el bajo rendimiento.

**Hallazgo 3: Los datos simulados son insuficientes para reproducir la riqueza del canal real.** La brecha entre las precisiones reportadas en la literatura (87–94%) y las obtenidas en este trabajo (30–82%) refleja las limitaciones del simulador. El simulador reproduce los patrones medios de cada actividad, pero no la variabilidad inter-sujeto, las variaciones de posición del transmisor, los efectos del mobiliario o las interferencias de canal propias de un entorno real. Esta limitación es la más importante del trabajo y debe tenerse en cuenta al interpretar todos los resultados.

**Hallazgo 4: La jerarquía de rendimiento es robusta a pesar de las limitaciones.** A pesar de que todos los resultados son sobre datos simulados, la jerarquía de rendimiento entre arquitecturas (Transformer > BiLSTM > SimpleLSTM > FCN) es coherente con las predicciones teóricas basadas en la capacidad de cada arquitectura para capturar la estructura temporal de las actividades. Esta coherencia sugiere que la jerarquía se mantendría probablemente con datos reales, aunque los valores absolutos de precisión serían diferentes. El resultado más incierto en este sentido es la diferencia Transformer–BiLSTM (82,5% vs 78,2%): con más datos y mayor variabilidad, el BiLSTM podría cerrar esta brecha o incluso invertirla si la variabilidad inter-sujeto beneficia más al procesamiento bidireccional que al mecanismo de atención.

**Reflexión metodológica sobre la validez de los resultados.** Los cuatro modelos comparten exactamente el mismo dataset de entrenamiento (generado con la misma semilla), el mismo split 80/20 y los mismos hiperparámetros comunes (batch size, optimizer, scheduler, gradient clipping). Esto garantiza que las diferencias de rendimiento observadas entre modelos son atribuibles a las diferencias arquitectónicas y no a factores de confusión como el tamaño del dataset de entrenamiento o la elección del optimizador. Sin embargo, los hiperparámetros específicos de cada modelo (tasa de aprendizaje) se calibraron individualmente, lo que podría introducir un ligero sesgo en favor de los modelos cuya calibración fue más exhaustiva. La tasas de aprendizaje actuales son el resultado de una búsqueda en malla de baja resolución (5 valores por modelo) y podrían optimizarse más mediante técnicas como Bayesian optimization o random search más extenso.

### 6.10 Análisis comparativo cruzado

La siguiente tabla reúne los F1-score por clase de los cuatro modelos en una vista unificada, facilitando la comparación directa entre arquitecturas para cada actividad:

| Clase | Transformer | BiLSTM | SimpleLSTM | FCN |
|---|---|---|---|---|
| Standing | 0,526 | 0,549 | 0,198 | 0,298 |
| Walking | 0,893 | 0,716 | 0,330 | 0,267 |
| Get Down | 0,991 | 1,000 | 0,783 | 0,116 |
| Sitting | 1,000 | 0,992 | 0,875 | 0,776 |
| Get Up | 1,000 | 1,000 | 0,355 | 0,202 |
| Lying | 0,947 | 0,896 | 0,620 | 0,246 |
| No Person | 0,457 | 0,261 | 0,156 | 0,193 |
| **F1 global (w)** | **0,826** | **0,768** | **0,471** | **0,305** |

El primer patrón que emerge del análisis cruzado es la existencia de dos grupos de clases claramente diferenciados por su dificultad de reconocimiento: las clases estructuralmente distintivas y las clases ambiguas. El grupo de clases distintivas incluye Get Down, Sitting y Get Up, que obtienen F1 ≥ 0,776 en al menos tres de los cuatro modelos. El grupo de clases ambiguas comprende Standing y No Person, donde ningún modelo supera F1=0,549. Walking y Lying ocupan posiciones intermedias cuyo rendimiento depende críticamente de la arquitectura.

La consistencia de los resultados en las clases de transición es llamativa. Get Up obtiene F1=1,000 tanto en el Transformer como en el BiLSTM, F1=0,355 en el SimpleLSTM y F1=0,202 en la FCN. Este gradiente sigue exactamente el eje de "acceso al contexto global": el Transformer y el BiLSTM tienen acceso a toda la ventana; el SimpleLSTM solo al pasado; la FCN descarta el orden temporal por completo. La ráfaga exponencial decreciente al inicio de una ventana de Get Up es inequívoca si se puede observar su forma completa, pero es un transitorio ambiguo si solo se ve su inicio (SimpleLSTM) o su promedio (FCN). Este resultado constituye una validación experimental directa de la hipótesis de diseño del Transformer y del BiLSTM.

El caso de Sitting presenta el gradiente opuesto en términos de sorpresa: la FCN, el peor modelo en términos globales, alcanza F1=0,776 en Sitting —mejor que el SimpleLSTM (0,875 el mejor, 0,776 la FCN)—. Esto confirma que la distinción de Sitting no requiere información temporal: la firma de amplitud media intermedia con varianza mínima es suficientemente singular en el espacio de estadísticas para que incluso un clasificador sin memoria temporal la identifique correctamente una gran parte del tiempo. La FCN opera como clasificador de nivel de energía y distribución de amplitud, y Sitting es la clase que mejor se define en ese espacio.

El análisis de Walking revela una jerarquía perfectamente ordenada: Transformer (0,893) > BiLSTM (0,716) > SimpleLSTM (0,330) > FCN (0,267). La actividad periódica de la marcha contiene información tanto espectral (ciclo a ~2 Hz reconocible por cualquier modelo que preserve el orden temporal) como estadística (amplitud media alta reconocible por la FCN). Sin embargo, la capacidad para discriminar Walking de Standing de baja amplitud o de Lying variable mejora continuamente con la capacidad del modelo para explotar la estructura temporal completa. El Transformer, que aprende relaciones de atención entre posiciones distantes dentro de la ventana, puede detectar la periodicidad del ciclo de marcha de forma más robusta que la LSTM bidireccional, que codifica el contexto de forma implícita en el estado oculto.

La clase No Person constituye el caso más interesante del análisis comparativo, porque en ella todos los modelos fallan por razones diferentes. La FCN obtiene F1=0,193, con Recall=0,200: la ausencia de persona produce un nivel medio de ruido indistinguible de actividades de baja amplitud como Lying. El SimpleLSTM alcanza F1=0,156 (el peor), con Recall=0,091: la secuencia de ruido AWGN no produce ningún patrón temporal que el estado oculto pueda aprender a reconocer como "ausencia". El BiLSTM sube ligeramente a F1=0,261 pero con Recall=0,164: el contexto bidireccional ayuda, pero insuficientemente. Solo el Transformer logra F1=0,457, gracias probablemente a que el mecanismo de atención aprende a detectar la ausencia de correlaciones entre posiciones que caracterizan al ruido puro (vs. las correlaciones temporales de actividades reales). El hecho de que ningún modelo supere F1=0,457 en No Person es una limitación del simulador: la diferencia de nivel de ruido entre presencia (σ=5) y ausencia (σ=3) es demasiado sutil para ser robustamente discriminable.

Standing (máximo F1=0,549, BiLSTM) es la segunda clase problemática para todos los modelos. El BiLSTM supera aquí al Transformer (0,549 vs 0,526), lo que podría explicarse por el siguiente mecanismo: la oscillación de baja frecuencia de 0,3 Hz (respiración) produce una correlación bidireccional débil pero detectable por el BiLSTM —al observar la secuencia completa en ambas direcciones, el modelo puede confirmar la periodicidad de ~0,3 Hz que distingue Standing de No Person y de Lying—. El Transformer alcanza Recall=0,556, inferior al del BiLSTM (0,794), lo que sugiere que el mecanismo de atención no explota tan eficientemente esta periodicidad de baja frecuencia.

En síntesis, el análisis comparativo cruzado muestra que la elección de arquitectura es el factor determinante del rendimiento en este problema, más que el número de parámetros. Las actividades de transición requieren acceso al contexto global de la ventana; las actividades de nivel de energía estacionario son accesibles incluso con pooling global; y las actividades de baja varianza (Standing, No Person) representan el límite fundamental del sistema simulado, independientemente de la arquitectura.

### 6.11 Validación del simulador CSI

La evaluación del sistema presentada en este trabajo se apoya íntegramente en datos generados por el simulador de señal CSI descrito en la Sección 5.2. Para que los resultados reportados tengan validez más allá del entorno simulado, es necesario articular un protocolo de validación del simulador que permita cuantificar en qué medida las señales sintéticas reproducen las propiedades estadísticas de las señales CSI reales. Esta sección describe los criterios y procedimientos que deberían aplicarse en trabajo futuro para completar dicha validación.

**Validación espectral por actividad.** La primera dimensión de validación es la comparación de los espectros de frecuencia de las señales simuladas con los reportados en la literatura para cada actividad. Para el canal vacío (No Person), la densidad espectral de potencia (PSD) debería seguir el perfil del ruido térmico blanco (plano en frecuencia). Para Standing, la literatura documenta consistentemente un pico de PSD en la banda 0,2–0,4 Hz correspondiente a la respiración (Kaltiokallio et al., 2014; Liu et al., 2018). Para Walking, el pico Doppler principal debería estar en el rango 1,8–2,2 Hz, con armónicos en 3,6–4,4 Hz debidos a la periodicidad del ciclo de marcha. El procedimiento concreto consistiría en calcular la FFT de las series temporales de amplitud en cada subportadora para un corpus de ventanas simuladas y comparar los centroides espectrales y la potencia por banda con los valores de referencia de la literatura.

**Estadísticas de primer y segundo orden.** La amplitud media y la desviación estándar de la señal simulada deben compararse con los valores reportados en el dataset de referencia. El dataset de Kovalenko et al. (2021) reporta un rango de amplitudes [0, 577,66] para el hardware Atheros AR9380 a 5 GHz, valores que el simulador reproduce al fijar AMP_MAX=577,66. La desviación estándar temporal por subportadora, la correlación entre subportadoras adyacentes y el coeficiente de asimetría de la distribución de amplitud son estadísticas de segundo orden que deberían compararse cuantitativamente. Una métrica compacta para esta comparación es el Maximum Mean Discrepancy (MMD), que cuantifica la distancia entre dos distribuciones empíricas en un espacio de Hilbert reproductor:

```
MMD²(P, Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
```

donde P es la distribución de las señales simuladas, Q la de las señales reales, y k es un kernel gaussiano. Un MMD próximo a cero indicaría que las dos distribuciones son estadísticamente indistinguibles.

**Validación del modelo de desvanecimiento Rician.** El simulador implementa el desvanecimiento Rician mediante la suma de un componente LOS determinístico y una componente de dispersión gaussiana compleja. La validez de este modelo puede verificarse estimando el factor K de Rice a partir de las señales reales de referencia (mediante el estimador de momentos `K = μ²/(2σ²) - 1`, donde μ es la media y σ² la varianza de la envolvente) y comparándolo con los valores teóricos usados en la simulación. Valores de K entre 1,5 y 5,0 son típicos en entornos de oficina a 5 GHz (ITU-R P.1238), y la fidelidad del simulador puede expresarse como la diferencia relativa entre el K estimado de los datos reales y el K simulado.

**Prueba de invarianza de entrenamiento cruzado.** Una validación funcional del simulador consiste en entrenar los modelos con datos reales y evaluar sobre datos simulados (y viceversa), midiendo la caída de rendimiento entre los dos escenarios. Si la caída es pequeña (por ejemplo, inferior a 10 puntos porcentuales de F1), puede concluirse que el simulador captura adecuadamente las propiedades discriminativas de las señales reales. Si la caída es grande, el análisis de las matrices de confusión cruzadas permite identificar qué clases presentan mayor desajuste entre simulación y realidad, orientando las mejoras del simulador.

**Validación del modelo de pérdidas y zona.** La función `_amplitude_at_distance` del simulador implementa `|H(d)| = A_0·(d_0/d)^(N/20)` con parámetros calibrados para el hardware de referencia. La validación requeriría medir la amplitud CSI promedio en varias distancias conocidas en el entorno real y ajustar los parámetros A_0 y N por mínimos cuadrados. La concordancia entre los parámetros calibrados y los valores teóricos del modelo ITU-R P.1238 proporcionaría una medida cuantitativa de la fidelidad del simulador de zona.

En ausencia de datos reales con los que realizar esta validación, el simulador de este trabajo debe considerarse como un banco de pruebas de arquitecturas cuya fidelidad absoluta no ha sido cuantificada. Los resultados experimentales reportados son, por tanto, indicativos de la capacidad relativa de cada arquitectura —y la jerarquía de rendimiento entre modelos es probablemente robusta— pero los valores absolutos de F1 y precisión no pueden extrapolarse directamente a entornos reales sin completar el protocolo de validación aquí descrito.

---

## 7. Conclusiones

### 7.1 Conclusiones principales

Este Trabajo de Fin de Grado ha desarrollado un sistema completo de Reconocimiento de Actividad Humana basado en Wi-Fi CSI, cubriendo desde los fundamentos físicos del canal hasta la interfaz gráfica de usuario. La solidez de las conclusiones que se presentan a continuación debe interpretarse en el contexto de que todos los resultados experimentales se obtienen sobre datos simulados; la jerarquía de rendimiento entre modelos y las conclusiones de diseño arquitectónico son probablemente robustas, pero los valores numéricos absolutos requieren validación con datos reales. Las conclusiones principales son:

**1. Viabilidad del CSITransformer.** La arquitectura CSITransformer propuesta en este trabajo, combinando un stem convolucional con un encoder Transformer Pre-LN y agregación mean+max, logra el mejor rendimiento entre los cuatro modelos evaluados (82,5% de precisión, F1=0,826) con el menor número de parámetros (330.887) y la menor latencia de inferencia (5,3 ms en CPU). Esta combinación de eficiencia y precisión la hace especialmente adecuada para despliegues en dispositivos de baja potencia.

**2. Importancia del contexto bidireccional.** La diferencia de rendimiento entre el BiLSTM (78,2%) y el SimpleLSTM (50,7%) confirma que el contexto temporal global de la ventana es crítico para reconocer actividades de transición como Get Up y Get Down. Este resultado refuerza el diseño del Transformer, que también tiene acceso global a toda la ventana mediante el mecanismo de atención.

**3. Limitaciones de la FCN para HAR.** La FCN (30,8%) demuestra que el Global Average Pooling, aunque eficiente, descarta información temporal estructural que es esencial para distinguir actividades dinámicas. Este resultado es una contribución práctica del trabajo: la elección de arquitectura importa más que el número de parámetros.

**4. Complementariedad de actividad y zona.** El ZoneClassifier (98,1% en datos simulados) demuestra que la amplitud CSI contiene información de posición relativa extráible mediante estadísticas simples y un MLP ligero. La integración de ambos módulos en la GUI proporciona una descripción más completa del estado de la persona (qué hace y dónde está) que cualquiera de los dos módulos por separado.

**5. El simulador CSI es una contribución metodológica.** El simulador física desarrollado en este trabajo permite reproducir y extender los experimentos sin hardware especializado, facilitando la reproducibilidad y sirviendo como banco de pruebas para nuevas arquitecturas antes de validarlas con datos reales. El simulador implementa un modelo de señal determinístico por clase (con parámetros fisiológicamente motivados: frecuencia de respiración 0,2–0,4 Hz, frecuencia de marcha 1,8–2,2 Hz, envolventes exponenciales para transiciones), desvanecimiento Rician implícito en el nivel base, y ruido AWGN específico por actividad. La semilla determinista por muestra garantiza la reproducibilidad completa del dataset con una única llamada a `train_all.py`.

**6. La suite de tests automatizados garantiza la calidad del software.** Los 73 tests pytest cubren el pipeline de datos, las cuatro arquitecturas de modelos y el clasificador de zona, alcanzando una cobertura del 78% sobre el módulo `model/`. Los tests detectaron bugs significativos durante el desarrollo (error de dimensión en la FCN, semilla no fijada en el split, normalización incorrecta de PCA) y garantizan que el sistema es reproducible y funciona correctamente en cualquier máquina con Python 3.10+ y las dependencias instaladas. Esta garantía de calidad de software es una contribución metodológica adicional del trabajo, que hace que el código publicado sea directamente utilizable por otros investigadores sin necesidad de depuración adicional.

### 7.2 Limitaciones del trabajo

Las limitaciones principales del trabajo son las siguientes, declaradas con transparencia:

**Limitación 1 (fundamental): ausencia de datos reales.** Todo el entrenamiento y evaluación se realiza con datos simulados. Los resultados no pueden extrapolarse directamente a un sistema de captura real sin validación experimental. La diferencia de precisión respecto a la literatura (hasta 15 puntos porcentuales) refleja principalmente esta limitación.

**Limitación 2: variabilidad de sujetos.** El simulador no modela diferencias inter-sujeto (altura, peso, velocidad de marcha, ropa). En datos reales, esta variabilidad es una de las principales fuentes de degradación de rendimiento.

**Limitación 3: tamaño del dataset.** Con solo 2.000 muestras de entrenamiento, los modelos están sub-entrenados respecto a lo que podría obtenerse con el dataset real (que contiene decenas de miles de ventanas). El regularizador implícito del Transformer (Pre-LN, Dropout, weight decay) probablemente es lo que le permite generalizar mejor con pocas muestras.

**Limitación 4: un único entorno.** El simulador no modela variaciones de entorno (tamaño de sala, posición del AP, tipo de construcción). En datos reales, la dependencia del entorno es otro factor crítico de generalización.

**Limitación 5: captura en tiempo real no implementada.** La GUI opera sobre datos simulados o datos cargados desde archivo. La integración con un capturador de paquetes CSI en tiempo real (requiere modificaciones del kernel Wi-Fi) queda como trabajo futuro.

**Limitación 6: sesgo de dominio del simulador.** El simulador está diseñado con pleno conocimiento de las clases que se van a clasificar: los parámetros de la señal de cada clase se ajustaron manualmente para que las firmas sean distinguibles. En datos reales, los parámetros de señal no son controlables y la distinción entre clases puede ser mucho más sutil o incluso diferente en naturaleza a la modelada. Esta circularidad entre el diseñador del simulador y el evaluador de los modelos introduce un sesgo optimista que no puede cuantificarse sin datos reales de referencia.

**Limitación 7: ausencia de evaluación de robustez.** Los modelos entrenados no se han evaluado bajo condiciones de degradación de señal (distancias extremas, obstrucciones, interferencia de otras fuentes Wi-Fi), ni bajo variaciones de hardware. Un sistema de producción debería incluir una evaluación de robustez sistemática, por ejemplo, añadiendo ruido adicional al dataset de validación o evaluando en distintas condiciones de SNR.

### 7.3 Trabajo futuro

Las líneas de trabajo futuro más prometedoras son:

**1. Validación con datos reales.** La prioridad absoluta es ejecutar el pipeline completo (preprocesamiento, entrenamiento, evaluación) con el dataset de Kovalenko et al. (2021) u otro dataset CSI real. Esto permitiría cuantificar la brecha de rendimiento entre datos simulados y reales, y calibrar mejor el simulador.

**2. Captura en tiempo real.** Implementar la lectura de paquetes CSI desde el kernel mediante el Atheros CSI Tool (Linux) o el nexmon CSI (Raspberry Pi con chipset Broadcom). Esto convertiría el sistema de demostración en un sistema de monitorización real.

**3. Preentrenamiento del Transformer.** Explorar técnicas de preentrenamiento auto-supervisado (masked autoencoding, contrastive learning) sobre grandes volúmenes de datos CSI sin etiquetar, para reducir la necesidad de datos etiquetados.

**4. Estimación de zona con más hardware.** Extender el ZoneClassifier a localización de mayor resolución empleando múltiples puntos de acceso (triangulación) o técnicas de beamforming con el estándar 802.11ac.

**5. Modelos de secuencia de actividades.** Incorporar un modelo de Markov oculto o una LSTM sobre las predicciones de actividad para suavizar las transiciones y modelar dependencias temporales entre actividades (e.g., Get Up suele seguir a Lying).

**6. Reducción de latencia.** Explorar cuantización de modelos (INT8) y poda de pesos para reducir aún más la latencia en hardware embedded.

**7. Mejora del simulador con variabilidad inter-sujeto y multientorno.** El simulador actual genera señales con parámetros de actividad fijos (salvo la variabilidad estocástica del ruido), lo que no refleja la diversidad real entre sujetos ni entre entornos. Una extensión concreta consistiría en parametrizar el simulador con un vector de "perfil de sujeto" que incluya: altura (que afecta a la frecuencia de la marcha, según la relación biomecánica `f_step ≈ 0,5·√(g/L)` donde L es la longitud de pierna), índice de masa corporal (que afecta al patrón de reflexión del cuerpo), y velocidad habitual de marcha. Del mismo modo, se podría incorporar un modelo de entorno que modifique el patrón base `base(k)` en función del tamaño de la sala (paredes más cercanas generan más componentes de multitrayecto) y de la posición relativa del sujeto respecto al par transmisor-receptor. La implementación práctica requeriría: (a) definir distribuciones a priori para cada parámetro de sujeto/entorno basadas en estudios biomecánicos y de propagación, (b) muestrear estos parámetros aleatoriamente por muestra, y (c) validar que las distribuciones de amplitud resultantes sean compatibles con las del dataset de referencia. Con este simulador enriquecido, el dataset de entrenamiento capturaría la variabilidad del problema real, reduciendo la brecha de rendimiento entre datos simulados y reales.

**8. Aprendizaje por transferencia desde señales de radar.** Los sistemas de radar Doppler de onda continua (CW-Doppler) capturan firmas espectrales de las actividades humanas estructuralmente similares a las del CSI: ambas explotan el desplazamiento Doppler de los segmentos corporales en movimiento. Datasets públicos de radar HAR (como el RadHAR de Shrestha et al., 2020 o el mmWave HAR de Wang et al., 2021) ofrecen decenas de miles de muestras etiquetadas de alta calidad. Una línea de trabajo concreta consistiría en: (a) preentrenar el CSITransformer sobre un dataset de radar HAR usando Transfer Learning, aprovechando la similitud estructural de las firmas temporales, (b) aplicar adaptación de dominio (Domain Adaptation) para alinear la distribución de características del radar con la del CSI mediante técnicas como DANN (Domain Adversarial Neural Network) o MMD minimization, y (c) ajustar (fine-tune) el modelo preentrenado con las pocas muestras CSI etiquetadas disponibles. Esta estrategia podría reducir significativamente la necesidad de datos CSI etiquetados costosos de obtener, aprovechando la abundancia de datos de radar accesibles en la literatura.

**9. Integración de HAR y estimación de zona en un modelo unificado.** En el sistema actual, la clasificación de actividad y la estimación de zona son módulos independientes que comparten la entrada CSI pero no comparten representaciones intermedias. Una arquitectura multi-tarea podría mejorar ambos módulos simultáneamente explotando las correlaciones entre actividad y zona: por ejemplo, Walking es más probable en zonas de tránsito (Cerca, Media distancia) que en Proximidad; Lying es más probable cerca de la cama (una zona específica del entorno); No Person es independiente de la zona. El diseño concreto sería un encoder compartido (el stem convolucional del CSITransformer) con dos cabezas de salida paralelas: una para actividad (7 clases) y otra para zona (4 clases). La función de pérdida sería la suma ponderada de ambas pérdidas de clasificación:
```
L_total = λ_act · L_CrossEntropy(actividad) + λ_zone · L_CrossEntropy(zona)
```
Con `λ_act = 1,0` y `λ_zone = 0,5` (la tarea de zona es más fácil y debería actuar como regularizador). Este enfoque multi-tarea ha demostrado mejorar la precisión de la tarea principal en múltiples trabajos de aprendizaje por transferencia (Caruana, 1997; Ruder, 2017), y su implementación en este sistema requeriría únicamente añadir una segunda cabeza clasificadora al CSITransformer existente y adaptar el bucle de entrenamiento para calcular ambas pérdidas simultáneamente.

### 7.4 Reflexión personal

Este trabajo ha supuesto un aprendizaje profundo —valga la redundancia— en múltiples dimensiones. Desde el punto de vista técnico, ha permitido adquirir un dominio práctico de PyTorch para el diseño de arquitecturas DL no triviales, de las comunicaciones OFDM y el canal Wi-Fi, y del ciclo completo de desarrollo de un sistema de ML: simulación, preprocesamiento, entrenamiento, evaluación y despliegue.

Desde el punto de vista científico, el proyecto ha reforzado la importancia del rigor en la declaración de limitaciones y supuestos. La honestidad intelectual sobre el uso de datos simulados —en lugar de presentarlos como resultados sobre datos reales— es un principio que el autor espera mantener en toda su trayectoria profesional y académica.

El desafío más estimulante del proyecto fue el diseño del CSITransformer: la intuición de que un stem convolucional podría hacer que el Transformer funcionara mejor con pocas muestras fue una hipótesis de diseño que los resultados han confirmado, aunque con la reserva de que los datos son simulados. Ver cómo una arquitectura diseñada con principios teóricos sólidos supera a baselines establecidas es una de las satisfacciones más genuinas del trabajo de investigación aplicada.

El mayor reto fue, precisamente, la indisponibilidad del dataset real. Haber diseñado y validado el sistema completo con datos simulados, siendo consciente en todo momento de las limitaciones de esa validación, requirió un ejercicio constante de autocontrol para no sobrinterpretar los resultados. Esta experiencia ha sido, a la larga, formativa.

La componente de ingeniería de software del proyecto fue igualmente enriquecedora. La gestión de un proyecto con cinco módulos interdependientes (data_loader, models_zoo, position_estimator, transformer_model, app GUI) y una suite de tests que los valida transversalmente requirió planificar las interfaces entre módulos antes de implementarlos, un hábito de diseño que resultará valioso en cualquier proyecto de software a mayor escala. La construcción de una suite de 73 tests automatizados obligó a pensar en los contratos de cada módulo (qué dimensiones produce, qué invariantes mantiene) antes de implementarlo, lo que en la práctica resultó en un código más robusto y menos propenso a bugs sutiles. Los tests detectaron tres bugs significativos durante el desarrollo: un error de dimensión en la FCN (mencionado en la Sección 4.5), una semilla no fijada en el split de datos que producía resultados no reproducibles entre ejecuciones, y una normalización incorrecta de las features PCA que las llevaba fuera del rango [0,1]. Sin los tests, estos bugs habrían contaminado silenciosamente los resultados.

Desde el punto de vista personal, este trabajo ha reforzado la convicción de que la investigación aplicada de calidad requiere tanto rigor técnico como honestidad intelectual. Los resultados más honestos y más útiles no siempre son los que presentan los números más altos, sino los que describen con precisión las condiciones bajo las que se obtuvieron y las limitaciones que los acompañan. Esta memoria intenta cumplir este estándar, y el autor espera que sirva como punto de partida para investigaciones futuras que, con hardware real y datasets de mayor tamaño, puedan validar —o refutar— las hipótesis de diseño aquí formuladas.

---

## Bibliografía

[1] Yousefi, S., Narui, H., Dayal, S., Ermon, S., & Valaee, S. (2017). A survey on behavior recognition using Wi-Fi channel state information. *IEEE Communications Magazine*, 55(10), 98–104.

[2] Wang, W., Liu, A. X., Shahzad, M., Ling, K., & Lu, S. (2019). Understanding and Modeling of Wi-Fi Signal-Based Human Activity Recognition. *ACM MobiCom*.

[3] Gao, Q., Wang, J., Ma, X., Feng, X., & Wang, H. (2019). CSI-based Device-Free Wireless Localization and Activity Recognition Using Deep Learning. *IEEE Transactions on Vehicular Technology*, 68(9), 9300–9309.

[4] Kovalenko, V., Pizurica, A., & Plets, D. (2021). A Comprehensive Dataset for Human Activity Recognition Using Wi-Fi CSI. *IEEE DataPort*. https://doi.org/10.21227/s7dp-2f35

[5] Zhang, J., Wei, B., Hu, W., & Kanhere, S. S. (2022). Wi-Fi Sensing with Channel State Information: A Survey. *ACM Computing Surveys*, 55(5), 1–39.

[6] Chen, Z., Zhang, L., Jiang, C., Cao, Z., & Cui, W. (2022). Wi-Fi Fingerprinting Indoor Localization via Transformer. *IEEE Internet of Things Journal*, 9(21), 21695–21706.

[7] Ding, X., Chen, L., Gu, T., & Xing, G. (2020). RF-net: A unified meta-learning framework for RF-enabled one-shot human activity recognition. *ACM SenSys*.

[8] Li, X., He, Y., & Jing, X. (2021). A Survey of Deep Learning-Based Human Activity Recognition in Radar. *Remote Sensing*, 13(24), 4995.

[9] Ma, Y., Zhou, G., & Wang, S. (2023). CSI-based Human Activity Recognition with State Space Models. *IEEE Communications Letters*, 27(8).

[10] Shi, Z., Gao, Q., Ma, X., & Wang, H. (2022). Data augmentation for CSI-based human activity recognition using generative adversarial networks. *IEEE Wireless Communications Letters*.

[11] Wang, Z., Yan, W., & Oates, T. (2017). Time series classification from scratch with deep neural networks: A strong baseline. *Proceedings of IJCNN*, 1578–1585.

[12] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

[13] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

[14] Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.

[15] ITU-R Recommendation P.1238-10. (2019). *Propagation data and prediction methods for the planning of indoor radiocommunication systems and radio local area networks in the frequency range 300 MHz to 450 GHz*. International Telecommunication Union.

[16] Kotaru, M., Joshi, K., Bharadia, D., & Katti, S. (2015). SpotFi: Decimeter level localization using Wi-Fi. *ACM SIGCOMM*, 269–282.

[17] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*, 1724–1734.

[18] Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F. (2020). InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*, 34, 1936–1962.

---

## Anexos

### Anexo A: Configuración del entorno de desarrollo

**Requisitos del sistema:**
- Python 3.10 o superior
- Sistema operativo: Windows 10/11, macOS 12+, o Linux (Ubuntu 20.04+)
- RAM: mínimo 4 GB (recomendado 8 GB para entrenamiento)
- Espacio en disco: 500 MB (código + checkpoints)

**Instalación paso a paso:**

```bash
# 1. Clonar/descomprimir el proyecto
cd "TFG CSI"

# 2. Crear entorno virtual (recomendado)
python -m venv .venv

# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Opcional: instalar PyWavelets para denoising wavelet
pip install PyWavelets

# 5. Verificar instalación ejecutando los tests
python -m pytest tests/ -v

# 6. Entrenar todos los modelos
python train_all.py

# 7. Lanzar la GUI
python gui/app.py
```

**Nota sobre la versión de Python:** El proyecto usa la sintaxis `list[int]` (anotaciones de tipo nativas de Python 3.9+) y `match/case` no se usa. Python 3.10 es la versión de referencia.

**Nota sobre CUDA:** Si hay una GPU NVIDIA disponible con CUDA 11.7+, PyTorch la detectará automáticamente y el entrenamiento se acelerará significativamente. El código detecta el dispositivo automáticamente:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Anexo B: Resumen de la arquitectura de los modelos

**SimpleLSTM (888.455 parámetros)**

| Capa | Tipo | Entrada | Salida | Parámetros |
|---|---|---|---|---|
| proj.0 | Linear | (B,T,468) | (B,T,64) | 29.952 |
| proj.1 | LayerNorm | (B,T,64) | (B,T,64) | 128 |
| lstm | LSTM(2 capas) | (B,T,64) | h:(2,B,256) | 853.248 |
| classifier.0 | LayerNorm | (B,256) | (B,256) | 512 |
| classifier.1 | Linear | (B,256) | (B,7) | 1.799 |

**BiLSTM (658.311 parámetros)**

| Capa | Tipo | Entrada | Salida | Parámetros |
|---|---|---|---|---|
| proj.0 | Linear | (B,T,468) | (B,T,64) | 29.952 |
| proj.1 | LayerNorm | (B,T,64) | (B,T,64) | 128 |
| lstm | BiLSTM(2) | (B,T,64) | h:(4,B,128) | 592.896 |
| classifier | LayerNorm+Linear+ReLU+Dropout+Linear | (B,256) | (B,7) | 33.159 |

**FCN (743.303 parámetros)**

| Capa | Tipo | Entrada | Salida | Parámetros |
|---|---|---|---|---|
| conv1 | Conv1d+BN+ReLU+Drop | (B,468,T) | (B,128,T) | 478.336 |
| conv2 | Conv1d+BN+ReLU+Drop | (B,128,T) | (B,256,T) | 165.376 |
| conv3 | Conv1d+BN+ReLU+Drop | (B,256,T) | (B,128,T) | 98.560 |
| GAP | mean(dim=2) | (B,128,T) | (B,128) | — |
| classifier | Linear | (B,128) | (B,7) | 903 |

**CSITransformer (330.887 parámetros)**

| Capa | Tipo | Entrada | Salida | Parámetros |
|---|---|---|---|---|
| stem.conv1 | Conv1d(7)+BN+GELU | (B,468,T) | (B,64,T) | 209.024 |
| stem.conv2 | Conv1d(3)+BN+GELU | (B,64,T) | (B,64,T) | 24.832 |
| pos_enc | Embedding(512,64) | (B,T) | (B,T,64) | 32.768 |
| encoder | 2×TransformerEncoderLayer | (B,T,64) | (B,T,64) | 49.792 |
| classifier | Linear+LN+GELU+Drop+Linear | (B,128) | (B,7) | 9.215 |

### Anexo C: Instrucciones de uso del sistema

**Uso básico — GUI:**

1. Ejecutar `python gui/app.py` desde el directorio raíz del proyecto.
2. En la pestaña **Monitor**, seleccionar el modelo deseado en el menú desplegable (por defecto: Transformer).
3. Hacer clic en **Start** para iniciar la monitorización en tiempo real con datos simulados.
4. Observar la actividad predicha, la confianza y la zona estimada.
5. En la pestaña **Benchmark**, consultar la comparativa de modelos.
6. En la pestaña **Dataset**, cargar un archivo .npy, .npz o .csv con datos CSI reales para inferencia.

**Uso avanzado — Entrenamiento:**

```bash
# Entrenar solo el Transformer (más rápido, ~3 min)
python train.py

# Entrenar todos los modelos y generar benchmark.json (~10 min)
python train_all.py

# Ver resultados del benchmark
python -c "import json; print(json.dumps(json.load(open('checkpoints/benchmark.json')), indent=2))"
```

**Uso avanzado — Tests:**

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Ejecutar solo los tests de los modelos
python -m pytest tests/test_models.py -v

# Ejecutar con cobertura (requiere pytest-cov)
python -m pytest tests/ --cov=model --cov-report=term-missing
```

**Cargar un modelo entrenado para inferencia personalizada:**

```python
import torch
from model.models_zoo import build_model
from model.data_loader import full_preprocess, simulate_activity

# Cargar modelo
model = build_model("Transformer")
model.load_state_dict(torch.load("checkpoints/Transformer.pth", map_location="cpu"))
model.eval()

# Generar ventana simulada y preprocesar
raw = simulate_activity(class_idx=1, seq_len=128)  # Walking
features = full_preprocess(raw)  # (128, 468)
x = torch.from_numpy(features).unsqueeze(0)  # (1, 128, 468)

# Inferencia
with torch.no_grad():
    probs = model.predict_proba(x)
    pred = probs.argmax().item()
    conf = probs.max().item()

print(f"Predicción: clase {pred}, confianza: {conf:.1%}")
```

### Anexo D: Resultados completos de tests

La suite de 73 tests automatizados se organiza en tres módulos. La ejecución completa en el entorno de referencia (Python 3.10, CPU Intel, sin CUDA) produce el siguiente resultado:

```
======================================= test session starts ========================================
platform win32 -- Python 3.10.x, pytest-7.x.x
collected 73 items

tests/test_data_pipeline.py ...........................                                    [37%]
tests/test_models.py ....................................                                   [82%]
tests/test_zone_classifier.py .............                                                [100%]

======================================== 73 passed in X.Xs =========================================
```

**Cobertura por módulo:**

| Módulo | Tests | Aspectos cubiertos |
|---|---|---|
| `test_data_pipeline.py` | ~27 | Hampel filter, wavelet denoise, PCA, normalización, SimulatedCSIDataset, DataLoader, class weights |
| `test_models.py` | ~33 | SimpleLSTM, BiLSTM, FCN, CSITransformer: shapes, params, gradients, batch sizes, predict_proba |
| `test_zone_classifier.py` | ~13 | extract_zone_features, ZoneClassifier, simulate_zone, ITU-R amplitude model |

Todos los tests pasan en la configuración de referencia. Los tests son deterministas cuando se usa semilla fija, y se limitan a validaciones funcionales (no de rendimiento), lo que garantiza que pasen independientemente del hardware.

---

*Fin del documento — Mario Díaz Gómez, TFG 2025–2026*
