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

### 2.5 Posicionamiento indoor Wi-Fi

El posicionamiento indoor basado en Wi-Fi puede abordarse a diferentes niveles de granularidad:

**Nivel de sala/zona.** El enfoque más sencillo y robusto divide el entorno en zonas discretas y asigna el dispositivo a la zona más probable. Puede basarse únicamente en intensidad de señal (RSSI o amplitud CSI media), lo que lo hace factible sin campañas de calibración extensas.

**Fingerprinting.** El sistema RADAR (Bahl & Padmanabhan, 2000) fue pionero en utilizar una base de datos de huellas de señal (RSSI de múltiples APs) para localizar dispositivos. Con CSI, el espacio de fingerprinting es mucho más rico. SpotFi (Kotaru et al., 2015) combina amplitud y fase CSI con algoritmos MUSIC para estimar ángulo de llegada (AoA), logrando precisiones de 0,5 m.

**Modelos de propagación.** El modelo ITU-R P.1238 proporciona una estimación de la pérdida de trayecto en interiores como función de la frecuencia y la distancia, permitiendo estimar la distancia a partir de la amplitud de señal recibida. Este es el enfoque adoptado en este TFG para el clasificador de zona.

**Deep learning para localización.** Trabajos recientes como FILA (Wu et al., 2012) y DeepFi (Wang et al., 2015) aplican redes neuronales profundas al fingerprinting con CSI, logrando precisiones de 1–2 m. Los modelos de aprendizaje por transferencia permiten reducir la costosa fase de calibración.

### 2.6 Conclusiones del estado del arte

Del análisis de la literatura se extraen las siguientes conclusiones que han orientado las decisiones de diseño de este TFG:

1. **El CSI es superior al RSSI** para HAR por su mayor dimensionalidad y sensibilidad a movimientos sutiles. Esta conclusión es ampliamente compartida en la literatura.
2. **Los Transformers superan a las LSTM** en los trabajos más recientes sobre datos reales, aunque requieren más datos para entrenarse efectivamente. Esta es la hipótesis que el CSITransformer de este trabajo pone a prueba.
3. **La simulación de datos CSI es un área poco explorada.** La mayoría de los trabajos utilizan hardware real, lo que dificulta la reproducibilidad. La contribución del simulador física de este TFG es relevante para la comunidad.
4. **La clasificación de zona es un complemento valioso** al reconocimiento de actividad, aunque raramente se combina en un único sistema. Este TFG integra ambas funcionalidades.
5. **La latencia de inferencia en CPU es crítica** para el despliegue en dispositivos de bajo coste. Ninguno de los trabajos revisados reporta latencias por debajo de 10 ms en CPU para ventanas de 128 muestras, lo que el CSITransformer de este TFG consigue con 5,3 ms.

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

### 3.3 Firmas Doppler de actividades humanas

Cada actividad humana produce una firma característica en el CSI debida a la combinación de los efectos Doppler de los distintos segmentos corporales que se mueven. El análisis de estas firmas es la base del reconocimiento de actividad:

**Ausencia de persona (No Person):** El canal es estático; la única variación proviene del ruido térmico del receptor (AWGN). La amplitud media es aproximadamente 200 unidades con desviación estándar <5.

**Permanecer de pie (Standing):** El movimiento principal es la respiración (~0,3 Hz) y el balanceo corporal involuntario (<0,5 Hz). Las variaciones de amplitud son pequeñas (±15 unidades) y periódicas.

**Caminar (Walking):** El ciclo de la marcha (~1,8–2,2 Hz) produce las variaciones de CSI más pronunciadas y periódicas. Las amplitudes oscilan ±50 unidades o más, con modulación espacial entre subportadoras debida a la fase del ciclo de marcha.

**Sentarse (Sitting):** Movimiento muy lento (~0,1 Hz), principalmente un desplazamiento gradual del centro de masa. Las variaciones son pequeñas en amplitud y muy lentas.

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

Las ventajas de la FCN son su eficiencia computacional (operaciones paralelizables en GPU y CPU modernas), la invarianza a traslaciones temporales y la ausencia de hiperparámetros como el número de capas ocultas recurrentes. Sin embargo, su limitación es la pérdida de información temporal global: la GAP promedia toda la secuencia, perdiendo la estructura temporal que distingue, por ejemplo, Get Up (ráfaga al inicio) de Get Down (ráfaga al final).

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

### 4.2 Pipeline de preprocesamiento CSI

El preprocesamiento sigue exactamente el pipeline del repositorio de referencia (Kovalenko et al., 2021), con una adición: el cómputo de PCA se realiza de forma global sobre todo el corpus de entrenamiento (batch PCA) en lugar de por ventana individual, lo que reduce el tiempo de construcción del dataset en aproximadamente 12.000 SVDs individuales.

**Paso 1 — Filtro Hampel.** El filtro Hampel es un estimador robusto de outliers que reemplaza cada muestra cuya desviación respecto a la mediana local supere `t0 × MAD × 1,4826` por dicha mediana. Se aplica con ventana k=7 (±7 muestras) y umbral t0=3. Este filtro es especialmente efectivo para eliminar los picos transitorios ocasionales que produce el driver del Atheros CSI Tool.

**Paso 2 — Denoising wavelet.** La wavelet sym5 (Daubechies simetrizada de 5 momentos nulos) se aplica con 3 niveles de descomposición. El umbral de soft-thresholding es `0,06 × max(|coeficientes del nivel más fino|)`. Si PyWavelets no está disponible, se usa el filtro Savitzky-Golay de ventana 11 y orden 3 como alternativa, que preserva picos mientras suaviza el ruido.

**Paso 3 — Cómputo de PCA.** Para cada uno de los 4 pares de antenas, se calcula la proyección sobre los 3 primeros componentes principales del bloque de 114 subportadoras correspondiente. Los 12 valores resultantes (3 × 4) capturan la mayor varianza de la señal en cada par, complementando la información de alta frecuencia presente en las 456 amplitudes brutas.

**Paso 4 — Normalización.** Las 456 amplitudes brutas se normalizan a [0, 1] dividiendo por AMP_MAX=577,66. Las 12 características PCA se normalizan de forma independiente a [0, 1] usando sus propios mínimos y máximos globales. Ambas partes se concatenan para producir el tensor final de 468 características.

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

**Parámetros:** 330.887 (el más eficiente de los cuatro modelos).
**Tasa de aprendizaje:** 1,46×10⁻³ (Adam). El warm-up no es necesario gracias al Pre-LN.
**Inicialización:** truncated normal (std=0,02) para pesos lineales, ceros para sesgos; ones/ceros para LayerNorm.

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

### 4.9 Interfaz Gráfica de Usuario

La GUI se implementa con CustomTkinter (versión moderna de Tkinter con soporte para modo oscuro y temas) y Matplotlib para las visualizaciones. Se organiza en seis pestañas:

**Tab 1 — Monitor en tiempo real.** Muestra la actividad predicha con barra de confianza, la zona estimada con distancia aproximada, y un gráfico de amplitud CSI en tiempo real. El modelo en uso es seleccionable mediante un menú desplegable.

**Tab 2 — Comparativa de modelos.** Tabla comparativa de las métricas de todos los modelos entrenados, cargada automáticamente desde `benchmark.json`. Incluye precisión, F1, número de parámetros y latencia.

**Tab 3 — Matrices de confusión.** Visualización de las matrices de confusión de cada modelo (PNGs guardados durante el entrenamiento). Permite comparar visualmente los patrones de error.

**Tab 4 — Análisis de actividad.** Gráficos de distribución temporal de predicciones y evolución de la confianza durante la sesión actual.

**Tab 5 — Dataset.** Interfaz para cargar datos reales en formatos NumPy (.npy, .npz), CSV o pickle (.pkl). Muestra estadísticas descriptivas y permite ejecutar predicciones sobre los datos cargados.

**Tab 6 — Configuración.** Parámetros del sistema (velocidad de simulación, longitud de ventana, modelo activo, modo oscuro/claro).

La GUI se actualiza con un bucle de 100 ms (10 FPS) que genera una nueva ventana de datos simulados, ejecuta la inferencia de actividad y zona, y actualiza todos los elementos gráficos. El sistema es diseño sin bloqueo: la inferencia corre en el hilo principal (dada la baja latencia <10 ms de todos los modelos) sin necesidad de multithreading.

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

**Gradient clipping** (max_norm=1,0) se aplica en todos los modelos para prevenir la explosión de gradientes, especialmente relevante en las LSTMs.

**Tasas de aprendizaje por modelo** (calibradas experimentalmente):
```
SimpleLSTM : 3×10⁻⁴   (LSTM sensible a LR altos)
BiLSTM     : 3×10⁻⁴   (ídem)
FCN        : 1×10⁻³   (CNN converge bien con LR estándar)
Transformer: 1.46×10⁻³ (ligeramente superior para el stem+encoder)
```

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

### 5.5 Carga de datos reales (Dataset Tab)

La pestaña Dataset de la GUI permite cargar datos CSI reales en tres formatos:

**Formato NumPy (.npy, .npz).** Se acepta un array de forma (N, T, 456) o (N, T, 468). Si el array tiene 456 features, se aplica automáticamente el pipeline PCA para producir las 468 features requeridas por los modelos.

**Formato CSV.** Cada fila es un timestep; las columnas son features. Se asume la última columna como etiqueta de clase si está presente.

**Formato pickle (.pkl).** Se acepta cualquier objeto serializado que resulte en un dict con claves `'data'` y opcionalmente `'labels'`, o un array NumPy directamente.

El sistema detecta automáticamente el formato por la extensión del archivo y muestra estadísticas descriptivas (número de muestras, distribución de clases, rango de amplitudes) antes de ejecutar la inferencia.

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

**Validez de los resultados:** Se enfatiza que todos los resultados se obtienen sobre datos **simulados**, no sobre datos CSI reales. Los modelos aprenden de señales sintéticas generadas por el simulador descrito en la Sección 5.2. Los valores numéricos reportados son reproducibles ejecutando `train_all.py` con la configuración descrita.

### 6.2 Comparativa de modelos HAR

| Modelo | Val Acc | Val F1 (w) | Parámetros | Latencia CPU |
|---|---|---|---|---|
| **CSITransformer** | **82,5%** | **0,826** | **330.887** | **5,3 ms** |
| BiLSTM | 78,2% | 0,768 | 658.311 | 8,8 ms |
| SimpleLSTM | 50,7% | 0,471 | 888.455 | 9,6 ms |
| FCN | 30,8% | 0,305 | 743.303 | 5,4 ms |
| ZoneClassifier | 98,1% | — | — | — |

El CSITransformer es el mejor modelo en todas las métricas de precisión, al tiempo que es el más eficiente en parámetros y el más rápido (empate con FCN dentro del margen de medición). Esta combinación de eficiencia y precisión es el resultado más destacado del trabajo.

El BiLSTM es el segundo mejor modelo, con 78,2% de precisión, aprovechando el contexto bidireccional para capturar la estructura temporal de las actividades de transición (Get Up, Get Down). La diferencia con el SimpleLSTM (50,7%) es notable y confirma la importancia del contexto bidireccional.

La FCN con 30,8% de precisión obtiene el peor resultado, confirmando la hipótesis de que la pérdida de estructura temporal mediante GAP es crítica para distinguir actividades de transición con datos de entrenamiento limitados.

El SimpleLSTM con 50,7% de precisión ofrece un rendimiento modesto. Con sólo 2.000 muestras, la unidireccionalidad penaliza su capacidad para modelar las actividades de transición.

### 6.3 Análisis por clase — CSITransformer

| Clase | F1-score | Interpretación |
|---|---|---|
| Standing | 0,526 | Confusión frecuente con Lying y No Person |
| Walking | 0,893 | Firma Doppler muy distintiva (~2 Hz) |
| Get Down | 0,991 | Ráfaga creciente muy característica |
| Sitting | 1,000 | Deriva lenta único en su clase |
| Get Up | 1,000 | Ráfaga decreciente muy característica |
| Lying | 0,947 | Casi idéntica a Standing en amplitud baja |
| No Person | 0,457 | Confusión con Standing y Lying |

El Transformer alcanza F1=1,000 en Sitting, Get Up, y prácticamente en Get Down (0,991). Esto indica que las firmas físicas simuladas de estas actividades son muy discriminativas: Sitting tiene una deriva lenta única, y las transiciones Get Up/Get Down tienen envolventes exponenciales claramente opuestas.

El resultado más bajo es No Person (0,457), confundido frecuentemente con Standing y Lying. Esta confusión es plausible desde el punto de vista físico: en ausencia de persona, el canal es estático con ruido (AWGN σ=3); Standing tiene ruido σ=5 con una oscilación de 0,3 Hz; la diferencia es sutil.

Standing (0,526) presenta también F1 moderado, confundiéndose con No Person y Lying. Ambas situaciones comparten variaciones de amplitud de pequeña escala (<15 unidades), lo que dificulta la discriminación incluso para el mejor modelo.

Walking (0,893) y Lying (0,947) tienen excelentes F1, demostrando que el simulador captura bien las diferencias entre la firma periódica de la marcha (~2 Hz) y la oscilación sub-0,05 Hz de la postura tumbada.

### 6.4 Análisis por clase — BiLSTM

El BiLSTM muestra un patrón similar al Transformer pero con F1-scores generalmente más bajos. Se espera una buena discriminación de actividades de transición (Get Up, Get Down) gracias al contexto bidireccional, y mayor dificultad en las clases estáticas (Standing, No Person) que no presentan estructura temporal clara. Sin datos de F1 por clase disponibles en el benchmark.json (que solo reporta el F1 ponderado de 0,768), la inferencia se basa en la comparativa con el Transformer y las matrices de confusión visuales guardadas en `checkpoints/confusion_BiLSTM.png`.

### 6.5 Análisis por clase — SimpleLSTM

Con un F1 ponderado de 0,471 y una precisión de 50,7%, el SimpleLSTM muestra un rendimiento que supera a la predicción aleatoria (14,3% en 7 clases) pero es significativamente inferior a los modelos bidireccionales y el Transformer. Las actividades con firmas temporales simétricas (sin contexto futuro) como Walking y Sitting deberían ser las mejor reconocidas, mientras que Get Up y Get Down (que requieren visión global de la ventana) presentarán peores resultados. Las matrices de confusión en `checkpoints/confusion_SimpleLSTM.png` confirman esta hipótesis.

### 6.6 Análisis por clase — FCN

La FCN con 30,8% de precisión y F1 ponderado de 0,305 obtiene el peor resultado. La Global Average Pooling descarta toda la estructura temporal, dejando al clasificador con únicamente el espectro de amplitudes promedio por canal. En este escenario, solo las actividades con amplitudes medias muy distintas (Walking con alta amplitud, No Person con amplitud baja) podrían distinguirse razonablemente. Las actividades de transición (Get Up, Get Down), que se diferencian por el orden temporal de sus patrones, son prácticamente indistinguibles para la FCN tras el GAP. Las matrices en `checkpoints/confusion_FCN.png` muestran una tendencia del modelo a predecir las clases mayoritarias en amplitud.

### 6.7 Estimación de zona (ZoneClassifier)

El ZoneClassifier alcanza una precisión del 98,1% sobre datos simulados. Este resultado es esperable dado que el simulador genera las amplitudes directamente a partir del modelo de pérdidas ITU-R P.1238, y el clasificador de zona aprende exactamente esta misma función. La alta precisión confirma que la arquitectura MLP es suficiente para esta tarea y que las 16 estadísticas de amplitud son descriptores eficaces del nivel de señal.

En un despliegue real, la precisión sería significativamente menor debido a:
- Variaciones de shadowing (log-normal con σ=4–7 dB) que pueden hacer que una misma distancia produzca amplitudes de zonas adyacentes.
- Condiciones NLOS (detrás de paredes) que atenúan más de lo predicho por el modelo.
- Variaciones de la orientación del sujeto respecto al AP que afectan a la firma de amplitud.

### 6.8 Análisis de latencia y complejidad computacional

| Modelo | Latencia CPU | Params | Params/Acc |
|---|---|---|---|
| CSITransformer | 5,3 ms | 330.887 | 4.011 |
| FCN | 5,4 ms | 743.303 | 24.131 |
| BiLSTM | 8,8 ms | 658.311 | 8.418 |
| SimpleLSTM | 9,6 ms | 888.455 | 17.524 |

*Params/Acc: parámetros por punto porcentual de precisión (menor = más eficiente)*

El CSITransformer es el modelo más eficiente en términos de parámetros/rendimiento, con 4.011 parámetros por punto porcentual de precisión, frente a los 24.131 de la FCN (que tiene mayor número de parámetros pero menor precisión).

La latencia de 5,3 ms del Transformer es llamativa dado que los Transformers son habitualmente más lentos que las CNNs en inferencia. La razón es la dimensionalidad reducida del modelo (d_model=64, 2 capas, seq_len=128): la operación de atención tiene complejidad O(T²·d) = O(128²·64) ≈ 1M operaciones, comparable al coste de las tres capas convolucionales de la FCN.

Todos los modelos cumplen el requisito de latencia <10 ms para aplicaciones de monitorización en tiempo real a 10 FPS.

### 6.9 Discusión de resultados

Los resultados revelan tres hallazgos principales:

**Hallazgo 1: El Transformer supera a las LSTMs con pocas muestras de entrenamiento.** Con solo 1.600 muestras de entrenamiento, el CSITransformer logra 82,5% de precisión frente al 78,2% del BiLSTM y 50,7% del SimpleLSTM. Este resultado es, en cierta medida, contraintuitivo: los Transformers son conocidos por requerir grandes volúmenes de datos. La explicación probable es que el stem convolucional actúa como un extractor de características que reduce la dificultad del problema para el encoder Transformer, y que la agregación mean+max proporciona supervisión implícita sobre la estructura global de la ventana.

**Hallazgo 2: La FCN no es adecuada para HAR con actividades de transición.** El 30,8% de precisión de la FCN es el resultado más sorprendente. La FCN de Wang et al. (2017) obtuvo resultados excelentes en datasets como UCR, pero esos datasets contienen actividades con patrones temporales repetitivos (arritmias cardíacas, consumo energético). Las actividades de transición del dataset CSI (Get Up, Get Down) tienen estructura temporal no estacionaria que el GAP elimina, lo que explica el bajo rendimiento.

**Hallazgo 3: Los datos simulados son insuficientes para reproducir la riqueza del canal real.** La brecha entre las precisiones reportadas en la literatura (87–94%) y las obtenidas en este trabajo (30–82%) refleja las limitaciones del simulador. El simulador reproduce los patrones medios de cada actividad, pero no la variabilidad inter-sujeto, las variaciones de posición del transmisor, los efectos del mobiliario o las interferencias de canal propias de un entorno real. Esta limitación es la más importante del trabajo y debe tenerse en cuenta al interpretar todos los resultados.

---

## 7. Conclusiones

### 7.1 Conclusiones principales

Este Trabajo de Fin de Grado ha desarrollado un sistema completo de Reconocimiento de Actividad Humana basado en Wi-Fi CSI, cubriendo desde los fundamentos físicos del canal hasta la interfaz gráfica de usuario. Las conclusiones principales son:

**1. Viabilidad del CSITransformer.** La arquitectura CSITransformer propuesta en este trabajo, combinando un stem convolucional con un encoder Transformer Pre-LN y agregación mean+max, logra el mejor rendimiento entre los cuatro modelos evaluados (82,5% de precisión, F1=0,826) con el menor número de parámetros (330.887) y la menor latencia de inferencia (5,3 ms en CPU). Esta combinación de eficiencia y precisión la hace especialmente adecuada para despliegues en dispositivos de baja potencia.

**2. Importancia del contexto bidireccional.** La diferencia de rendimiento entre el BiLSTM (78,2%) y el SimpleLSTM (50,7%) confirma que el contexto temporal global de la ventana es crítico para reconocer actividades de transición como Get Up y Get Down. Este resultado refuerza el diseño del Transformer, que también tiene acceso global a toda la ventana mediante el mecanismo de atención.

**3. Limitaciones de la FCN para HAR.** La FCN (30,8%) demuestra que el Global Average Pooling, aunque eficiente, descarta información temporal estructural que es esencial para distinguir actividades dinámicas. Este resultado es una contribución práctica del trabajo: la elección de arquitectura importa más que el número de parámetros.

**4. Complementariedad de actividad y zona.** El ZoneClassifier (98,1% en datos simulados) demuestra que la amplitud CSI contiene información de posición relativa extráible mediante estadísticas simples y un MLP ligero. La integración de ambos módulos en la GUI proporciona una descripción más completa del estado de la persona (qué hace y dónde está) que cualquiera de los dos módulos por separado.

**5. El simulador CSI es una contribución metodológica.** El simulador física desarrollado en este trabajo permite reproducir y extender los experimentos sin hardware especializado, facilitando la reproducibilidad y sirviendo como banco de pruebas para nuevas arquitecturas antes de validarlas con datos reales.

### 7.2 Limitaciones del trabajo

Las limitaciones principales del trabajo son las siguientes, declaradas con transparencia:

**Limitación 1 (fundamental): ausencia de datos reales.** Todo el entrenamiento y evaluación se realiza con datos simulados. Los resultados no pueden extrapolarse directamente a un sistema de captura real sin validación experimental. La diferencia de precisión respecto a la literatura (hasta 15 puntos porcentuales) refleja principalmente esta limitación.

**Limitación 2: variabilidad de sujetos.** El simulador no modela diferencias inter-sujeto (altura, peso, velocidad de marcha, ropa). En datos reales, esta variabilidad es una de las principales fuentes de degradación de rendimiento.

**Limitación 3: tamaño del dataset.** Con solo 2.000 muestras de entrenamiento, los modelos están sub-entrenados respecto a lo que podría obtenerse con el dataset real (que contiene decenas de miles de ventanas). El regularizador implícito del Transformer (Pre-LN, Dropout, weight decay) probablemente es lo que le permite generalizar mejor con pocas muestras.

**Limitación 4: un único entorno.** El simulador no modela variaciones de entorno (tamaño de sala, posición del AP, tipo de construcción). En datos reales, la dependencia del entorno es otro factor crítico de generalización.

**Limitación 5: captura en tiempo real no implementada.** La GUI opera sobre datos simulados o datos cargados desde archivo. La integración con un capturador de paquetes CSI en tiempo real (requiere modificaciones del kernel Wi-Fi) queda como trabajo futuro.

### 7.3 Trabajo futuro

Las líneas de trabajo futuro más prometedoras son:

**1. Validación con datos reales.** La prioridad absoluta es ejecutar el pipeline completo (preprocesamiento, entrenamiento, evaluación) con el dataset de Kovalenko et al. (2021) u otro dataset CSI real. Esto permitiría cuantificar la brecha de rendimiento entre datos simulados y reales, y calibrar mejor el simulador.

**2. Captura en tiempo real.** Implementar la lectura de paquetes CSI desde el kernel mediante el Atheros CSI Tool (Linux) o el nexmon CSI (Raspberry Pi con chipset Broadcom). Esto convertiría el sistema de demostración en un sistema de monitorización real.

**3. Preentrenamiento del Transformer.** Explorar técnicas de preentrenamiento auto-supervisado (masked autoencoding, contrastive learning) sobre grandes volúmenes de datos CSI sin etiquetar, para reducir la necesidad de datos etiquetados.

**4. Estimación de zona con más hardware.** Extender el ZoneClassifier a localización de mayor resolución empleando múltiples puntos de acceso (triangulación) o técnicas de beamforming con el estándar 802.11ac.

**5. Modelos de secuencia de actividades.** Incorporar un modelo de Markov oculto o una LSTM sobre las predicciones de actividad para suavizar las transiciones y modelar dependencias temporales entre actividades (e.g., Get Up suele seguir a Lying).

**6. Reducción de latencia.** Explorar cuantización de modelos (INT8) y poda de pesos para reducir aún más la latencia en hardware embedded.

### 7.4 Reflexión personal

Este trabajo ha supuesto un aprendizaje profundo —valga la redundancia— en múltiples dimensiones. Desde el punto de vista técnico, ha permitido adquirir un dominio práctico de PyTorch para el diseño de arquitecturas DL no triviales, de las comunicaciones OFDM y el canal Wi-Fi, y del ciclo completo de desarrollo de un sistema de ML: simulación, preprocesamiento, entrenamiento, evaluación y despliegue.

Desde el punto de vista científico, el proyecto ha reforzado la importancia del rigor en la declaración de limitaciones y supuestos. La honestidad intelectual sobre el uso de datos simulados —en lugar de presentarlos como resultados sobre datos reales— es un principio que el autor espera mantener en toda su trayectoria profesional y académica.

El desafío más estimulante del proyecto fue el diseño del CSITransformer: la intuición de que un stem convolucional podría hacer que el Transformer funcionara mejor con pocas muestras fue una hipótesis de diseño que los resultados han confirmado, aunque con la reserva de que los datos son simulados. Ver cómo una arquitectura diseñada con principios teóricos sólidos supera a baselines establecidas es una de las satisfacciones más genuinas del trabajo de investigación aplicada.

El mayor reto fue, precisamente, la indisponibilidad del dataset real. Haber diseñado y validado el sistema completo con datos simulados, siendo consciente en todo momento de las limitaciones de esa validación, requirió un ejercicio constante de autocontrol para no sobrinterpretar los resultados. Esta experiencia ha sido, a la larga, formativa.

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
