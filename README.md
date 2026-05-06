# CSI-HAR: Wi-Fi CSI Human Activity Recognition System

**TFG — Trabajo de Fin de Grado**
**Autor:** Mario Díaz Gómez | **Curso:** 2025-2026
**Repositorio:** https://github.com/ErGranPepe/csi-har-tfg

---

## Descripción

Sistema completo de **Reconocimiento de Actividad Humana (HAR)** basado en **Channel State Information (CSI) de Wi-Fi**. Clasifica 7 actividades en tiempo real con cuatro arquitecturas de redes neuronales profundas, estimación de zona de proximidad e interfaz gráfica avanzada de 6 pestañas.

| Modelo | Val Acc | Val F1 (w) | Parámetros | Latencia CPU |
|---|---|---|---|---|
| **CSITransformer** | **82.5%** | **0.826** | 330,887 | 5.3 ms |
| BiLSTM | 78.2% | 0.768 | 658,311 | 8.8 ms |
| SimpleLSTM | 50.7% | 0.471 | 888,455 | 9.6 ms |
| FCN | 30.8% | 0.305 | 743,303 | 5.4 ms |
| ZoneClassifier | 98.1% acc | — | — | — |

> **Nota científica:** Entrenado con datos CSI simulados físicamente (dataset real de 9 GB no disponible localmente). El pipeline de inferencia es idéntico al hardware real. Resultados reproducibles con `py train_all.py`.

---

## Actividades reconocidas

| ID | Actividad | Firma Doppler característica |
|----|-----------|------------------------------|
| 0 | Standing (De pie) | Oscilación respiratoria ~0.3 Hz |
| 1 | Walking (Caminando) | Ciclo de marcha ~2 Hz, alta varianza |
| 2 | Get Down (Tumbarse) | Ráfaga transitoria creciente |
| 3 | Sitting (Sentado) | Deriva lenta ~0.1 Hz |
| 4 | Get Up (Levantarse) | Ráfaga transitoria decreciente |
| 5 | Lying (Tumbado) | Señal casi estática, σ muy bajo |
| 6 | No Person (Sin persona) | Referencia estática de sala vacía |

---

## Instalación rápida

```bash
# Clonar e instalar
git clone https://github.com/<usuario>/csi-har-tfg.git
cd csi-har-tfg
pip install -r requirements.txt
pip install pillow          # Opcional — matrices de confusión en GUI

# Lanzar interfaz gráfica (pesos entrenados incluidos)
py gui/app.py              # Windows
python gui/app.py          # Linux/Mac
```

---

## Estructura del proyecto

```
csi-har-tfg/
├── model/
│   ├── models_zoo.py          # 4 modelos HAR + registro BUILD_MODEL
│   ├── transformer_model.py   # CSITransformer (ConvStem + TransformerEncoder)
│   ├── position_estimator.py  # ZoneClassifier MLP + ZoneDataset
│   └── data_loader.py         # Simulación CSI, PCA, Hampel, wavelet, Dataset
├── gui/
│   └── app.py                 # Interfaz 6 pestañas (1760 líneas)
├── tests/                     # 112 tests pytest (112 passed, 0 failed)
│   ├── test_models.py
│   ├── test_data_pipeline.py
│   ├── test_zone_classifier.py
│   └── test_public_loaders.py
├── checkpoints/               # Pesos entrenados + métricas
│   ├── Transformer.pth        # Mejor modelo (F1=0.826)
│   ├── BiLSTM.pth / SimpleLSTM.pth / FCN.pth
│   ├── zone_classifier.pth    # Estimador de zona (acc=98.1%)
│   ├── zone_stats.npz         # Estadísticas de normalización
│   ├── benchmark.json         # Métricas reproducibles
│   └── confusion_*.png        # Matrices de confusión
├── docs/
│   └── MEMORIA.md             # Memoria TFG completa (~50 páginas)
├── SCIENCE.md                 # Documentación científica con ecuaciones
├── train_all.py               # Entrena todos los modelos (~45 min CPU)
├── requirements.txt
└── README.md
```

---

## Pipeline de inferencia

```
Ventana CSI bruta (128 paquetes × 456 subportadoras)
    ↓ Normalización [0,1]  (AMP_MAX = 577.66)
    ↓ PCA: 3 componentes × 4 pares antena = 12 características
    ↓ Concatenación: 456 + 12 = 468 características/timestep
    ↓ Red neuronal: (1, 128, 468) → logits (1, 7)
    ↓ Softmax → probabilidades de actividad
    ↓ ZoneClassifier: 16 estadísticas → zona (0-3)
```

---

## Interfaz gráfica (6 pestañas)

| Pestaña | Contenido |
|---------|-----------|
| **Monitor** | Inferencia tiempo real: actividad, zona, señal CSI 4 antenas, confianza |
| **Analysis** | Comparativa modelos, matrices de confusión, métricas por clase |
| **Signal** | Espectrograma STFT, FFT Doppler, selector de antena |
| **Config** | Todos los parámetros configurables (gui_config.json) |
| **Dataset** | Carga 7 datasets públicos + NPZ/CSV/repo + auto-detección de formato |
| **Help** | Documentación científica integrada: CSI, modelos, limitaciones |

---

## Resultados por clase — CSITransformer

| Actividad | Precisión | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Standing | 0.500 | 0.556 | 0.526 |
| Walking | 0.900 | 0.885 | 0.893 |
| Get Down | 1.000 | 0.981 | 0.991 |
| Sitting | 1.000 | 1.000 | 1.000 |
| Get Up | 1.000 | 1.000 | 1.000 |
| Lying | 0.947 | 0.947 | 0.947 |
| No Person | 0.480 | 0.436 | 0.457 |
| **Macro** | **0.832** | **0.829** | **0.831** |

---

## Tests automatizados

```bash
py -m pytest tests/ -v
# 112 passed, 0 failed
```

| Módulo | Tests | Cubre |
|---|---|---|
| `test_models.py` | 22 | Shapes, gradientes, softmax, registro |
| `test_data_pipeline.py` | 28 | Simulación, PCA, preprocesamiento, dataset |
| `test_zone_classifier.py` | 23 | Features de zona, MLP, ZoneDataset |
| `test_public_loaders.py` | 39 | Loaders públicos: UT-HAR, NTU-Fi, MM-Fi, SignFi, WiAR, ARIL, interpolación |

---

## Reentrenar

```bash
py train_all.py
# Salida: checkpoints/*.pth + benchmark.json + confusion_*.png
# Tiempo: ~45 min en CPU (Intel i5/i7 moderno)
```

## Cargar datos reales

La pestaña **Dataset** incluye un **catálogo de 7 datasets públicos** con carga directa:

| Dataset | Actividades | Hardware | Formato | Ref |
|---------|-------------|----------|---------|-----|
| **UT-HAR** | 7 (fall, walk, run…) | Intel 5300, 90 feat | NPY / CSV | Yousefi 2017 |
| **NTU-Fi-HAR** | 6 (box, circle, fall…) | Intel 5300, 342 feat | MAT | Chen 2022 |
| **MM-Fi** | 27 rehabilitación | Intel AX200, 342 feat | MAT | Yang 2023 |
| **SignFi** | 276 ASL + 20 gestos | Intel 5300, 90 feat | MAT | Ha 2019 |
| **WiAR** | 16 actividades | Intel 5300, 90 feat | CSV | Guo 2019 |
| **ARIL** | 6 actividades | Intel 5300, 90 feat | CSV | Restuccia 2019 |
| **Kovalenko-2021** | 7 (repo nativo) | TP-Link, 456 feat | CSV | Kovalenko 2021 |

Todos los formatos se normalizan mediante `universal_csi_interpolate` a **(128, 456)** antes del preprocesado. El botón **Auto-detect** identifica automáticamente el formato por extensión y estructura de carpetas.

Formatos genéricos también soportados:
- **NPZ:** arrays `data: (N, T, 456)` y `labels: (N,)`
- **CSV:** filas aplanadas + etiqueta como última columna
- **Carpeta repo:** `data.csv` + `label.csv` (formato Kovalenko)

---

## Limitaciones científicas

1. **Datos simulados:** Entrenado con simulación física, no hardware real. Esperar pérdida de F1 en despliegue real.
2. **Estimación de zona:** 98.1% en simulación. Sin calibración de sitio, precisión real ~ ±1-2 zonas.
3. **Dataset pequeño:** 2000 ventanas (25 épocas). El repo fuente usa 9 GB con ventanas de 1024 muestras.

---

## Referencias principales

- Vaswani et al. (2017). *Attention Is All You Need*. NeurIPS.
- Wang et al. (2017). *Time Series Classification from Scratch*. IJCNN.
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.
- Kovalenko et al. (2021). *Wi-Fi CSI Dataset*. IEEE DataPort.
- Yousefi et al. (2017). *A Survey on Behavior Recognition Using WiFi CSI*. IEEE Communications Magazine. (**UT-HAR**)
- Chen et al. (2022). *WiFi CSI Sensing Benchmark*. arXiv. (**NTU-Fi-HAR**)
- Yang et al. (2023). *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset*. NeurIPS D&B. (**MM-Fi**)
- Ha & Kornél (2019). *SignFi: Sign Language Recognition Using WiFi*. ACM IMWUT. (**SignFi**)
- Guo et al. (2019). *WiAR: Device-Free WiFi Human Activity Recognition*. IEEE Access. (**WiAR**)
- Restuccia et al. (2019). *ARIL: CSI-based Indoor Localization and Activity Recognition*. IEEE INFOCOM. (**ARIL**)
- ITU-R P.1238-10 (2019). *Indoor propagation model*.

---

**Documentación completa:** [`docs/MEMORIA.md`](docs/MEMORIA.md) | [`SCIENCE.md`](SCIENCE.md)

*MIT License — libre para uso académico.*
