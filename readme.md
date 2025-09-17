# 🩸 Skin Lesion Classification

Questo progetto affronta la classificazione di immagini dermatoscopiche di lesioni cutanee mediante due approcci distinti:  
1. Una pipeline **Deep Learning** basata su Autoencoder Convoluzionale.
2. Una pipeline **XGBoost** che utilizza feature estratte da un **Vision Transformer (ViT)** pre-addestrato.

Il tracciamento degli esperimenti è gestito tramite [Comet ML](https://www.comet.com/). Il progetto è sviluppato in PyTorch e scikit-learn.

| | |
| --- | --- |
| **Progetto** | Skin lesion classification using Deep Learning and XGBoost |
| **Autore** | Giovanni Giuseppe Iacuzzo |
| **Corso** | [Machine Learning](https://unikore.it) |

---

## 📌 Indice

- [Introduzione](#introduzione)
- [Requisiti](#requisiti)
- [Struttura del Codice](#struttura-del-Codice)
- [Utilizzo](#utilizzo)
- [Confronto tra i modelli](#confronto-tra-i-modelli)

---

## 🧠 Introduzione

Il progetto ha l'obiettivo di classificare le lesioni cutanee in base alle immagini. Sono stati implementati e confrontati due approcci:

### 1. Pipeline Deep Learning
- Addestramento di un **Autoencoder Convoluzionale** per l'estrazione non supervisionata di embedding.
- Un **Classificatore Feedforward** viene addestrato sugli embedding per la classificazione.

### 2. Pipeline XGBoost
- Estrazione delle feature dalle immagini usando un **Vision Transformer (ViT)** pre-addestrato.
- Le feature vengono utilizzate per addestrare un **classificatore XGBoost**, con ottimizzazione iperparametrica tramite **GridSearchCV**.

Entrambi gli approcci sono valutati su dataset suddiviso in modo stratificato (train, validation, test).

---

## 📦 Requisiti

Il progetto richiede **Python 3.11+**. Le principali librerie utilizzate includono:

- `torch`, `torchvision` — per la rete neurale
- `transformers` — per l’estrazione feature con ViT
- `scikit-learn`, `xgboost` — per classificazione classica ed evaluation
- `matplotlib`, `seaborn` — per la visualizzazione
- `comet_ml` — per il tracciamento degli esperimenti
- `PyYAML`, `tqdm`, `joblib`, `PIL` — utilità varie

Per installare i requisiti:

```bash
pip install -r requirements.txt
```
---

## 📁  Struttura del Codice

```bash
Esame-Machine-Learning/
│
├── dataset/
│   └── dataset.py                  # Caricamento e suddivisione del dataset
│
├── models/
│   ├── models_autoencoder.py       # Architettura autoencoder
│   └── model_classifier.py         # Classificatore feedforward
│
├── utils/
│   ├── utils.py                    # Utility e visualizzazioni
│   ├── loss.py                     # Loss per il modello di autoencoder
│   ├── train_autoencoder.py        # Addestramento autoencoder
│   ├── train_classifier.py         # Addestramento classificatore
│   ├── feature_extraction.py       # Estrazione embedding dall’autoencoder
│   └── test.py                     # Script di valutazione
│
├── vision_embeddings.py            # Estrazione embedding da ViT
├── extract_features.py             # Estrazione feature da immagini con ViT
│
├── main_neural_net.py              # Pipeline completa per DL
├── main_XGB_classifier.py          # Pipeline XGBoost
│
├── save_model_autoencoder/         # Cartella Modello autoencoder
├── save_model_embeddings/          # Cartella Modello embeddings
├── save_model_classifier/          # Cartella Modello classifier
├── reconstructions/                # Cartella delle ricostruzioni
├── images/                         # Cartella dove salvare immagini varie
│   ├── images/latent_space/        # Cartella rappresentazioni latenti tra le epoche    
│
├── config.yml                      # Configurazione parametri
├── requirements.txt                # Librerie da scaricare
├── .comet.config                   # Config per Comet ML
│
└── README.md
```
---

## ⚙️ Utilizzo
ricordare di eseguire prima il file di preparazione.

- Mac:
```bash
prepare.sh
```

- Windows:
```bash
prepare.bat
```

Per eseguire l'intera pipeline basata su rete neurale (autoencoder + classificatore):

```bash
python main_neural_net.py
```

Esecuzione pipeline XGBoost

- Estrazione feature da immagini con Vision Transformer:
```bash
python extract_features.py
```

- Addestramento e valutazione XGBoost:
```bash
python main_XGB_classifier.py
```
---
## 🔄 Confronto tra i Modelli

Per valutare l'efficacia di diversi approcci nella classificazione delle lesioni cutanee, sono stati implementati e confrontati due modelli distinti:

---

### 📘 Approccio 1 — Autoencoder + Classificatore Neurale

Questo approccio utilizza un **Autoencoder Convoluzionale** per apprendere rappresentazioni latenti (embedding) delle immagini, seguito da un **classificatore fully-connected** addestrato su tali embedding.

- **Caratteristiche**:
  - L’autoencoder è addestrato in modo non supervisionato sui dati del dataset, il che permette di ottenere feature apprese direttamente dalle immagini delle lesioni.
  - Il classificatore opera nello spazio latente, cercando di distinguere le classi a partire da rappresentazioni compresse ma informative.

- **Obiettivo**:
  - Sfruttare la capacità dell’autoencoder di estrarre feature significative legate al dominio medico, con una pipeline interamente progettata e addestrata ad hoc.

---

### 🤖 Approccio 2 — Vision Transformer + XGBoost

Il secondo modello sfrutta **ViT (Vision Transformer)** pre-addestrato su ImageNet per l’estrazione di embedding visivi, seguiti da un classificatore **XGBoost**, ottimizzato tramite (`GridSearchCV`).

- **Caratteristiche**:
  - Il Vision Transformer fornisce rappresentazioni ad alto livello già apprese da un modello generalista e potente.
  - XGBoost, noto per la sua robustezza e interpretabilità, viene addestrato sulle feature estratte per eseguire la classificazione.

- **Obiettivo**:
  - Valutare la performance di un approccio ibrido che combina modelli pre-addestrati di visione artificiale con tecniche di apprendimento supervisionato classiche.

---

### 📊 Riepilogo

| Caratteristica                 | Autoencoder + NN                     | ViT + XGBoost                             |
|-------------------------------|--------------------------------------|-------------------------------------------|
| Estrazione delle feature      | Autoencoder convoluzionale           | Vision Transformer (`google/vit-base`)    |
| Classificatore                | Rete neurale fully-connected         | XGBoost (con GridSearchCV)                |
| Tipo di feature               | Apprese dai dati del dataset         | Pre-addestrate su ImageNet                |
| Complessità di addestramento | Medio-alta                           | Bassa (solo XGBoost viene addestrato)     |
| Flessibilità                  | Alta: pipeline personalizzabile      | Media: feature extractor fisso            |
| Interpretabilità              | Limitata                             | Alta (importanza delle feature)           |

---

## 📎 Licenza

Questo progetto è distribuito sotto licenza 
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](https://creativecommons.org/licenses/by-nc-nd/4.0/).

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

### 📩 Contatti
Per domande: [giovanni.iacuzzo@unikorestudent.it](mailto:giovanni.iacuzzo@unikorestudent.it)
