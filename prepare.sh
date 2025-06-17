#!/bin/bash

# ------------------------------------------------------------- #
#   Setup per il progetto Skin Lesion Analysis Classification   #
# ------------------------------------------------------------- #

echo "Creazione ambiente per il progetto..."

# 1. Installazione dipendenze
echo "Installazione dei pacchetti da requirements.txt..."
pip install -r requirements.txt

# 2. Creazione delle directory necessarie
echo "Creazione directory..."

mkdir -p save_model_embeddings/
mkdir -p save_model_autoencoder/
mkdir -p save_model_classifier/
mkdir -p reconstructions/
mkdir -p images/
mkdir -p images/latent_space/

echo "Setup completato con successo!"

# 3. Messaggio finale
echo ""
echo "Se hai già il dataset, assicurati che sia posizionato correttamente nella struttura prevista."
echo "Dataset usato: ISIC Archive 2018"
echo "Link originale: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#"
