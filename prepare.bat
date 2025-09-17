@echo off
REM -------------------------------------------------------------
REM   Setup per il progetto Skin Lesion Analysis Classification
REM -------------------------------------------------------------

echo Creazione ambiente per il progetto...

REM 1. Installazione dipendenze
echo Installazione dei pacchetti da requirements.txt...
pip install -r requirements.txt

REM 2. Creazione delle directory necessarie
echo Creazione directory...

if not exist save_model_embeddings mkdir save_model_embeddings
if not exist save_model_autoencoder mkdir save_model_autoencoder
if not exist save_model_classifier mkdir save_model_classifier
if not exist reconstructions mkdir reconstructions
if not exist images mkdir images
if not exist images\latent_space mkdir images\latent_space

echo Setup completato con successo!

REM 3. Messaggio finale
echo.
echo Se hai già il dataset, assicurati che sia posizionato correttamente nella struttura prevista.
echo Dataset usato: ISIC Archive 2018
echo Link originale: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#

pause
