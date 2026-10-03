from pathlib import Path
import os

from fastapi import FastAPI
import pandas as pd
import joblib

# Chargement du modèle
modele = joblib.load(
    "./models/modele_retour_emploi.joblib"
)

app = FastAPI(
    title="API Retour Emploi"
)


@app.get("/health")
def accueil():

    return {
        "message":
        "API de prédiction du retour à l'emploi"
    }


@app.post("/predict")
def predict(donnees: dict):

    df = pd.DataFrame(
        [donnees]
    )

    # Ajout des 2 colonnes supplémentaires département + famille_rome
    df["departement"] = (
    df["code_insee_commune"].str[:2]
    )
    
    df["famille_rome"] = (
    df["code_rome_vise"].str[0]
    )


    
    prediction = modele.predict(df)

    return {
        "classe_predite":
        int(prediction[0])
    }