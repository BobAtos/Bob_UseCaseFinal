# tests/test_api.py

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


#-------------------------------------------------#
# 1er test : bon fonctionnement de l'API (health) #
# ------------------------------------------------#
def test_health():

    response = client.get("/health")

    assert response.status_code == 200


#---------------------------------------------------------------------------#
# 2e test : préediction d'un cas favorable à un retour à l'emploi (predict) #
# Réponse attendue = classe 0                                               #
#---------------------------------------------------------------------------#
def test_predict0():

    payload = {
        "age": 28,
        "niveau_diplome": "Bac+5",
        "anciennete_poste_ans": 5.0,
        "code_rome_vise": "M1805",
        "code_insee_commune": "75056",
        "est_allocataire": 0,
        "nationalite_hors_ue": 0,
        "synthese_entretien": "Excellente présentation, compétences techniques à jour."
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200



#-----------------------------------------------------------------------------#
# 3e test : préediction d'un cas fraigile pour un retour à l'emploi (predict) #
# Réponse attendue = classe 2                                                 #
#-----------------------------------------------------------------------------#
def test_predict2():

    payload = {
        "age": 56,
        "niveau_diplome": "Sans diplôme",
        "anciennete_poste_ans": 0.2,
        "code_rome_vise": "N4301",
        "code_insee_commune": "93066",
        "est_allocataire": 1,
        "nationalite_hors_ue": 1,
        "synthese_entretien": "Freins périphériques majeurs. Situation d'illettrisme numérique constatée."
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200