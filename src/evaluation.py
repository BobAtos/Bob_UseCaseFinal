# src/evaluation.py

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)
from src.preprocessing import construire_preprocesseur


# ------------------------------------------- #
# Fonction pour compter les erreurs critiques #
# ------------------------------------------- #

def compter_erreurs_critiques(
    y_reel,
    y_predit
):
    """
    Compte les erreurs :
    classe réelle 2 -> classe prédite 0
    """

    matrice = confusion_matrix(
        y_reel,
        y_predit,
        labels=[0, 1, 2]
    )
    
    # Dans la matrice de confusion sklearn : les lignes correspondent aux classes réelles & les colonnes correspondent aux classes prédites.
    return matrice[2, 0]


# ----------------------- #
# Evaluation des scenarii #
# ----------------------- #

def evaluer_scenario(
    nom_scenario,
    X_train,
    X_test,
    y_train,
    y_test,
    variables_numeriques,
    variables_categorielles,
    variable_texte
):

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                construire_preprocesseur(
                    variables_numeriques,
                    variables_categorielles,
                    variable_texte
                )
            ),
            (
                "modele",
                LinearSVC(
                    C=0.1,
                    class_weight={
                        0: 1,
                        1: 1,
                        2: 2
                    },
                    random_state=42
                )
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    return {
        "scenario": nom_scenario,
        "accuracy": round(
            accuracy_score(y_test, y_pred),
            3
        ),
        "f1_macro": round(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            ),
            3
        ),
        "f1_weighted": round(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            3
        ),
        "recall_classe_2": round(
            recall_score(
                y_test,
                y_pred,
                labels=[2],
                average=None,
                zero_division=0
            )[0],
            3
        ),
        "erreurs_critiques": compter_erreurs_critiques(
            y_test,
            y_pred
        )
    }