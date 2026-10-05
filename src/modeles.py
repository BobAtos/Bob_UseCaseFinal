# src/modeles.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def construire_modeles():

    modeles = {
        "Regression logistique": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "Linear SVC": LinearSVC(
            C=0.1,
            random_state=42
        )
    }

    return modeles
