# src/preprocessing.py


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------- #
# Définition des listes de variables #
# ---------------------------------- #

# Définition de la liste de Variables numériques
VARIABLES_NUMERIQUES = [
    "age",
    "anciennete_poste_ans"
]

# Définition de la liste de Variables catégorielles
VARIABLES_CATEGORIELLES = [
    "niveau_diplome",
    "famille_rome",
    "departement",
    "est_allocataire",
    "nationalite_hors_ue"
]

# Définition de la liste de Variables catégorielles sans la donnée sensible 'nationalite_hors_ue'
VARIABLES_CATEGORIELLES_SANS_SENSIBLE = [
    "niveau_diplome",
    "famille_rome",
    "departement",
    "est_allocataire"
]

# Définition de la variable texte
VARIABLE_TEXTE = "synthese_entretien"

# Définition de la variable cible
CIBLE = "classe_retour_emploi"


# ------------------ #
# Pipeline numérique #
# ------------------ #

pipeline_numerique = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------- #
# Pipeline catégoriel #
# ------------------- #

pipeline_categoriel = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)



# ---------------------------- #
# Construction du preprocessor #
# ---------------------------- #

def construire_preprocesseur(
    variables_numeriques=VARIABLES_NUMERIQUES,
    variables_categorielles=VARIABLES_CATEGORIELLES,
    variable_texte=VARIABLE_TEXTE
):

    transformers = []

    if variables_numeriques:
        transformers.append(
            (
                "num",
                pipeline_numerique,
                variables_numeriques
            )
        )

    if variables_categorielles:
        transformers.append(
            (
                "cat",
                pipeline_categoriel,
                variables_categorielles
            )
        )

    if variable_texte:
        transformers.append(
            (
                "txt",
                TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2)
                ),
                variable_texte
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )
    


    