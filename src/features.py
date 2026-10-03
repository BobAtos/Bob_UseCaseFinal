# src/features.py

def extraire_departement(code):

    code = str(code)

    # Attention au cas particulier de la Corse
    if code.startswith("2A"):
        return "2A"

    if code.startswith("2B"):
        return "2B"

    return code[:2]


def creer_features(df):

    df = df.copy()

    # Nettoyage de la variable texte
    df["synthese_entretien"] = (
        df["synthese_entretien"]
        .fillna("")
        .astype(str)
    )

    df["departement"] = (
        df["code_insee_commune"]
        .apply(extraire_departement)
    )

    df["famille_rome"] = (
        df["code_rome_vise"]
        .str[0]
    )

    return df