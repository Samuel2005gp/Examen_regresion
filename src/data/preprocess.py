# ===============================
# PREPROCESAMIENTO (CRISP-DM: Data Preparation)
# ===============================

def prepare_data(df):
    """
    Separa features (X) y variable objetivo (y).
    Convierte 'Si'/'No' a 1/0.
    
    Args:
        df (pd.DataFrame): Dataset original
    
    Returns:
        X (pd.DataFrame): Features de entrada
        y (pd.Series): Variable objetivo en formato numérico
    """
    X = df[["HorasEstudio"]]
    y = df["Resultado"].map({"No": 0, "Si": 1})  # Convertir a 0/1
    return X, y
