# ===============================
# INGENIERÍA DE FEATURES
# ===============================

def build_features(df):
    """
    Aplica transformaciones adicionales al dataset si se necesitan.
    En este caso simple, devuelve el mismo dataframe.
    
    Args:
        df (pd.DataFrame): Dataset original
    
    Returns:
        pd.DataFrame: Dataset con features procesadas
    """
    # Por ahora usamos HorasEstudio directamente
    # Aquí se podrían agregar: normalización, nuevas columnas, etc.
    return df
