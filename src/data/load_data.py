# ===============================
# CARGA DE DATOS (CRISP-DM: Data Understanding)
# ===============================

import pandas as pd

def load_data(path):
    """
    Carga el archivo CSV desde la ruta especificada.
    
    Args:
        path (str): Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: Dataset cargado
    """
    df = pd.read_csv(path)
    return df
