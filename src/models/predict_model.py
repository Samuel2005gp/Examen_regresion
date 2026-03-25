# ===============================
# PREDICCIÓN CON EL MODELO
# ===============================

import joblib
import numpy as np
import pandas as pd
import os

# Ruta absoluta al modelo, funciona tanto local como en Render
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_logistico.pkl")

# Cargar el modelo guardado
model = joblib.load(MODEL_PATH)

def predict(horas_estudio):
    """
    Predice si un estudiante aprobará según las horas de estudio.
    
    Args:
        horas_estudio (float): Número de horas estudiadas
    
    Returns:
        pred (int): 1 = Aprueba, 0 = No aprueba
        prob (float): Probabilidad de aprobar (entre 0 y 1)
    """
    # Usamos DataFrame con nombre de columna para evitar warnings
    data = pd.DataFrame([[horas_estudio]], columns=["HorasEstudio"])
    prob = model.predict_proba(data)[0][1]  # Probabilidad de aprobar
    pred = model.predict(data)[0]           # 0 o 1
    return int(pred), float(prob)
