# ===============================
# PREDICCIÓN CON EL MODELO
# ===============================

import joblib
import numpy as np

# Cargar el modelo guardado
model = joblib.load("models/modelo_logistico.pkl")

def predict(horas_estudio):
    """
    Predice si un estudiante aprobará según las horas de estudio.
    
    Args:
        horas_estudio (float): Número de horas estudiadas
    
    Returns:
        pred (int): 1 = Aprueba, 0 = No aprueba
        prob (float): Probabilidad de aprobar (entre 0 y 1)
    """
    data = np.array([[horas_estudio]])
    prob = model.predict_proba(data)[0][1]  # Probabilidad de aprobar
    pred = model.predict(data)[0]           # 0 o 1
    return int(pred), float(prob)
