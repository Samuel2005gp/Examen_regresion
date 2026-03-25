# ===============================
# API CON FASTAPI (Deployment)
# ===============================

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.predict_model import predict

# Crear la aplicación FastAPI
app = FastAPI(
    title="Examen Prediction API",
    description="API para predecir si un estudiante aprobará según horas de estudio",
    version="1.0"
)

# Definir el modelo de datos de entrada
class EstudianteInput(BaseModel):
    horas_estudio: float

@app.get("/")
def root():
    """Endpoint raíz - verifica que la API está activa"""
    return {"message": "API de predicción activa"}

@app.post("/predict")
def predict_exam(data: EstudianteInput):
    """
    Predice si el estudiante aprobará o no.
    
    - **horas_estudio**: Número de horas que estudió el estudiante
    """
    resultado, probabilidad = predict(data.horas_estudio)
    return {
        "horas_estudio": data.horas_estudio,
        "aprobara": bool(resultado),
        "probabilidad": round(probabilidad, 4)
    }
