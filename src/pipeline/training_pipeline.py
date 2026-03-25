# ===============================
# PIPELINE DE ENTRENAMIENTO COMPLETO
# Un pipeline es una secuencia automatizada de procesos donde la
# salida de cada etapa se convierte en la entrada de la siguiente.
# ===============================

import joblib
import sys
import os

# Agregar el directorio raíz al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.load_data import load_data
from src.data.preprocess import prepare_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate

def run_training():
    """
    Ejecuta el pipeline completo:
    1. Carga datos
    2. Preprocesa
    3. Entrena modelo
    4. Evalúa
    5. Guarda modelo
    """
    print("=" * 40)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO")
    print("=" * 40)
    
    # 1. Cargar datos
    print("\n[1/5] Cargando datos...")
    df = load_data("data/estudiantes_examen.csv")
    print(f"     Registros cargados: {len(df)}")
    
    # 2. Preprocesar
    print("\n[2/5] Preprocesando datos...")
    X, y = prepare_data(df)
    print(f"     Features: {list(X.columns)}")
    print(f"     Distribución: {y.value_counts().to_dict()}")
    
    # 3. Entrenar modelo
    print("\n[3/5] Entrenando modelo de Regresión Logística...")
    model, X_test, y_test = train_model(X, y)
    print("     Modelo entrenado correctamente")
    
    # 4. Evaluar
    print("\n[4/5] Evaluando modelo...")
    acc, cm, report = evaluate(model, X_test, y_test)
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  Matriz de Confusión:\n{cm}")
    print(f"\n  Reporte de Clasificación:\n{report}")
    
    # 5. Guardar modelo
    print("\n[5/5] Guardando modelo...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/modelo_logistico.pkl")
    print("     Modelo guardado en: models/modelo_logistico.pkl")
    
    print("\n" + "=" * 40)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 40)

if __name__ == "__main__":
    run_training()

# Ejecutar con:
# python -m src.pipeline.training_pipeline
