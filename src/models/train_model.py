# ===============================
# ENTRENAMIENTO DEL MODELO (CRISP-DM: Modeling)
# ===============================

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    """
    Divide los datos y entrena el modelo de Regresión Logística.
    
    Args:
        X (pd.DataFrame): Features de entrada
        y (pd.Series): Variable objetivo
    
    Returns:
        model: Modelo entrenado
        X_test: Datos de prueba (features)
        y_test: Datos de prueba (objetivo real)
    """
    # Dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Crear y entrenar el modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test
