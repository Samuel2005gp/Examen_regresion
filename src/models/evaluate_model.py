# ===============================
# EVALUACIÓN DEL MODELO (CRISP-DM: Evaluation)
# ===============================

# accuracy_score: calcula el porcentaje de predicciones correctas
# confusion_matrix: crea la matriz de confusión mostrando aciertos y errores
# classification_report: genera un reporte con precision, recall y F1-score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    """
    Evalúa el modelo usando métricas de clasificación.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba (features)
        y_test: Resultados reales
    
    Returns:
        acc (float): Exactitud del modelo
        cm (array): Matriz de confusión
        report (str): Reporte de clasificación completo
    """
    # Predecir resultados sobre los datos de prueba
    # y_pred será un array con valores 0 o 1 según lo predicho
    y_pred = model.predict(X_test)
    
    # Accuracy = (número de predicciones correctas) / (total de predicciones)
    acc = accuracy_score(y_test, y_pred)
    
    # Matriz de confusión: muestra TP, TN, FP, FN
    cm = confusion_matrix(y_test, y_pred)
    
    # Reporte completo: precision, recall, f1-score y support
    report = classification_report(y_test, y_pred)
    
    return acc, cm, report
