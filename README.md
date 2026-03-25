# ML API – Regresión Logística (Aprueba / No Aprueba)

## Descripción
Pipeline completo de Machine Learning para predecir si un estudiante aprobará un examen según sus horas de estudio. Implementa la metodología **CRISP-DM** completa.

## Estructura del Proyecto
```
ml-api-reglogistica-examen/
├── data/
│   └── estudiantes_examen.csv
├── models/
│   └── modelo_logistico.pkl
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_model.py
│   │   ├── evaluate_model.py
│   │   └── predict_model.py
│   └── pipeline/
│       └── training_pipeline.py
├── api/
│   └── main.py
├── notebooks/
│   └── analisis_exploratorio.py
├── requirements.txt
├── render.yaml
└── README.md
```

## Instalación
```bash
pip install -r requirements.txt
```

## Uso

### 1. Exploración de datos (EDA)
```bash
python notebooks/analisis_exploratorio.py
```

### 2. Entrenar el modelo
```bash
python -m src.pipeline.training_pipeline
```

### 3. Ejecutar la API
```bash
uvicorn api.main:app --reload
```

### 4. Documentación automática
Abrir: http://localhost:8000/docs

## Fases CRISP-DM
1. **Business Understanding**: Predecir aprobación según horas de estudio
2. **Data Understanding**: EDA con histogramas y estadísticas
3. **Data Preparation**: Codificación Si/No → 1/0
4. **Modeling**: LogisticRegression de scikit-learn
5. **Evaluation**: Accuracy, Matriz de Confusión, F1-Score
6. **Deployment**: FastAPI en Render (plan gratuito)

## Despliegue en Render
1. Subir proyecto a GitHub
2. Crear cuenta en [render.com](https://render.com) con GitHub
3. Crear nuevo Web Service y seleccionar el repo
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port 10000`
5. Plan: Free → Deploy
