# ===============================
# DATA UNDERSTANDING (CRISP-DM)
# ===============================

# Importamos librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Cargar el dataset
df = pd.read_csv("data/estudiantes_examen.csv")

# Mostrar las primeras filas
# Esto permite ver cómo vienen los datos (estructura y valores)
print("=" * 40)
print("PRIMERAS FILAS DEL DATASET")
print("=" * 40)
print(df.head(10))

# ===============================
# DESCRIPCIÓN GENERAL DE LOS DATOS
# ===============================

# describe() muestra estadísticas básicas:
# - count: cantidad de datos
# - mean: promedio
# - std: desviación estándar
# - min/max: valores extremos
# - percentiles: distribución
print("\n" + "=" * 40)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 40)
print(df.describe())

# ===============================
# DISTRIBUCIÓN DE HORAS DE ESTUDIO
# ===============================

# Histograma: muestra cuántos estudiantes hay en cada rango de horas
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df['HorasEstudio'].hist(bins=10, color='steelblue', edgecolor='white')
plt.title("Distribución de Horas de Estudio")
plt.xlabel("Horas de Estudio")
plt.ylabel("Cantidad de Estudiantes")

# ===============================
# ANÁLISIS POR RESULTADO
# ===============================

# Agrupar por Resultado (Si / No)
# count() cuenta cuántos registros hay por cada grupo
resultado_count = df.groupby('Resultado').count()

print("\n" + "=" * 40)
print("CONTEO POR RESULTADO")
print("=" * 40)
print(resultado_count)

plt.subplot(1, 2, 2)
resultado_count['HorasEstudio'].plot(kind='bar', color=['tomato', 'mediumseagreen'], edgecolor='white')
plt.title("Estudiantes por Resultado")
plt.xlabel("Resultado")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("notebooks/analisis_exploratorio.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfico guardado en notebooks/analisis_exploratorio.png")
