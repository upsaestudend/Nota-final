import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Cargar dataset
df = pd.read_csv("calificaciones_1000_estudiantes_con_id.csv")

# Aplicar bono por asistencia > 95%
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.20, 0)
df["TP_Modificado"] = df["TP"] + df["Bono"]

# Anular examen si asistencia < 80%
df["Final_Usado"] = np.where(df["Asistencia"] < 80, 0, df["Final"])

# Calcular nota final
df["Nota_Final_Calculada"] = (
    0.1333 * df["Parcial_1"] + 
    0.1333 * df["Parcial_2"] +
    0.1333 * df["Parcial_3"] +
    0.20 * df["TP_Modificado"] +
    0.40 * df["Final_Usado"]
).round(1)

# Clasificación
def clasificar(nota):
    if nota >= 91:
        return "Excelente"
    elif nota >= 81:
        return "Óptimo"
    elif nota >= 71:
        return "Satisfactorio"
    elif nota >= 61:
        return "Bueno"
    elif nota >= 51:
        return "Regular"
    else:
        return "Insuficiente"

df["Clasificacion"] = df["Nota_Final_Calculada"].apply(clasificar)

# Variables predictoras y objetivo
X = df[["Parcial_1", "Parcial_2", "Parcial_3", "Asistencia"]]
y = df["Nota_Final_Calculada"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predicción
y_pred = modelo.predict(X_test)

# Evaluación
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Aquí está la corrección

# Clasificación para matriz de confusión
def clasificacion(nota):
    if nota >= 91:
        return "Excelente"
    elif nota >= 81:
        return "Óptimo"
    elif nota >= 71:
        return "Satisfactorio"
    elif nota >= 61:
        return "Bueno"
    elif nota >= 51:
        return "Regular"
    else:
        return "Insuficiente"

y_test_clas = y_test.apply(clasificacion)
y_pred_clas = pd.Series(y_pred).apply(clasificacion)

# Matriz de confusión
etiquetas = ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
cm = confusion_matrix(y_test_clas, y_pred_clas, labels=etiquetas)

# Graficar y guardar matriz de confusión
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=etiquetas, yticklabels=etiquetas, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Clasificación de Notas")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.close()

# Guardar modelo
joblib.dump(modelo, "modelo_entrenado.pkl")
print("✅ Modelo entrenado y guardado como modelo_entrenado.pkl")