import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv("calificaciones_1000_estudiantes_con_id.csv")

# Calcular bono
df["Bono"] = np.where(df["Asistencia"] > 95, df["TP"] * 0.20, 0)

# TP modificada y Final modificada según asistencia
df["TP_Modificada"] = df["TP"] + df["Bono"]
df["Final_Modificada"] = np.where(df["Asistencia"] < 80, 0, df["Final"])

# Nota final calculada
df["Nota_Final_Calculada"] = (
    0.1333 * df["Parcial_1"] +
    0.1333 * df["Parcial_2"] +
    0.1333 * df["Parcial_3"] +
    0.20 * df["TP_Modificada"] +
    0.40 * df["Final_Modificada"]
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

# Entrenamiento del modelo
X = df[["Parcial_1", "Parcial_2", "Parcial_3", "Asistencia"]]
y = df["Nota_Final_Calculada"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(modelo, "modelo_entrenado.pkl")

# Matriz de confusión
y_pred = modelo.predict(X_test).round(1)
clas_real = [clasificar(n) for n in y_test]
clas_pred = [clasificar(n) for n in y_pred]

etiquetas = ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
cm = confusion_matrix(clas_real, clas_pred, labels=etiquetas)

# Guardar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=etiquetas, yticklabels=etiquetas, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig("matriz_confusion.png")
plt.close()

print("✅ Modelo entrenado y guardado como modelo_entrenado.pkl")

