import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Predicción de Notas", layout="centered")

st.title("🎓 Predicción de la Nota Final del Estudiante")

# Cargar el modelo y dataset
modelo = joblib.load("modelo_entrenado.pkl")
df = pd.read_csv("calificaciones_1000_estudiantes_con_id.csv")

# Calcular promedio de TP y Final para simular en predicción
tp_promedio = df["TP"].mean()
final_promedio = df["Final"].mean()

# Clasificación por nota
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

# Formulario de entrada
st.sidebar.header("🧾 Ingrese las calificaciones")

parcial_1 = st.sidebar.number_input("Parcial 1", min_value=0.0, max_value=100.0, step=1.0)
parcial_2 = st.sidebar.number_input("Parcial 2", min_value=0.0, max_value=100.0, step=1.0)
parcial_3 = st.sidebar.number_input("Parcial 3", min_value=0.0, max_value=100.0, step=1.0)

asistencia = st.sidebar.selectbox("Asistencia (%)", options=[round(x, 1) for x in np.linspace(50, 100, 101)])

# Botón para predecir
if st.sidebar.button("🔍 Predecir"):

    # Calcular bono y final permitido
    bono = tp_promedio * 0.20 if asistencia > 95 else 0
    tp_modificado = tp_promedio + bono
    final_usable = 0 if asistencia < 80 else final_promedio

    # Calcular nota final manualmente con misma ponderación
    nota_final = (
        0.1333 * parcial_1 +
        0.1333 * parcial_2 +
        0.1333 * parcial_3 +
        0.20 * tp_modificado +
        0.40 * final_usable
    )
    nota_final = round(nota_final, 1)
    clasificacion = clasificar(nota_final)

    # Mostrar resultados
    st.subheader("📈 Resultado de la Predicción")
    st.write(f"🧪 **Trabajos Prácticos (TP):** {tp_modificado:.1f} puntos")
    st.write(f"📝 **Examen Final:** {final_usable:.1f} puntos")
    st.write(f"📊 **Nota Final Estimada:** {nota_final:.1f}")
    st.write(f"🏅 **Clasificación:** {clasificacion}")

    # Gráficos adicionales
    st.subheader("📊 Estadísticas del Dataset")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Distribución de Clasificaciones:")
        clas_counts = df["Clasificacion"].value_counts().reindex(
            ["Excelente", "Óptimo", "Satisfactorio", "Bueno", "Regular", "Insuficiente"]
        ).fillna(0)
        fig1, ax1 = plt.subplots()
        clas_counts.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Cantidad de estudiantes")
        ax1.set_title("Clasificaciones")
        st.pyplot(fig1)

    with col2:
        st.write("Distribución de Notas Finales:")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["Nota_Final_Calculada"], bins=20, kde=True, ax=ax2)
        ax2.set_title("Histograma de Notas Finales")
        st.pyplot(fig2)

    st.subheader("🧮 Matriz de Confusión del Modelo")
    st.image("matriz_confusion.png", caption="Comparación real vs. predicho")

