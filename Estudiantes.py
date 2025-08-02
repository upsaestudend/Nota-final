import streamlit as st

# Función para calcular la nota final y clasificación
def calcular_nota(asistencia, p1, p2, p3):
    tp = 80.0
    examen_final = 85.0
    bono = 0
    usar_examen = True

    # Regla del bono
    if asistencia > 95:
        bono = round(tp * 0.20, 1)
        tp += bono
    elif asistencia < 80:
        examen_final = 0.0
        usar_examen = False

    # Cálculo de la nota final con ponderaciones
    nota_final = 0.1333 * p1 + 0.1333 * p2 + 0.1333 * p3 + 0.20 * tp + 0.40 * examen_final
    nota_final = round(nota_final, 1)

    # Clasificación
    if nota_final >= 91:
        clasificacion = "Excelente"
    elif nota_final >= 81:
        clasificacion = "Óptimo"
    elif nota_final >= 71:
        clasificacion = "Satisfactorio"
    elif nota_final >= 61:
        clasificacion = "Bueno"
    elif nota_final >= 51:
        clasificacion = "Regular"
    else:
        clasificacion = "Insuficiente"

    return nota_final, clasificacion, bono, usar_examen

# Interfaz de usuario con Streamlit
st.title("📊 Predicción de Nota Final del Estudiante")
st.markdown("Introduce las siguientes calificaciones (entre 0 y 100):")

p1 = st.number_input("Parcial 1", min_value=0.0, max_value=100.0, value=80.0)
p2 = st.number_input("Parcial 2", min_value=0.0, max_value=100.0, value=85.0)
p3 = st.number_input("Parcial 3", min_value=0.0, max_value=100.0, value=90.0)
asistencia = st.number_input("Asistencia (%)", min_value=0.0, max_value=100.0, value=92.0)

if st.button("Calcular Nota Final"):
    nota_final, clasificacion, bono, usar_examen = calcular_nota(asistencia, p1, p2, p3)

    st.success(f"📌 Nota Final: **{nota_final}**")
    st.info(f"🏅 Clasificación: **{clasificacion}**")

    if bono > 0:
        st.write(f"✅ Bono aplicado sobre trabajos prácticos: **+{bono} puntos**")

    if not usar_examen:
        st.warning("⚠️ La asistencia es menor al 80%, no se consideró el examen final.")
