import streamlit as st
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load the model once ---
@st.cache_resource
def load_model():
    with open('C:/Users/HP/Desktop/py/rf1.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
app_mode = st.selectbox("Choisissez une option", ["Home", "Prediction", "Analyse_et_visualisation"])

if app_mode == 'Home':
    st.title('🧪 Santé Étudiants')    
    st.image("Capture d'écran 2025-05-08 144835.png", caption='Loan Dataset',width=700)

    st.markdown('Dataset:')
    # Charger et afficher le dataset
    data = pd.read_csv('medical_students_dataset.csv')
    data = data.drop(columns=['Student ID', 'Height', 'Blood Type', 'Blood Pressure'])
    # 🧹 Nettoyage des données - Suppression des valeurs manquantes
    df_clean = data.dropna()  # Suppression des lignes avec des valeurs manquantes
    st.subheader("Données de dataset:")
    st.dataframe(df_clean.head())  # عرض البيانات بعد التنظيف
    st.subheader("Les colones principaux de dataset:")
    st.dataframe(df_clean.columns)

elif app_mode == "Analyse_et_visualisation":
    st.title("📈 Lancer l'Analyse et visualisation")
    st.image("Capture d'écran 2025-05-08 151245.png", caption='Loan Dataset',width=500)
    
    data = pd.read_csv("medical_students_dataset.csv")
    data = data.drop(columns=['Student ID', 'Height', 'Blood Type', 'Blood Pressure'])
    st.title("Affichage De Data")
    df_clean = data.dropna()  # Suppression des lignes avec des valeurs manquantes
    st.subheader("Données après suppression des valeurs manquantes:")
    st.dataframe(df_clean.head())  # عرض البيانات بعد التنظيف

    st.subheader("📊 Histogramme d'une colonne")
    column = st.selectbox("Choisir une colonne", data.columns)
    fig1, ax1 = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔹 Moyenne de BMI, Temperature par tabagisme")
    st.dataframe(data.groupby('Smoking')[[ 'BMI', 'Temperature']].mean())

    # 🔹 Gender vs Fumeur
    st.subheader("🔹 Tableau croisé Sexe - Fumeur")
    st.dataframe(pd.crosstab(data['Gender'], data['Smoking']))

    # 📈 Scatter Plot
    st.subheader("🔹Scatter Plot")
    x = st.selectbox("X-axis", data.columns)
    y = st.selectbox("Y-axis", data.columns)
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=data, x=x, y=y, ax=ax2)
    st.pyplot(fig2)

    # Calcul de l'âge moyen par genre
    st.subheader("📊 Analyse de l'âge moyen selon le genre")
    catg_age = data.groupby('Gender')['Age'].mean()
    sizes = catg_age.values
    labels = catg_age.index
    # Création du graphe
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title("Âge moyen des étudiants par genre")
    # Affichage dans Streamlit
    st.pyplot(fig)

    # 📊 Bar Chart
    st.subheader("📊Bar Chart d'une variable catégorique")
    cat_col = st.selectbox("Colonne catégorique", data.select_dtypes(include='object').columns)
    fig3, ax3 = plt.subplots()
    data[cat_col].value_counts().plot(kind='bar', ax=ax3)
    st.pyplot(fig3)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Gender", data=data, palette="Set2", ax=ax)
    ax.set_title("Répartition des genres")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Nombre d'étudiants")
    st.pyplot(fig)

    # Créer la table croisée
    cross_tab = pd.crosstab(data["Smoking"], data["Diabetes"])

    # Créer la figure
    fig, ax = plt.subplots()
    sns.heatmap(cross_tab, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
    ax.set_title("Relation entre smoking et diabète")
    ax.set_xlabel("Diabète")
    ax.set_ylabel("Fumeur (0 = Non, 1 = Oui)")


    st.subheader("📊Répartition des fumeurs selon le genre")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=data, x="Gender", hue="Smoking", ax=ax4)
    ax4.set_title("Fumeurs selon le genre")
    st.pyplot(fig4)

elif app_mode == "Prediction":

    st.title("🧠 Lancer la Prédiction")
    st.image("Capture d'écran 2025-05-08 150301.png", caption='Loan Dataset',width=500)
    # Entrée de l'utilisateur
    BMI = st.number_input("BMI", min_value=10.18, max_value=44.29, step=0.01)
    Temperature = st.number_input("Temperature", min_value=96.755, max_value=100.45, step=0.01)
    Heart_Rate = st.number_input("Heart Rate", min_value=60.0, max_value=99.0, step=0.1)
    Cholesterol = st.number_input("Cholesterol", min_value=120.0, max_value=249.0, step=0.1)
    Age = st.number_input("Âge de la personne", min_value=18, max_value=80)
    Weight = st.number_input("Poids (en kg)", step=0.1, format="%.2f")
    Temperature = st.number_input("Température (en °F)", step=0.1, format="%.2f")
    Gender = st.selectbox('Sexe', ['Male', 'Female'])

    # --- Encodage des données ---
    gender_binary = 1 if Gender == 'Male' else 0

    # Entrée pour le tabagisme
    Smoking = st.radio("Smoking", ('Non', 'Oui'))

    # Conversion de "Oui" et "Non" en 1 et 0
    Smoking = 1 if Smoking == 'Oui' else 0

    input_data = np.array([[ 
        Age, Weight, BMI, Cholesterol, Temperature, Heart_Rate, gender_binary, Smoking
    ]]).reshape(1, -1)

    # Bouton de prédiction
    if st.button("Prédire"):
        try:
            prediction = model.predict(input_data)

            # Affichage de la prédiction
            if prediction[0] == 1:
                st.error("❌ L'étudiant est malade (diabétique).")
            else:
                st.success("✅ L'étudiant n'est pas malade (pas diabétique).")

        except Exception as e:
            st.error(f"Erreur: {str(e)}")