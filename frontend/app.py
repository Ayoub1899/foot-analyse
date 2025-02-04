import streamlit as st
import requests
import tempfile
import os

API_URL = "http://localhost:8000/upload/"  # URL de l'API FastAPI
DOWNLOAD_BASE_URL = "http://localhost:8000/download/"  # URL de base pour télécharger la vidéo annotée

st.title("Détection de joueurs, d'arbitres et du ballon en temps réel ⚽")

# Ajout d'une section tutoriel
st.sidebar.title("📘 Guide d'utilisation")
st.sidebar.markdown("""
**Comment utiliser cette application :**

1. **Téléversez une vidéo** :
   - Cliquez sur le bouton "Browse files".
   - Sélectionnez une vidéo au format MP4, AVI, MOV ou MKV.

2. **Démarrez la détection** :
   - Une fois la vidéo téléversée, cliquez sur le bouton "Démarrer la détection".
   - Le traitement peut prendre quelques minutes, soyez patient.

3. **Téléchargez la vidéo annotée** :
   - Une fois le traitement terminé, un lien de téléchargement apparaîtra.
   - Cliquez sur le lien pour télécharger la vidéo annotée.

**Remarque** : Assurez-vous que la vidéo contient des joueurs et un ballon pour des résultats optimaux.
""")

uploaded_file = st.file_uploader("Uploader une vidéo", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    if st.button("Démarrer la détection"):
        st.text("Le traitement peut prendre plusieurs minutes, veuillez patienter...")
        with st.spinner("Analyse de la vidéo..."):
            files = {"file": open(temp_file_path, "rb")}
            response = requests.post(API_URL, files=files)
            files["file"].close()
            os.remove(temp_file_path)  # Supprimer le fichier temporaire après envoi

            if response.status_code == 200:
                result_filename = response.json().get("result")  # Récupérer le nom du fichier annoté
                if result_filename:
                    download_url = f"{result_filename}"  # Construire l'URL de téléchargement

                    st.success("Détection terminée avec succès !")

                    # Bouton de téléchargement
                    st.markdown(f"[📥 Télécharger la vidéo annotée]({download_url})", unsafe_allow_html=True)
                else:
                    st.error("Le nom du fichier annoté est introuvable.")
            else:
                st.error("Erreur lors du traitement de la vidéo.")