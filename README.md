# 🎯 Foot-Analyse  

Une application de détection et d'analyse des joueurs de football dans une vidéo de match, utilisant **YOLO** et **OpenCV** pour le traitement des images, ainsi que **FastAPI** et **Streamlit** pour l'interface utilisateur.  

---

## 📌 Fonctionnalités  
✅ Détection des joueurs, arbitres et du ballon dans une vidéo de match  
✅ Segmentation des joueurs par équipe grâce au clustering (K-Means)  
✅ Suivi des joueurs et du ballon avec un filtre de Kalman  
✅ Interface web pour télécharger et analyser une vidéo  

---

## 🚀 Démo  
Ajoutez ici une capture d'écran ou un GIF montrant votre application en action.  

---

## 🛠️ Installation  

### 1️⃣ Cloner le projet  
```bash
git clone https://github.com/Ayoub1899/foot-analyse.git
cd foot-analyse

```

### 2️⃣ Installer les dépendances
Assurez-vous d'avoir Python 3.8+ installé. Puis exécutez :
```bash
pip install -r requirements.txt
```
---
### 🔧 Utilisation
1️⃣ Lancer le backend (FastAPI)
Depuis le dossier backend/, exécutez :
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Le backend sera accessible à l'adresse : http://127.0.0.1:8000

### 2️⃣ Lancer le frontend (Streamlit)
Dans un autre terminal, depuis frontend/, exécutez :
```bash
streamlit run app.py
```

L'interface utilisateur sera disponible sur http://localhost:8501
