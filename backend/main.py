from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
import cv2
import numpy as np
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans
from filterpy.kalman import KalmanFilter
from fastapi.responses import FileResponse


# Initialisation de l'application FastAPI
app = FastAPI()

# Chargement du modèle YOLO
model = YOLO("model/best.pt")

# Dossier de stockage
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# Initialisation du filtre de Kalman pour le ballon
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # État = (x, y, vx, vy)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # Modèle de transition

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Observation directe de x, y

    kf.P *= 1000  # Incertitude initiale
    kf.R *= 10  # Bruit de mesure
    kf.Q = np.eye(4) * 0.1  # Bruit du modèle
    return kf


ball_kf = init_kalman()  # Kalman pour le ballon
ball_position = None  # Dernière position connue du ballon


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processed_video = process_video(file_path)
    return {"message": "Video processed successfully", "result": processed_video}


def extract_dominant_color(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    roi = roi.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(roi)  # Augmenter les clusters
    dominant_colors = kmeans.cluster_centers_
    return tuple(map(int, np.median(dominant_colors, axis=0)))  # Utiliser la médiane des couleurs


def process_video(video_path):
    global ball_kf, ball_position
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    result_filename = "annotated_" + os.path.basename(video_path)  # Nom du fichier annoté
    output_path = os.path.join(RESULT_FOLDER, result_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    team_colors = None  # Stockage des couleurs des équipes

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        player_colors = []
        player_boxes = []
        detected_ball = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label == "joueur":
                    color = extract_dominant_color(frame, (x1, y1 + (y2 - y1) // 3, x2, y2))
                    player_colors.append(color)
                    player_boxes.append((x1, y1, x2, y2))

                if label == "ballon":
                    detected_ball = (x1 + x2) // 2, (y1 + y2) // 2  # Centre du ballon

        if player_colors and team_colors is None:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(player_colors)
            team_colors = kmeans.cluster_centers_

        # Mise à jour du filtre de Kalman pour le ballon
        if detected_ball:
            ball_kf.update(np.array(detected_ball))  # Mise à jour avec la mesure actuelle
            ball_position = detected_ball  # Stocker la position réelle détectée
        else:
            ball_kf.predict()  # Prédiction de la position future
            ball_position = (int(ball_kf.x[0]), int(ball_kf.x[1]))

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label == "joueur" and team_colors is not None:
                    color = extract_dominant_color(frame, (x1, y1 + (y2 - y1) // 3, x2, y2))
                    team_idx = np.argmin(
                        [np.linalg.norm(color - team_colors[0]), np.linalg.norm(color - team_colors[1])])
                    bbox_color = (0, 255, 0) if team_idx == 0 else (0, 0, 255)  # Équipe A en vert, Équipe B en rouge
                elif label == "arbitre":
                    bbox_color = (255, 255, 0)  # Arbitre en jaune
                elif label == "ballon":
                    bbox_color = (0, 165, 255)  # Orange pour le ballon
                else:
                    bbox_color = (255, 0, 0)  # Autres classes en bleu

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

        # Dessiner la position interpolée du ballon
        if ball_position:
            cv2.circle(frame, ball_position, 5, (0, 165, 255), -1)  # Dessiner le ballon en orange

        out.write(frame)

    cap.release()
    out.release()

    # Retourner l'URL de téléchargement
    return f"http://localhost:8000/download/{result_filename}"


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    return {"error": "Fichier introuvable"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
