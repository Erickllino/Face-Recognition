import cv2
import keras
from deepface import DeepFace
from datetime import datetime
import numpy as np
import json
import os

REGISTERED_FACES_FILE = "registered_faces.json"
ATTENDANCE_FILE = "attendance.json"
image_path = "image2.jpg"

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

registered_faces = load_data(REGISTERED_FACES_FILE)
attendance = load_data(ATTENDANCE_FILE)

def register_attendance(name):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance[name] = time_now
    print(f"{name} marcou presença às {time_now}")
    save_data(attendance, ATTENDANCE_FILE)

def detect_and_register(image):
    print("Registrando novo aluno...")
    try:
        
        face_embedding = DeepFace.represent(image, model_name="Facenet")[0]["embedding"]
        
        new_name = input("Digite o nome do novo aluno: ").strip()

        if new_name in registered_faces:
            print(f"Aluno {new_name} já está registrado!")
            return

        registered_faces[new_name] = face_embedding
        save_data(registered_faces, REGISTERED_FACES_FILE)
        print(f"Aluno {new_name} registrado com sucesso!")
    except Exception as e:
        print("Erro ao registrar aluno:", e)

def detect_and_recognize(image):
    try:
        detections = DeepFace.extract_faces(image, detector_backend="mtcnn", enforce_detection=True)[0]

        face_embedding = DeepFace.represent(detections, model_name="Facenet")[0]["embedding"]

        recognized_name = "Desconhecido"
        min_similarity = float("inf")
        for name, ref_embedding in registered_faces.items():
            similarity = np.linalg.norm(np.array(face_embedding) - np.array(ref_embedding))
            if similarity < min_similarity and similarity < 0.4:  # Ajuste o limite conforme necessário
                recognized_name = name
                min_similarity = similarity

        if recognized_name != "Desconhecido":
            register_attendance(recognized_name)
        else:
            detect_and_register(image)
    except Exception as e:
        print("Erro ao reconhecer aluno:", e)

image = cv2.imread(image_path)
if image is None:
    image_path = input("Erro: Caminho da imagem incorreto. Insira um novo caminho: ").strip()
    image = cv2.imread(image_path)

if image is not None:
    
    detect_and_recognize(image)
    
    #detect_and_register(image)
    

print("\nRegistro de Presença:")
for name, time in attendance.items():
    print(f"{name}: {time}")

print("\nAlunos Registrados:")
for name in registered_faces.keys():
    print(name)
