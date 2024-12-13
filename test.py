import cv2
from deepface import DeepFace
from datetime import datetime
import numpy as np
import json
import os

REGISTERED_FACES_FILE = "registered_faces.json"
ATTENDANCE_FILE = "attendance.json"
image_path = "image2.jpg"

def load_data(file_path):
    """Carrega dados de um arquivo JSON."""
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_data(data, file_path):
    """Salva dados em um arquivo JSON."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# Carregar os dados existentes
registered_faces = load_data(REGISTERED_FACES_FILE)
attendance = load_data(ATTENDANCE_FILE)

def register_attendance(name):
    """Registra a presença de um aluno."""
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name not in attendance:
        attendance[name] = time_now
        print(f"{name} marcou presença às {time_now}")
        save_data(attendance, ATTENDANCE_FILE)
    else:
        print(f"{name} já teve presença registrada.")

def register_student(face_embedding):
    """Registra um novo aluno no sistema."""
    new_name = input("Digite o nome do novo aluno: ").strip()

    if new_name in registered_faces:
        print(f"Aluno {new_name} já está registrado!")
        return

    registered_faces[new_name] = face_embedding
    save_data(registered_faces, REGISTERED_FACES_FILE)
    print(f"Aluno {new_name} registrado com sucesso!")

def detect_and_recognize(image):
    """Detecta e reconhece rostos em uma imagem."""
    try:
        # Detectar rostos na imagem
        detections = DeepFace.extract_faces(image, detector_backend="mtcnn", enforce_detection=True)

        for detection in detections:
            face_image = detection["face"]

            # Obter o embedding do rosto
            face_embedding = DeepFace.represent(face_image, model_name="Facenet")[0]["embedding"]

            # Comparar com os rostos registrados
            recognized_name = "Desconhecido"
            min_similarity = float("inf")
            for name, ref_embedding in registered_faces.items():
                similarity = np.linalg.norm(np.array(face_embedding) - np.array(ref_embedding))
                if similarity < min_similarity and similarity < 0.5:  # Ajuste o limite conforme necessário
                    recognized_name = name
                    min_similarity = similarity

            if recognized_name != "Desconhecido":
                register_attendance(recognized_name)
            else:
                print("Rosto desconhecido encontrado. Solicitando registro.")
                register_student(face_embedding)

    except Exception as e:
        print("Erro ao reconhecer alunos:", e)

# Carregar a imagem
image = cv2.imread(image_path)
if image is None:
    image_path = input("Erro: Caminho da imagem incorreto. Insira um novo caminho: ").strip()
    image = cv2.imread(image_path)

if image is not None:
    detect_and_recognize(image)

# Exibir os registros
print("\nRegistro de Presença:")
for name, time in attendance.items():
    print(f"{name}: {time}")

print("\nAlunos Registrados:")
for name in registered_faces.keys():
    print(name)
