import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import json
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

REGISTERED_FACES_FILE = "registered_faces.json"
ATTENDANCE_FILE = "attendance.json"
image_path = "img.jpg"

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

registered_faces = {k: np.array(v) for k, v in load_data(REGISTERED_FACES_FILE).items()}
attendance = load_data(ATTENDANCE_FILE)

def calculate_similarity(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2) 


def register_attendance(name):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance[name] = time_now
    print(f"{name} marcou presença às {time_now}")
    save_data(attendance, ATTENDANCE_FILE)
    

def register_aluno(face_mesh, image):
    print("Registrando novo aluno...")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            embedding = np.array([face_landmarks.landmark[i].x for i in range(len(face_landmarks.landmark))])
            new_name = input("Digite o nome do novo aluno: ").strip()

            # Recarregar os dados para garantir que registros antigos não sejam perdidos
            current_faces = load_data(REGISTERED_FACES_FILE)
            current_faces = {k: np.array(v) for k, v in current_faces.items()}  # Convert listas para arrays

            if new_name in current_faces:
                print(f"Aluno {new_name} já está registrado!")
                return

            # Adicionar o novo aluno
            current_faces[new_name] = embedding

            # Salvar convertendo os arrays novamente para listas
            save_data({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in current_faces.items()}, REGISTERED_FACES_FILE)
            print(f"Aluno {new_name} registrado com sucesso!")
            return

    print("Nenhum rosto detectado. Tente novamente.")



image = cv2.imread(image_path)

if image is None:
    image_path = input("Erro: Caminho da imagem incorreto. Insira um novo caminho: ").strip()
    image = cv2.imread(image_path)

if image is not None:
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                embedding = np.array([face_landmarks.landmark[i].x for i in range(len(face_landmarks.landmark))])
                recognized_name = "Desconhecido"
                min_similarity = float("inf")
                for name, ref_embedding in registered_faces.items():
                    embedding = embedding / np.linalg.norm(embedding)
                    ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
                    similarity = calculate_similarity(embedding, np.array(ref_embedding))
                    print(similarity)
                    if similarity < min_similarity and similarity < 0.4:
                        recognized_name = name
                        min_similarity = similarity

                if recognized_name != "Desconhecido":
                    register_attendance(recognized_name)
                else:
                    register_aluno(face_mesh, image)

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

        output_path = "imagem_anotada.jpg"
        cv2.imwrite(output_path, image)
        print(f"Imagem processada salva em: {output_path}")

        try:
            cv2.imshow("Imagem Processada", image)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

print("\nRegistro de Presença:")
for name, time in attendance.items():
    print(f"{name}: {time}")

print("\nAlunos Registrados:")
for name in registered_faces.keys():
    print(name)
