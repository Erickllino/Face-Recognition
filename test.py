import cv2
import mediapipe as mp
from datetime import datetime
import numpy as np
import json
import os

# Inicialização do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Arquivos de dados
REGISTERED_FACES_FILE = "registered_faces.json"
ATTENDANCE_FILE = "attendance.json"
image_path = "img.jpg"

# Funções para manipulação de arquivos JSON
def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

# Carregando dados registrados
registered_faces = {k: np.array(v) for k, v in load_data(REGISTERED_FACES_FILE).items()}
attendance = load_data(ATTENDANCE_FILE)

# Função para calcular a similaridade
def calculate_similarity(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)  # Distância Euclidiana

# Função para normalizar os embeddings
def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

# Função para criar um embedding a partir dos landmarks do rosto
def create_embedding(face_landmarks):
    embedding = np.array([
        [landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark
    ]).flatten()  # Vetor 1D
    return normalize_embedding(embedding)

# Registro de presença
def register_attendance(name):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance[name] = time_now
    print(f"{name} marcou presença às {time_now}")
    save_data(attendance, ATTENDANCE_FILE)

# Registro de novo aluno
def register_aluno(face_mesh, image):
    print("Registrando novo aluno...")
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            embedding = create_embedding(face_landmarks)
            new_name = input("Digite o nome do novo aluno: ").strip()

            # Recarregar os dados para evitar sobrescrever registros
            current_faces = load_data(REGISTERED_FACES_FILE)
            current_faces = {k: np.array(v) for k, v in current_faces.items()}

            if new_name in current_faces:
                print(f"Aluno {new_name} já está registrado!")
                return

            # Adicionar o novo aluno e salvar
            current_faces[new_name] = embedding.tolist()  # Salvar como lista
            save_data(current_faces, REGISTERED_FACES_FILE)
            print(f"Aluno {new_name} registrado com sucesso!")
            return

    print("Nenhum rosto detectado. Tente novamente.")

# Processamento da imagem
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
                embedding = create_embedding(face_landmarks)
                recognized_name = "Desconhecido"
                min_similarity = float("inf")

                # Comparação com rostos registrados
                for name, ref_embedding in registered_faces.items():
                    similarity = calculate_similarity(embedding, ref_embedding)
                    print(f"Comparando com {name}: Similaridade = {similarity}")
                    if similarity < min_similarity and similarity < 0.5:  # Ajuste do limite
                        recognized_name = name
                        min_similarity = similarity

                # Ações com base no reconhecimento
                if recognized_name == "Desconhecido":
                    print("Rosto não reconhecido. Deseja registrar um novo aluno? (s/n)")
                    if input().strip().lower() == 's':
                        register_aluno(face_mesh, image)
                else:
                    register_attendance(recognized_name)

                # Desenhar landmarks na imagem
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

        # Salvar imagem processada
        output_path = "imagem_anotada.jpg"
        cv2.imwrite(output_path, image)
        print(f"Imagem processada salva em: {output_path}")

        # Exibir a imagem processada
        try:
            cv2.imshow("Imagem Processada", image)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

# Exibir os registros
print("\nRegistro de Presença:")
for name, time in attendance.items():
    print(f"{name}: {time}")

print("\nAlunos Registrados:")
for name in registered_faces.keys():
    print(name)
