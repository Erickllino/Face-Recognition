import cv2
from deepface import DeepFace
from datetime import datetime
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd


def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return {}

def save_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def register_attendance(attendance_dict, attendance_csv= None):
    """ creates a register of the attendance on a certain day
    the register is made by adding a new column on csv file 
        Args:
                attendance_dict: dict with bool values
                attendance_csv: path to the csv
    """

    # the column name corresponds to the date
    time_now = datetime.now().strftime("%d-%m-%Y")
    # if there is no csv, create the csv
    if attendance_csv is None or (not os.path.exists(attendance_csv)):
        df = pd.DataFrame(list(attendance_dict.items()), columns=['student_id', time_now])
    # if there is a csv, add a new column
    else:
        df = pd.read_csv(attendance_csv,index_col= "student_id")
        df[time_now] = df['student_id'].map(attendance_dict)

    df.to_csv("attendance.csv")

def register_new_students():
    "function to add new students in the embedding json"
    NotImplemented

# this function is here to help the case when the students wont all give their images individually
# just a single pic with all students should be enough to get all the required embeddings
def register_faces_from_single_pic(images_folder):

    detections = DeepFace.extract_faces(images_folder, detector_backend="mtcnn", enforce_detection=True)

    for i,detection in enumerate(detections):

        face_image = (255*detection['face']).astype(np.uint8)
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(images_folder, f'rosto{i}.png'),face_image_rgb)

# this function is for the case where all the students give a photo
def detect_and_register(images_folder, force= False):
    """generate a json file with the embeddings of each student
        images_folder: the path to a folder that has the image of all the students
    """
    # dict with student_id as keys and embeddings as values 
    registered_faces = {}

    for student_file in os.listdir(images_folder):

        student_id = Path(student_file).stem # removes the file extension from img file name
        image_path = os.path.join(images_folder, student_file)

        try:
            face_embedding = DeepFace.represent(image_path, model_name="Facenet")[0]["embedding"]
        except:
            print(f"erro com a imagem: {student_file}")
            continue

        if student_id in registered_faces and (not force):
            print(f"Aluno {student_id} já está registrado!")
            continue

        registered_faces[student_id] = face_embedding
        print(f"Aluno {student_id} registrado com sucesso!")

    # save embeddings
    faces_db_folder = os.path.dirname(images_folder)
    faces_db_path = os.path.join(faces_db_folder,"faces_embeddings.json")
    with open(faces_db_path, "w") as faces_f:
        json.dump(registered_faces,faces_f)
    
    return faces_db_path

def detect_and_recognize(image_path, faces_db_path, max_difference = 11):
    """image_path: path to the image of the classroom in a certain day with all the students in the photo"""

    # read the json with the faces' embeddings
    with open(faces_db_path, "r") as faces_f:
        registered_faces = json.load(faces_f)

    attendance_dict = {student_id : False for student_id in registered_faces.keys()}

    # get the embeddings from the faces in the image
    reps = DeepFace.represent(image_path, model_name="Facenet")

    # iterate through all the faces in the image
    for rep in reps:
        face_embedding = rep["embedding"]
        
        # for each face search a registered embedding
        for student_id, ref_embedding in registered_faces.items():

            # euclidean distance between the embeddings
            difference = np.linalg.norm(np.array(face_embedding) - np.array(ref_embedding))

            # if is more similar than a certain threshold, the presence is registered
            if difference < max_difference:
                attendance_dict[student_id] = True
               
    
    register_attendance(attendance_dict)


if __name__ == "__main__":
    faces_db_path = detect_and_register("Students")
    detect_and_recognize("Pessoas.jpeg", faces_db_path)