from deepface import DeepFace
import pandas as pd
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

print("\n\n\n\n")
image_path = 'Pessoas.jpeg'
image = cv2.imread(image_path)


df = DeepFace.represent("Caml.jpeg", model_name="Facenet")
caml_embedding = np.array(df[0]["embedding"])


df = DeepFace.represent("Raj2.jpeg", model_name="Facenet")
raj_embedding = np.array(df[0]["embedding"])

for rep in DeepFace.represent('Pessoas.jpeg', model_name="Facenet"):
    cur_embeddings = np.array(rep["embedding"])
    print(np.linalg.norm(cur_embeddings - caml_embedding),np.linalg.norm(cur_embeddings - raj_embedding))