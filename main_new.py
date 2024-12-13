from deepface import DeepFace
import pandas as pd
import cv2


image_path = 'image2.jpg'
image = cv2.imread(image_path)

df = DeepFace.represent(image, model_name="Facenet")[0]