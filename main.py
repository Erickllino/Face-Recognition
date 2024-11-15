import mediapipe as mp
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


img     =  cv2.imread('img.jpg')
img    =  cv2.resize(img, (600,600), interpolation = cv2.INTER_NEAREST)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(
    model_selection = 1 , min_detection_confidence = 0.3) as face_detection:
    results = face_detection.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    annoteted_image = img.copy()
    print("Number of people detected", len(results.detections))
    print('Description of people\n')
    for i , detection in enumerate(results.detections):
        print("Person ", i ,'\n')
        print('confidence score ', detection.score)
        box = detection.location_data.relative_bounding_box
        x_start, y_start = int(box.xmin * img.shape[1]) , int(box.ymin*img.shape[0])
        x_end, y_end = int((box.xmin+box.width) * img.shape[1]) , int((box.ymin+box.width)*img.shape[0])
        annoteted_image = cv2.rectangle(img, (x_start,y_start), (x_end, y_end), (0,255,0), 5)
        mp_drawing.draw_detection = (annoteted_image, detection)


cv2.imshow('',annoteted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()











## https://www.youtube.com/watch?v=vXl1Ncsu6yE&t=99s

