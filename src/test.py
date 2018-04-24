import cv2
from src.ai import FaceDetection, FaceRecognition

if __name__ == '__main__':
    detection = FaceDetection()
    recognition = FaceRecognition('/home/maarten/Documents/Models/facenet.pb')

    img = cv2.imread('/home/maarten/Pictures/testje.jpg')
    faces = detection.extract_faces_from_image(img)

    print(recognition.find_top_3_candidates(recognition.find_top_3_candidates(faces)))