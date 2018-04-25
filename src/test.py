import cv2
from ai import FaceDetection, FaceRecognition

if __name__ == '__main__':
    detection = FaceDetection()
    recognition = FaceRecognition('models/frozen_graph.pb')

    img = cv2.imread('/home/user/Downloads/mathilde1.jpg')
    faces, boxes = detection.extract_faces_from_image(img)
    embs = recognition.retrieve_embeddings_for_face_list(faces)

    print(recognition.classify_person(embs))
