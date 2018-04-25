import cv2
from ai import FaceDetection, FaceRecognition

if __name__ == '__main__':
    detection = FaceDetection()
    recognition = FaceRecognition('models/frozen_graph.pb')

    img = cv2.imread('/home/maarten/Pictures/tim-dupont.jpeg')
    faces, boxes = detection.extract_faces_from_image(img)
    print(len(faces))
    embs = recognition.retrieve_embeddings_for_face_list(faces)

    recognition.add_person('Tim Dupont', embs[0])
