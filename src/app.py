import cv2
import numpy as np
from flask import Flask, request, make_response
from src.ai import FaceDetection, FaceRecognition


PREFIX = '/api/v1/5g'

app = Flask(__name__, template_folder='project/templates')

app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'key'


@app.route(PREFIX, methods=['POST'])
def index():
    nparr = np.fromstring(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces, bounding_boxes = detection.extract_faces_from_image(img)
    if faces is not None and bounding_boxes is not None:
        embeddings = recognition.retrieve_embeddings_for_face_list(faces)
        best_3_candidates = recognition.find_best_3_candidates(embeddings)

    else:
        out_image = img


    _, img_encoded = cv2.imencode('.jpg', out_image)
    response = make_response(img_encoded.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'

    return response


if __name__ == '__main__':
    detection = FaceDetection()
    recognition = FaceRecognition('models/facenet.pb')
    app.run(host='0.0.0.0', port=8080)