import cv2
import numpy as np
from flask import Flask, request, make_response
from ai import FaceDetection, FaceRecognition

PREFIX = '/api/v1/5g'

app = Flask(__name__, template_folder='project/templates')

app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'key'


@app.route('{}/classify'.format(PREFIX), methods=['POST'])
def classify():
    nparr = np.fromstring(request.data, np.uint8)
    out_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces, bounding_boxes = detection.extract_faces_from_image(out_image)
    if faces is not None and bounding_boxes is not None:
        nrof_faces = len(faces)
        embeddings = recognition.retrieve_embeddings_for_face_list(faces)
        for index in range(nrof_faces):
            name = recognition.classify_person(embeddings[index])
            bounding_box = bounding_boxes[index]
            out_image = cv2.rectangle(out_image, (int(bounding_box[0]), int(bounding_box[1])),
                                      (int(bounding_box[2]), int(bounding_box[3])), (0, 255, 0), 2)
            out_image = cv2.putText(out_image, name, (int(bounding_box[0]), int(bounding_box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    _, img_encoded = cv2.imencode('.jpg', out_image)
    response = make_response(img_encoded.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'

    return response


if __name__ == '__main__':
    detection = FaceDetection()
    recognition = FaceRecognition('models/frozen_graph.pb')
    app.run(host='0.0.0.0', port=8080)
