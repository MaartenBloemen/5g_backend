import requests
import cv2
import argparse
import numpy as np
from ip_camera_connector import IpCamVideoStream


def main(args):
    headers = {'content-type': 'image/jpeg'}
    camera = IpCamVideoStream(args.camera_url, args.name, args.password)
    camera.start()

    while not camera.stopped:
        if camera.read() is not None:
            frame = cv2.resize(camera.read(), (1280, 720))
            _, img_encoded = cv2.imencode('.jpg', frame)
            request = requests.post('http://{}/image/classify'.format(args.backend_url), data=img_encoded.tostring(),
                                     headers=headers)

            if request.status_code == 200:
                nparr = np.fromstring(request.content, np.uint8)
                classified_img = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGBA)
                cv2.imshow('Classified image', cv2.cvtColor(classified_img, cv2.COLOR_BGR2RGBA))
                print('Time to classify frame in seconds: {}'.format(round(request.elapsed.total_seconds(), 6)))

                if cv2.waitKey(1) == 27:
                    camera.stop()
                    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('camera_url', type=str, help='The url of the IP camera.')
    parser.add_argument('--name', type=str, help='The login name for the IP camera.', default='admin')
    parser.add_argument('--password', type=str, help='The login password for the IP camera.', default='')
    parser.add_argument('--backend_url', type=str, help='The url of the backend.', default='localhost:8080/api/v1/5g')

    args = parser.parse_args()

    main(args)
