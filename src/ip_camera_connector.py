import cv2
import base64
import urllib
import threading
import numpy as np

class IpCamVideoStream:
    def __init__(self, url, name, password):
        """Interface to connect an IP camera
        :param url: The url of the IP camera stream.
        :param name: The username to access the camera stream (enter blank username if not required)
        :param password: The password to access the camera stream (enter blank password if not required)
        """
        try:
            auth = '{}:{}'.format(name, password)
            base64auth = base64.standard_b64encode(auth.encode('utf-8'))
            self.request = urllib.request.Request('http://{}'.format(url))
            self.request.add_header('Authorization', 'Basic {}'.format(base64auth.decode('utf-8')))
            self.camera_stream = urllib.request.urlopen(self.request)
        except Exception as e:
            print('[ERROR]: Unable to connect camera.  {}'.format(e))

        self.stopped = True
        self.frame = None

    def start(self):
        self.stopped = False
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        bytes = b''
        while True:
            bytes += self.camera_stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b + 2]
                bytes = bytes[b + 2:]

                self.frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if self.stopped:
                return

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
