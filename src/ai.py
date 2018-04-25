import itertools
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from dependencies import facenet
from dependencies import detect_face


class FaceDetection:
    MINSIZE = 20  # Minimum size of face
    THRESHOLD = [0.6, 0.7, 0.7]  # Three steps's threshold
    FACTOR = 0.709  # Scale factor

    def __init__(self, gpu_memory_fraction: float = 0.2):
        """Face detection class initialisation

        :param gpu_memory_fraction: The upperbound of GPU memory that can be used for face detection (default=20%)
        """
        self.pnet, self.rnet, self.onet = self.create_network_face_detection(gpu_memory_fraction)

    def create_network_face_detection(self, gpu_memory_fraction: float):
        """Create MTCNN face detection network

        :param gpu_memory_fraction: The upperbound of GPU memory that can be used for face detection
        :return: MTCNN network instances (Proposal network, Refinement network and Output network)
        """
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

        return pnet, rnet, onet

    def extract_faces_from_image(self, image: np.ndarray, image_size: int = 160, margin: int = 44):
        """Find and extracts all faces in an image

        :param image: The target frame (image) to find faces in
        :param image_size: The size of the face images (default=160)
        :param margin: The margin around the faces (default=44)
        :return: List of found faces on the frame
        :raises: ThresholdNotMetException: Raised when a face has a certainty below 95%
        :raises: NoFacesFoundExeption: Raised when no faces are found in the frame
        """
        faces = []
        final_bounding_boxes = []
        bounding_boxes, _ = detect_face.detect_face(image, self.MINSIZE,
                                                    self.pnet, self.rnet, self.onet, self.THRESHOLD, self.FACTOR)
        nrof_faces = len(bounding_boxes)

        if nrof_faces > 0:
            frame_size = np.asarray(image.shape)[0:2]
            for face_index_on_frame in range(nrof_faces):
                if bounding_boxes[face_index_on_frame][4] > 0.95:
                    det = np.squeeze(bounding_boxes[face_index_on_frame, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, frame_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, frame_size[0])
                    cropped = np.array(image)[bb[1]:bb[3], bb[0]:bb[2], :]
                    aligned = Image.fromarray(cropped).resize((image_size, image_size), Image.ANTIALIAS)
                    prewhitened = facenet.prewhiten(np.array(aligned))
                    faces.append(prewhitened)
                    final_bounding_boxes.append(bounding_boxes[face_index_on_frame])
            if len(faces) > 0:
                return faces, final_bounding_boxes

        return None, None


class FaceRecognition:
    def __init__(self, pb_model_location: str, gpu_memory_fraction: float = 0.6):
        """Face recognition class initialisation

        :param pb_model_location: Frozen (.pb) facenet model location on disk
        :param gpu_memory_fraction: The upperbound of GPU memory that can be used for face detection (default=60%)
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as self.sess:
                facenet.load_model(pb_model_location)
                # Retrieve the required input and output tensors for classification from the model
                self.images_placeholder = self.sess.graph.get_tensor_by_name('input:0')
                self.embeddings = self.sess.graph.get_tensor_by_name('embeddings:0')
                self.phase_train_placeholder = self.sess.graph.get_tensor_by_name('phase_train:0')
                self.embedding_size = self.embeddings.get_shape()[1]

    def retrieve_embeddings_for_face_list(self, faces):
        """Retrieves the embeddings for a list of face images

        :param faces: List of images containing aligned faces
        :return: A list of embeddings corresponding to the faces
        """
        feed_dict = {self.images_placeholder: faces, self.phase_train_placeholder: False}
        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array

    def classify_person(self, unknown_embeddings):
        try:
            know_embeddings = np.load('models/known_persons.npy').item()
        except IOError:
            return 'Unknown'
        lowest_dist = 2
        name = ''

        for person in know_embeddings:
            for i in range(len(unknown_embeddings)):
                dist = np.sqrt(
                    np.sum(np.square(np.subtract(unknown_embeddings[i], know_embeddings.get(person)))))
                if dist < lowest_dist:
                    name = person
                    lowest_dist = dist

        if lowest_dist < 1:
            return name
        else:
            return 'Unknown'

    def add_person(self, name, embedding):
        try:
            know_embeddings = np.load('models/known_persons.npy').item()
        except IOError:
            know_embeddings = {}

        if name in know_embeddings:
            know_embeddings.update({name: embedding})
        else:
            know_embeddings[name] = embedding

        np.save('models/known_persons.npy', know_embeddings)
