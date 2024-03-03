import numpy as np
import random
import cv2

import tensorflow as tf

from dataclasses import dataclass

from options.load_labels_and_classes import load_labels_and_classes_dict

import warnings
warnings.filterwarnings("ignore")

loaded_dictionary = load_labels_and_classes_dict()

@dataclass
class CONFIG:
    IMAGE_SIZE = (224, 224)
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 32
    RANDOM_STATE = 42

def set_seed(random_seed):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def make_predictions(model, img):
    rescaled_image = img / 255.0
    resized_image = cv2.resize(rescaled_image, CONFIG.IMAGE_SIZE)
    model_input_image = np.expand_dims(resized_image, axis=0)
    prediction = model.predict(model_input_image)
    final_prediction = np.argmax(prediction, axis=1)
    matching_keys = [key for key, value in loaded_dictionary.items() if value == final_prediction[0]]
    confidence = prediction[0, np.argmax(prediction)]

    return matching_keys[0], confidence
