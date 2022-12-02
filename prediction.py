from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras.models import Model, load_model
import keras.utils as image
import numpy as np
from keras.applications.resnet import ResNet50, preprocess_input
from glob import glob
from datetime import datetime
import PIL

IMAGE_SIZE = [224,224]

def load_my_model(model_type):
    model = load_model('./Models/fire_and_smoke2_model.h5')
    if model_type == "Resnet50":
        model = load_model('./Models/fire_and_smoke2_model.h5')
    elif model_type == "VGG16":
        model = load_model('./Models/vgg16.h5')
    else:
        model = load_model('./Models/vgg19.h5')
    #model.summary()
    return model


#_model = load_my_model()

def read_image(image_encoded):
    pil_image = image.load_img(BytesIO(image_encoded), target_size = (224,224))
    pil_image =  image.img_to_array(pil_image)
    print("Method 2 === ",pil_image)
    return pil_image

def preprocess(input_image_arr):
    input_image_arr  = input_image_arr / 255
    #print("Test after devision", input_image)
    input_image_arr = np.expand_dims(input_image_arr, axis = 0)
    return input_image_arr


def predict(image: np.ndarray, model_name):
    _model = load_my_model(model_name)
    predictions = _model.predict(image)
    print("Result == ",predictions)
    return predictions

