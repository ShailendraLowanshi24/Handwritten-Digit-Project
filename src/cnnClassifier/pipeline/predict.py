import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)
    
        if result[0] == 1:
            prediction = '0'
            return [{ "image" : prediction}]
        elif result[1] == 2:
            prediction = '1'
            return [{ "image" : prediction}]
        elif result[2] == 3:
            prediction = '2'
            return [{ "image" : prediction}]
        elif result[3] == 4:
            prediction = '3'
            return [{ "image" : prediction}]
        elif result[4] == 5:
            prediction = '4'
            return [{ "image" : prediction}]
        elif result[5] == 6:
            prediction = '5'
            return [{ "image" : prediction}]
        elif result[6] == 7:
            prediction = '6'
            return [{ "image" : prediction}]
        elif result[7] == 8:
            prediction = '7'
            return [{ "image" : prediction}]
        elif result[8] == 9:
            prediction = '8'
            return [{ "image" : prediction}]
        else:
            prediction = '9'
            return [{ "image" : prediction}]