import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



#class PredictionPipeline:
#    def __init__(self,filename):
#        self.filename =filename


    
#    def predict(self):
        # load model
#        model = load_model(os.path.join("artifacts","training", "model.h5"))

#        imagename = self.filename
#        test_image = image.load_img(imagename, target_size = (224,224))
#        test_image = image.img_to_array(test_image)
#        test_image = np.expand_dims(test_image, axis = 0)
#        result = np.argmax(model.predict(test_image), axis=1)
        #print(result)

#        prediction = str(result)
#        return [{"image": prediction}]

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.class_mapping = {
            0: 'Class 0',
            1: 'Class 1',
            2: 'Class 2',
            3: 'Class 3',
            4: 'Class 4',
            5: 'Class 5',
            6: 'Class 6',
            7: 'Class 7',
            8: 'Class 8',
            9: 'Class 9'
        }

    def predict(self):
        # Load model
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = load_model(model_path)

        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Perform the prediction
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions[0])
        prediction_label = self.class_mapping.get(predicted_class, 'Unknown')

        return [{'image': prediction_label}]
    
        #if result[0] == 1:
        #    prediction = '0'
        #    return [{ "image" : prediction}]
        #elif result[1] == 2:
        #    prediction = '1'
        #    return [{ "image" : prediction}]
        #elif result[2] == 3:
        #    prediction = '2'
        #    return [{ "image" : prediction}]
        #elif result[3] == 4:
        #    prediction = '3'
        #    return [{ "image" : prediction}]
        #elif result[4] == 5:
        #    prediction = '4'
        #    return [{ "image" : prediction}]
        #elif result[5] == 6:
        #    prediction = '5'
        #    return [{ "image" : prediction}]
        #elif result[6] == 7:
        #    prediction = '6'
        #    return [{ "image" : prediction}]
        #elif result[7] == 8:
        #    prediction = '7'
        #    return [{ "image" : prediction}]
        #elif result[8] == 9:
        #    prediction = '8'
        #    return [{ "image" : prediction}]
        #else:
        #    prediction = '9'
        #    return [{ "image" : prediction}]