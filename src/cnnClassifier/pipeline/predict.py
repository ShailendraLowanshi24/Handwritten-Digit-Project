import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

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