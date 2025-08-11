import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename

    def predict(self):
        # It's good practice to specify the full path to the model
        model = load_model(os.path.join("model","model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        test_image = load_img(imagename, target_size = (224,224))
        test_image = img_to_array(test_image)
        # Normalize the image pixels to be between 0 and 1
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Get the model's prediction
        result = model.predict(test_image)
        # Find the index of the highest probability
        predicted_class_index = np.argmax(result, axis=1)
        
        print(f"Prediction probabilities: {result}")
        print(f"Predicted class index: {predicted_class_index}")


        class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
        
        # Map the index to the class name
        prediction = class_names[predicted_class_index[0]]
        
        return [{"image" : prediction}]