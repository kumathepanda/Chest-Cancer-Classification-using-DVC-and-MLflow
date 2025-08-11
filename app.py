from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from Chest_cancer_classification.utils.common import decodeImage
from Chest_cancer_classification.pipeline.stage_05_prediction import PredictionPipeline

# Set environment variables for consistent encoding
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Define the filename for the input image
FILENAME = "inputImage.jpg"

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Renders the main landing page."""
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def train_route():
    """
    Triggers the DVC pipeline to retrain the model.
    NOTE: This is a blocking call and will take a long time.
    In a production environment, this should be handled by a background worker.
    """
    os.system("dvc repro")
    return "Training completed successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    """
    Receives an image, saves it, and returns a prediction.
    """
    try:
        # Get the base64 encoded image from the request
        image_data = request.json['image']
        
        # Decode the image and save it to the predefined filename
        decodeImage(image_data, FILENAME)
        
        # Instantiate the prediction pipeline with the saved image
        # This is done here to ensure the latest model is always loaded
        classifier = PredictionPipeline(FILENAME)
        result = classifier.predict()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500


if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=8080) # Use port 8080 for deployment
