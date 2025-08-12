# End-to-End Lung Cancer Classification using MLOps

This project implements a complete end-to-end MLOps pipeline for classifying four types of lung cancer (Adenocarcinoma, Large Cell Carcinoma, Normal, and Squamous Cell Carcinoma) from CT scan images. The system is built following a reproducible workflow using **DVC** for pipelining, **MLflow** for experiment tracking, and is deployed as a web application using **Docker** and **Microsoft Azure**.

The core of the project is a deep learning model based on the **DenseNet121** architecture, which has been fine-tuned to achieve high performance on this specific medical imaging task.

## 🚀 Live Demo

*[https://chestcancerclassification-bxh6gefng0faandn.centralindia-01.azurewebsites.net/]*

## ✨ Features

  * **AI-Powered Prediction**: Classifies CT scan images into four distinct categories.
  * **Interactive Web UI**: A clean, user-friendly interface built with Flask and HTML/CSS/JS.
  * **Upload Your Own Image**: Users can upload their own CT scan images for prediction.
  * **Sample Images**: Includes pre-loaded sample images for each class to allow users to easily test the application's functionality.
  * **Reproducible Training Pipeline**: A fully versioned and reproducible training pipeline built with DVC.
  * **Experiment Tracking**: All experiments are tracked with MLflow, logging parameters, metrics, and model artifacts.
  * **CI/CD Deployment**: Automated deployment pipeline using GitHub Actions to build and deploy the application to Azure.

## 🛠️ Technology Stack

  * **Backend**: Python, Flask
  * **Deep Learning**: TensorFlow, Keras
  * **MLOps**: DVC, MLflow
  * **Deployment**: Docker, Gunicorn, Microsoft Azure (App Service & Container Registry)
  * **CI/CD**: GitHub Actions
  * **Frontend**: HTML, Tailwind CSS, JavaScript

## Workflow

This project follows a structured MLOps workflow, managed by DVC:

1.  **Data Ingestion**: Downloads and extracts the dataset.
2.  **Prepare Base Model**: Loads the `DenseNet121` model and prepares it for transfer learning.
3.  **Model Training**:
      * **Warm-up Phase**: Trains only the new classification head.
      * **Fine-Tuning Phase**: Unfreezes the top layers and fine-tunes the model with a low learning rate.
4.  **Model Evaluation**: Evaluates the trained model on a test set and logs metrics using MLflow.

## ⚙️ Local Setup and Installation

To run this project on your local machine, follow these steps:

### **1. Clone the Repository**

```bash
git clone https://github.com/kumathepanda/Chest-Cancer-Classification-using-DVC-and-MLflow.git
cd Chest-Cancer-Classification-using-DVC-and-MLflow
```

### **2. Create a Virtual Environment**

It's highly recommended to use a virtual environment.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### **4. Set Up MLflow Credentials**

This project uses MLflow for experiment tracking. Set up the following environment variables (you can use a `.env` file for local development).

```
export MLFLOW_TRACKING_URI="<your-mlflow-uri>"
export MLFLOW_TRACKING_USERNAME="<your-username>"
export MLFLOW_TRACKING_PASSWORD="<your-password>"
```

## 🚀 How to Run

### **1. Run the DVC Training Pipeline**

To run the entire end-to-end training pipeline, use the `dvc repro` command. This will execute all stages defined in `dvc.yaml`.

```bash
dvc repro
```

This will train the model and save the final `model.h5` in the `artifacts/training/` directory.

### **2. Run the Flask Application**

To run the web application locally, execute the `app.py` script:

```bash
python app.py
```

Then, open your web browser and navigate to `http://127.0.0.1:8080`.

## 🚢 Deployment

This application is designed for deployment using Docker and is hosted on Microsoft Azure.

1.  **Build the Docker Image**:
    ```bash
    docker build -t <your-acr-name>.azurecr.io/flaskapp:latest .
    ```
2.  **Push to Azure Container Registry**:
    ```bash
    docker push <your-acr-name>.azurecr.io/flaskapp:latest
    ```
3.  **Deploy on Azure App Service**: The application is deployed using an Azure App Service configured to run a Docker container. A GitHub Actions workflow is set up for CI/CD, which automatically builds and deploys the application whenever new changes are pushed to the `main` branch.

## 📂 Project Structure

```
├── .github/workflows/       # CI/CD pipeline with GitHub Actions
├── model/                     # Stores the final trained model for deployment
├── src/
│   └── Chest_cancer_classification/
│       ├── components/      # Modular components (data ingestion, training, etc.)
│       ├── config/          # Configuration manager
│       ├── pipeline/        # DVC pipeline stages
│       └── ...
├── templates/
│   └── index.html           # Frontend code
├── app.py                     # Main Flask application
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
└── dvc.yaml                   # DVC pipeline definition
```

## 🙏 Acknowledgements

This project was inspired by and builds upon the findings of the research paper: ["LCDctCNN: Lung Cancer Diagnosis of CT scan Images Using CNN Based Model"](https://www.google.com/search?q=https://example.com/link-to-paper).
