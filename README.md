Crop Leaf Identification 🌱

This repository presents a machine learning–based system for crop leaf disease identification using deep learning models.
The project is designed for inference-ready deployment, with trained models versioned and managed using Git LFS.

Features

Automated crop leaf disease classification
Pre-trained deep learning models using TensorFlow and Keras
Lightweight inference using TensorFlow Lite models
Clean and scalable machine learning repository structure
No datasets or virtual environments committed
Project Structure

crop-detectionv1.6

models

best_classifier.keras

classifier.keras

crop.tflite

embedding_model

train.py – Model training script

dataset.py – Dataset handling utilities

convert_tflite.py – Conversion to TensorFlow Lite

test.py – Model testing script

test_visualise.py – Prediction visualisation

.gitignore

.gitattributes

README.md

Model Details

Framework: TensorFlow and Keras

Model formats used:

.keras for full training models

.tflite for lightweight inference models

Models are versioned using Git LFS

Getting Started

Clone the repository from GitHub.

Install the required Python dependencies listed in requirements.txt.

Run the test script to perform inference or evaluation.

Important Notes

Datasets are not included in this repository

Virtual environments are ignored

Only final trained models are versioned

Suitable for academic, research, and deployment use

Dataset
Due to size constraints, datasets are excluded from version control.
Place datasets locally in any of the following directories:
data, dataset, or crops

Versioning
Model versions are managed using Git tags to track releases and updates.

Author
Madhavan Rangaraj

License
This project is intended for academic and research purposes only.
