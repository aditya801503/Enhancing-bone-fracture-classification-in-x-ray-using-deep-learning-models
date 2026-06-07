# Enhancing-bone-fracture-classification-in-x-ray-using-deep-learning-models
An AI-powered web application for bone fracture detection and classification from X-ray images using deep learning and machine learning techniques.

## Overview

**Enhancing Bone Fracture Detection in X-ray Images Using Deep Learning Model** is a web-based application developed using Python and Flask. The main objective of this project is to detect and classify different types of bone fractures from X-ray images using deep learning and machine learning techniques.

The system uses the **VGG19 deep learning model** for feature extraction and a trained **Random Forest classifier** for fracture type prediction. In addition to fracture detection, the application also includes user registration, login authentication, MySQL database connectivity, image upload functionality, and an AI chatbot powered by Google Gemini API.

This project demonstrates how artificial intelligence can be applied in the healthcare domain, especially for medical image analysis and computer-aided fracture detection.

## Key Features

- User registration and login system
- MySQL database connectivity
- X-ray image upload functionality
- Image preprocessing before prediction
- Deep feature extraction using VGG19
- Fracture classification using Random Forest
- AI chatbot integration using Google Gemini API
- Session-based user management
- Web-based user interface using Flask
- Educational and research-oriented medical image analysis system

## Technologies Used

- Python
- Flask
- MySQL
- TensorFlow
- Keras
- VGG19
- Scikit-learn
- Random Forest
- Joblib
- NumPy
- Pillow
- Google Generative AI
- HTML
- CSS
- JavaScript

## System Workflow

1. The user registers or logs in to the application.
2. The user uploads an X-ray image.
3. The uploaded image is resized and preprocessed.
4. VGG19 extracts deep image features from the X-ray image.
5. The extracted features are scaled using a trained scaler.
6. The trained Random Forest classifier predicts the fracture type.
7. The predicted fracture result is displayed to the user.
8. The user can also interact with the AI chatbot for assistance.

## Fracture Classes

The system can classify the following fracture types:

- Avulsion fracture
- Comminuted fracture
- Fracture Dislocation
- Greenstick fracture
- Hairline Fracture
- Impacted fracture
- Longitudinal fracture
- Oblique fracture
- Pathological fracture
- Spiral Fracture

## Database Setup

Create the MySQL database using the following SQL script:

```sql
DROP DATABASE IF EXISTS `animals`;
CREATE DATABASE `animals`;
USE `animals`;

CREATE TABLE `users` (
    `id` INT PRIMARY KEY AUTO_INCREMENT,
    `name` VARCHAR(1000),
    `email` VARCHAR(1000),
    `password` VARCHAR(225)
);
Installation and Setup
1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
2. Install Required Dependencies
pip install flask mysql-connector-python numpy tensorflow scikit-learn joblib pillow google-generativeai werkzeug
3. Configure MySQL Database
Make sure MySQL is installed and running on your system. Then create the database using the SQL script provided above.

Update the database connection in the Python file if required:

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database="animals"
)
4. Configure Gemini API Key
Add your Google Gemini API key in the Python file:

GEMINI_API_KEY = "your-api-key-here"
Note: Do not expose your real API key in a public GitHub repository. For better security, use environment variables.

5. Add Required Model Files
Make sure the trained model and scaler files are available in the project directory:

vgg19_random_forest.joblib
vgg19_scaler.joblib
6. Run the Application
python app.py
7. Open the Application
Open the following URL in your browser:

http://127.0.0.1:5000/
Project Modules
User Authentication Module
This module allows users to register and log in to the application. User details such as name, email, and password are stored in the MySQL database.

Fracture Prediction Module
This module allows users to upload X-ray images. The uploaded image is preprocessed and passed through the VGG19 model for feature extraction. The extracted features are then classified using a trained Random Forest model to predict the fracture type.

Chatbot Module
This module uses Google Gemini API to provide AI-based chatbot responses. Users can interact with the chatbot after logging in to the application.

Applications
Medical image analysis projects
Bone fracture classification research
Deep learning-based healthcare applications
Academic mini projects and major projects
Final-year project demonstrations
Future Scope
Improve model accuracy using a larger X-ray dataset
Add doctor/admin dashboard
Store prediction history in the database
Add password hashing for better security
Use environment variables for API keys and database credentials
Deploy the application on a cloud platform
Add support for more medical image categories
Generate downloadable prediction reports
Limitations
Prediction accuracy depends on the quality of the uploaded X-ray image.
The model performance depends on the dataset used for training.
The system is not intended for real-time clinical diagnosis.
The application requires trained model files to make predictions.
Internet access may be required for chatbot functionality.
Disclaimer
This project is developed for educational and research purposes only. The prediction results generated by this system should not be considered as professional medical advice. Always consult a qualified medical professional for diagnosis and treatment.

