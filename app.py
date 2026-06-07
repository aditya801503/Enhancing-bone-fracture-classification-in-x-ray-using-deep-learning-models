from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
import mysql.connector, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from PIL import UnidentifiedImageError, Image
from tensorflow.keras.applications.mobilenet import preprocess_input
from werkzeug.utils import secure_filename
import joblib
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'animals' 


# Configure Gemini API
GEMINI_API_KEY = ""  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model (using the correct model name)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')


# Database configuration
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='animals'
)

mycursor = mydb.cursor()


def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()
    return


def retrivequery1(query, values):
    mycursor.execute(query, values)
    data = mycursor.fetchall()
    return data


def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']

        if password == c_password:
            query = "SELECT email FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])

            if email not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        query = "SELECT email FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email in email_data_list:
            query = "SELECT * FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password == password__data[0][3]:
                session["user_email"] = email
                session["user_id"] = password__data[0][0]
                session["user_name"] = password__data[0][1]

                return redirect("/home")
            return render_template('login.html', message="Invalid Password!!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')


########################################################################################################################
############################################## PREDICTION SECTION #####################################################
########################################################################################################################

# Load models and scaler once at startup
rf_model = joblib.load('vgg19_random_forest.joblib')
scaler = joblib.load('vgg19_scaler.joblib')

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
feature_extractor = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

class_names = ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation',
               'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture',
               'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture',
               'Spiral Fracture']


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def make_prediction(image_path):
    img = preprocess_image(image_path)
    features = feature_extractor.predict(img)
    features_scaled = scaler.transform(features)
    pred_idx = rf_model.predict(features_scaled)[0]
    return class_names[pred_idx]


@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        img = request.files["img"]
        file_name = img.filename
        img_path = os.path.join('static/saves_images/', file_name)
        img.save(img_path)

        predicted_class = make_prediction(img_path)

        return jsonify({
            "predicted_class": predicted_class,
            "file_name": file_name
        })

    return render_template('prediction.html')


########################################################################################################################
############################################## CHATBOT SECTION ########################################################
########################################################################################################################

@app.route('/chatbot')
def chatbot():
    # Check if user is logged in
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():
    # Check if user is logged in
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Initialize chat history in session if not exists
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Create chat session with history
        chat_session = gemini_model.start_chat(history=session['chat_history'])
        
        # Send message and get response
        response = chat_session.send_message(user_message)
        bot_response = response.text
        
        # Update chat history
        session['chat_history'].append({
            'role': 'user',
            'parts': [user_message]
        })
        session['chat_history'].append({
            'role': 'model',
            'parts': [bot_response]
        })
        
        # Keep only last 20 messages to avoid session overflow
        if len(session['chat_history']) > 20:
            session['chat_history'] = session['chat_history'][-20:]
        
        session.modified = True
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Check if user is logged in
    if 'user_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    session['chat_history'] = []
    session.modified = True
    return jsonify({'status': 'success', 'message': 'Chat history cleared'})


if __name__ == '__main__':
    app.run(debug=True)

