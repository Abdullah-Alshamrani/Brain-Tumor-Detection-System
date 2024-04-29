from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import requests
import openai


app = Flask(__name__)


MODEL_PATH = 'brainTumorCnn.keras'
model = load_model(MODEL_PATH)


# Set the path to the directory containing the images
IMAGE_DIR = '/Users/abdullah/Desktop/GradProjectCode'  # Change this to the directory path


@app.route('/images/<filename>')
def image(filename):
   return send_from_directory(IMAGE_DIR, filename)


@app.route('/', methods=['GET'])
def index():
   image_name = 'IMAGEOFBRAIN.avif'  # Change this to your actual image file
   html_content = '''
   <!DOCTYPE html>
   <html>
   <head>
       <title>Brain Tumor Prediction</title>
       <h1>Welcome to The Brain Tumor Detection System</h1>
   </head>
   <body>
       <div class="container">
           <img src="/images/''' + image_name + '''" alt="Brain Image" class="brain-image"/>
           <form method="post" action="/predict" enctype="multipart/form-data">
               <input type="file" name="file" accept="image/*" required>
               <input type="submit" value="Upload and Predict">
           </form>
       </div>
   </body>
   <style>
   body {{
       font-family: Arial, sans-serif;
       margin: 0;
       padding: 0;
       background: gray;
       display: flex;
       justify-content: center;
       align-items: center;
       height: 100vh;
   }}


   .container {{
       background: white;
       padding: 20px;
       border-radius: 10px;
       box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
       text-align: center;
       width: 300px;  /* Adjust as needed */
   }}


   .brain-image {{
       max-width: 100%;  /* Adjust as needed */
       height: auto;
       margin-bottom: 20px;
   }}


   form {{
       display: inline-block;
       margin-top: 20px;
   }}


   input[type=file] {{
       margin-bottom: 10px;
   }}


   input[type=submit] {{
       cursor: pointer;
       background: red;
       color: white;
       border: none;
       padding: 10px 20px;
       border-radius: 5px;
       display: inline-block;
   }}


   input[type=submit]:hover {{
       background: darkred;
   }}
   </style>
   </html>
   '''
   return html_content


@app.route('/predict', methods=['POST'])
def predict():
   if 'file' not in request.files:
       return jsonify({'error': 'No file part'}), 400
   file = request.files['file']
   if file.filename == '':
       return jsonify({'error': 'No selected file'}), 400
   img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
   img = cv2.resize(img, (64, 64))
   img = img / 255.0
   img = img.reshape(1, 64, 64, 1)
   prediction = model.predict(img)
   result = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  
   # Generating advice based on the result using AI through OpenAI API KEY
   api_key = "sk-proj-XXXXXXXXX"


   if result == 'Positive':
       advice = ask_openai_for_advice(api_key, result)
   else:
       return "You teseted Negative."




  
   return jsonify({'prediction': result, 'advice': advice})




def ask_openai_for_advice(api_key, keyword):
   openai.api_key = api_key


   # Setup the health advice query using the keyword
   query = f"Act as a health advisor, I have been diagnosed with a brain tumor and it is {keyword}. What should I do next in five sentences?"


   try:
       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[{"role": "user", "content": query}]
       )


       if response.choices:
           return response.choices[0].message['content']
       else:
           return "No response received from API."


   except Exception as e:
       return str(e)


if __name__ == '__main__':
   app.run(debug=True)


