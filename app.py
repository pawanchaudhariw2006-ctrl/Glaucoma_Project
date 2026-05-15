import os
import numpy as np
import random
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'glaucoma_model.h5'
UPLOAD_FOLDER = 'static/uploads'
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "EyeHealthDB"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LOAD MODEL
print("Loading AI Model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# DATABASE CONNECTION
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db["patient_records"]
    print("✅ Connected to MongoDB.")
except:
    print("⚠️ Warning: MongoDB not connected.")

# ==========================================
# DYNAMIC ADVICE LOGIC (THE NEW PART)
# ==========================================
def get_dynamic_advice(is_glaucoma, confidence):
    
    # 1. POOL OF RANDOM TIPS (To make every result look unique)
    general_tips = [
        "Stay hydrated: Drinking water in small amounts frequently is better than large amounts at once.",
        "Sleep with your head elevated slightly to reduce intraocular pressure (IOP).",
        "Limit caffeine: High caffeine intake can temporarily raise eye pressure.",
        "Exercise safely: Avoid head-down yoga positions (like downward dog) which increase eye pressure.",
        "Eat Omega-3s: Include fish, walnuts, or flaxseeds in your diet.",
        "Protect eyes from UV: Always wear sunglasses when outdoors.",
        "Follow the 20-20-20 rule: Every 20 mins, look 20 feet away for 20 seconds."
    ]
    
    # Select 3 random tips for this specific patient
    selected_tips = random.sample(general_tips, 3)
    home_care_text = "\n".join([f"{i+1}. {tip}" for i, tip in enumerate(selected_tips)])

    # 2. SEVERITY BASED ADVICE
    if is_glaucoma:
        if confidence > 90:
            # SEVERE CASE
            return {
                "status": "High Risk: Glaucoma Detected",
                "color": "#d32f2f", # Dark Red
                "advice": "CRITICAL: The AI detected signs consistent with advanced Glaucoma. Immediate consultation with a specialist is required. Ask for an OCT scan and Visual Field Test.",
                "home_care": "1. Do not miss any prescribed eye drops.\n2. Avoid heavy lifting or straining.\n" + home_care_text
            }
        elif confidence > 75:
            # MODERATE CASE
            return {
                "status": "Glaucoma Detected",
                "color": "#ff5722", # Orange-Red
                "advice": "The scan indicates potential optic nerve damage. Schedule an appointment within the next 7 days. Early treatment can stop vision loss.",
                "home_care": home_care_text
            }
        else:
            # BORDERLINE CASE (50-75%)
            return {
                "status": "Suspected / Borderline",
                "color": "#ff9800", # Orange
                "advice": "Results are inconclusive but show risk factors. It might be early-stage Glaucoma or just Ocular Hypertension. Monitoring is essential.",
                "home_care": "1. Monitor your peripheral vision.\n" + home_care_text
            }
    else:
        # NORMAL CASE
        if confidence > 90:
            return {
                "status": "Healthy Eye",
                "color": "#388e3c", # Green
                "advice": "Excellent. Your optic nerve looks healthy with no signs of cupping. Keep up the good work!",
                "home_care": home_care_text
            }
        else:
            return {
                "status": "Likely Normal",
                "color": "#8bc34a", # Light Green
                "advice": "The eye appears normal, though some shadows were unclear. Ensure the image quality is good next time. Regular checkups are still recommended.",
                "home_care": home_care_text
            }

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_data = None
    
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict
            score = model.predict(img_array)[0][0]
            
            # Logic: < 0.5 is Glaucoma, > 0.5 is Normal
            is_glaucoma = score < 0.5
            
            # Calculate Confidence %
            if is_glaucoma:
                confidence = (1 - score) * 100
            else:
                confidence = score * 100
            
            # GET DYNAMIC ADVICE
            info = get_dynamic_advice(is_glaucoma, confidence)
            
            prediction_data = {
                "result": info['status'],
                "confidence": round(confidence, 2),
                "advice": info['advice'],
                "home_care": info['home_care'],
                "color": info['color'],
                "image_path": filepath
            }

            try:
                record = prediction_data.copy()
                record['date'] = datetime.now()
                collection.insert_one(record)
            except:
                pass

    return render_template('index.html', data=prediction_data)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)