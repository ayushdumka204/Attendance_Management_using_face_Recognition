from flask import Flask, render_template, request, jsonify
import csv
import os
import cv2
import pandas as pd
import datetime
import time
import Train_Image
from datetime import datetime
import pytz # Import this at the top of main.py

app = Flask(__name__)

# Folders setup (Automated)
folders = ["TrainingImage", "StudentDetails", "Attendance", "TrainingImageLabel"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

CSV_PATH = "StudentDetails/StudentDetails.csv"
ATTENDANCE_PATH = "Attendance/Attendance.csv"
MODEL_PATH = "TrainingImageLabel/Trainner.yml"

# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')

# ================= CHECK ID =================
@app.route('/check_id', methods=['POST'])
def check_id():
    new_id = request.form.get("id", "").strip()
    if not os.path.exists(CSV_PATH) or os.stat(CSV_PATH).st_size == 0:
        return jsonify({"exists": False})

    df = pd.read_csv(CSV_PATH)
    # Check if ID exists in the 'Id' column
    if not df.empty and str(new_id) in df['Id'].astype(str).values:
        return jsonify({"exists": True})
    
    return jsonify({"exists": False})

# ================= ADD PERSON & SAVE IMAGE =================
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        new_id = request.form.get("id", "").strip()
        new_name = request.form.get("name", "").strip()

        if not file or not new_id or not new_name:
            return "Missing Data! ID, Name, and Image are required. ❌", 400

        # Step 1: Temporary save for duplicate face check
        temp_path = "temp_check.jpg"
        file.save(temp_path)

        # Step 2: Recognition Logic (Duplicate Face Check)
        if os.path.exists(MODEL_PATH):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_PATH)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            img = cv2.imread(temp_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Lighting adjustment for registration
            gray = cv2.equalizeHist(gray)
            faces = faceCascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                predict_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                # LBPH: Confidence < 75 usually indicates a strong match
                if conf < 75: 
                    if os.path.exists(temp_path): os.remove(temp_path)
                    return f"❌ Face already registered with ID: {predict_id}", 400

        # Step 3: Final Save to Training folder
        final_filename = f"{new_name}.{new_id}.1.jpg"
        final_path = os.path.join("TrainingImage", final_filename)
        os.replace(temp_path, final_path)

        # Step 4: Update Student Database (CSV)
        file_exists = os.path.isfile(CSV_PATH) and os.stat(CSV_PATH).st_size > 0
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Id", "Name"])
            writer.writerow([new_id, new_name])

        return f"Person {new_name} added successfully! Training required. ✅", 200

    except Exception as e:
        return f"Server Error: {str(e)}", 500

# ================= TRAIN MODEL =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Model Trained Successfully! Recognition is now active. ✅"
    except Exception as e:
        return f"Training Error: {str(e)}", 500

@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type_ = request.form.get("type") # Expected: "IN" or "OUT" (Uppercase)

        if not file:
            return jsonify({"status": "error", "msg": "Image not found"})

        if not os.path.exists(MODEL_PATH):
            return jsonify({"status": "error", "msg": "Model is not trained!"})

        filepath = "temp_rec.jpg"
        file.save(filepath)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"status": "error", "msg": "No face detected!"})

        df_students = pd.read_csv(CSV_PATH)
        df_students['Id'] = df_students['Id'].astype(str)

        detected_name = "Unknown"
        detected_id = "N/A"

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 115: 
                row = df_students.loc[df_students['Id'] == str(Id)]
                if not row.empty:
                    detected_name = row['Name'].values[0]
                    detected_id = str(Id)

        if detected_name == "Unknown":
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"status": "error", "msg": "Face not recognized!"})

        # --- TIME SETUP (IST) ---
        IST = pytz.timezone('Asia/Kolkata')
        datetime_ist = datetime.now(IST)
        date = datetime_ist.strftime('%Y-%m-%d')
        timeStamp = datetime_ist.strftime('%H:%M:%S')

        # --- CSV HANDLING ---
        if not os.path.exists(ATTENDANCE_PATH):
            with open(ATTENDANCE_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID","Name","Date","IN","OUT","Duration"])

        rows = []
        with open(ATTENDANCE_PATH, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in list(reader) if len(row) >= 3] # Valid rows only

        found = False
        msg = ""
        
        # Optimized Loop for OUT entry
        for r in rows:
            # Match ID and Date (Using strip() to avoid hidden spaces)
            if str(r[0]).strip() == str(detected_id).strip() and str(r[2]).strip() == str(date).strip():
                found = True
                if type_ == "OUT":
                    # Check if OUT column (index 4) is empty or has a placeholder
                    if len(r) < 5 or r[4] == "" or r[4] == " ":
                        r[4] = timeStamp
                        try:
                            # Ensuring indices exist before calculation
                            t1 = datetime.strptime(r[3], "%H:%M:%S")
                            t2 = datetime.strptime(r[4], "%H:%M:%S")
                            r[5] = str(t2 - t1).split(".")[0]
                            msg = f"Clock OUT recorded for {detected_name}!"
                        except Exception as e:
                            r[5] = "00:00:00"
                            msg = "OUT recorded (Duration calc error)."
                    else:
                        msg = "Already clocked OUT for today."
                elif type_ == "IN":
                    msg = f"IN entry already exists for {detected_name}."
                break

        if not found:
            if type_ == "IN":
                # Create a clean row with all 6 columns
                rows.append([detected_id, detected_name, date, timeStamp, "", ""])
                msg = f"Welcome {detected_name}, IN recorded!"
            else:
                msg = "No IN entry found! Please Clock IN first."

        # Rewrite CSV with fixed data
        with open(ATTENDANCE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"status": "success", "name": detected_name, "id": detected_id, "msg": msg})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"status": "error", "msg": "Internal Server Error!"})

# ================= VIEW ATTENDANCE =================
@app.route('/attendance')
def attendance():
    try:
        if not os.path.exists(ATTENDANCE_PATH):
            return render_template("attendance.html", header=[], data=[])
        with open(ATTENDANCE_PATH, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            data = [row for row in list(reader) if row]
        return render_template("attendance.html", header=header, data=data[::-1]) # Descending order
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Ensure port is handled for cloud/local deployments
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)