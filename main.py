from flask import Flask, render_template, request, jsonify
import csv
import os
import cv2
import pandas as pd
import datetime
import time
import Train_Image

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
            return "Missing Data! ID, Name aur Image zaruri hain. ❌", 400

        temp_path = "temp_check.jpg"
        file.save(temp_path)

        if os.path.exists(MODEL_PATH):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_PATH)
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            img = cv2.imread(temp_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Lighting Fix for registration
            gray = cv2.equalizeHist(gray)
            faces = faceCascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                predict_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 75: 
                    if os.path.exists(temp_path): os.remove(temp_path)
                    return f"❌ Face already registered with ID: {predict_id}", 400

        final_filename = f"{new_name}.{new_id}.1.jpg"
        final_path = os.path.join("TrainingImage", final_filename)
        os.replace(temp_path, final_path)

        file_exists = os.path.isfile(CSV_PATH) and os.stat(CSV_PATH).st_size > 0
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Id", "Name"])
            writer.writerow([new_id, new_name])

        return f"Person {new_name} added successfully! ✅", 200

    except Exception as e:
        return f"Server Error: {str(e)}", 500

# ================= TRAIN MODEL =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Model Trained Successfully! ✅"
    except Exception as e:
        return f"Training Error: {str(e)}", 500

# ================= RECOGNIZE & ATTENDANCE LOGIC =================
@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type_ = request.form.get("type") 

        if not file:
            return jsonify({"status": "error", "msg": "Image not found"})

        if not os.path.exists(MODEL_PATH):
            return jsonify({"status": "error", "msg": "Model not trained!"})

        filepath = "temp_rec.jpg"
        file.save(filepath)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 🟢 LIGHTING IMPROVEMENT (IMPORTANT)
        gray = cv2.equalizeHist(gray)
        
        # 🟢 Sensitive Detection (1.1 instead of 1.2)
        faces = faceCascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"status": "error", "msg": "Face detect nahi hua!"})

        df_students = pd.read_csv(CSV_PATH)
        df_students['Id'] = df_students['Id'].astype(str)

        detected_name = "Unknown"
        detected_id = "N/A"

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print(f"Recognized ID: {Id}, Confidence: {conf}") # Console check
            
            # Confidence adjust: LBPH mein 100-120 ke beech threshold acha hota hai
            if conf < 115: 
                row = df_students.loc[df_students['Id'] == str(Id)]
                if not row.empty:
                    detected_name = row['Name'].values[0]
                    detected_id = str(Id)

        if detected_name == "Unknown":
            if os.path.exists(filepath): os.remove(filepath)
            return jsonify({"status": "error", "msg": "Chehra pehchaan mein nahi aaya!"})

        # Attendance CSV Handling
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        if not os.path.exists(ATTENDANCE_PATH):
            with open(ATTENDANCE_PATH, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID","Name","Date","IN","OUT","Duration"])

        rows = []
        with open(ATTENDANCE_PATH, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [row for row in list(reader) if row] # Skip empty rows

        found = False
        msg = ""
        for r in rows:
            if r[0] == str(detected_id) and r[2] == date:
                found = True
                if type_ == "IN":
                    msg = f"{detected_name}, IN pehle se marked hai."
                elif type_ == "OUT":
                    if r[4] == "":
                        r[4] = timeStamp
                        t1 = datetime.datetime.strptime(r[3], "%H:%M:%S")
                        t2 = datetime.datetime.strptime(r[4], "%H:%M:%S")
                        r[5] = str(t2 - t1).split(".")[0]
                        msg = f"OUT marked! Duration: {r[5]}"
                    else:
                        msg = "OUT pehle hi marked hai."
                break

        if not found:
            if type_ == "IN":
                rows.append([detected_id, detected_name, date, timeStamp, "", ""])
                msg = f"Welcome {detected_name}, IN marked!"
            else:
                msg = "Pehle IN mark karein!"

        with open(ATTENDANCE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"status": "success", "name": detected_name, "id": detected_id, "msg": msg})

    except Exception as e:
        return jsonify({"status": "error", "msg": f"System Error: {str(e)}"})

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
        return render_template("attendance.html", header=header, data=data[::-1])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)