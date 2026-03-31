from flask import Flask, render_template, request, jsonify
import csv
import os
import cv2
import pandas as pd
import datetime
import time
import Train_Image

app = Flask(__name__)

# Folders setup
if not os.path.exists("TrainingImage"): os.makedirs("TrainingImage")
if not os.path.exists("StudentDetails"): os.makedirs("StudentDetails")
if not os.path.exists("Attendance"): os.makedirs("Attendance")

# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')

# ================= ADD PERSON & SAVE IMAGE =================
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        Id = request.form.get("id")
        name = request.form.get("name")

        if not file or not Id or not name:
            return "ID, Name ya Image missing hai! ❌"

        # 1. Image Save karna (Yehi tera DB hai filhal)
        filepath = os.path.join("TrainingImage", f"{name}.{Id}.1.jpg")
        file.save(filepath)

        # 2. CSV update karna
        csv_path = "StudentDetails/StudentDetails.csv"
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a+", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Id", "Name"])
            writer.writerow([Id, name])

        return "Person Added & Image Saved ✅"

    except Exception as e:
        return f"Error: {str(e)}"

# ================= TRAIN MODEL =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Model Trained Successfully ✅"
    except Exception as e:
        return str(e)

# ================= RECOGNIZE & MARK ATTENDANCE =================
@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type_ = request.form.get("type") # IN / OUT

        if not file:
            return jsonify({"status": "error", "msg": "No image"})

        filepath = "temp.jpg"
        file.save(filepath)

        # Recognizer setup
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
            return jsonify({"status": "error", "msg": "No face detected"})

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        detected_name = "Unknown"
        detected_id = "N/A"

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                row = df.loc[df['Id'] == Id]
                if not row.empty:
                    detected_name = row['Name'].values[0]
                    detected_id = str(Id)
            
            # Attendance Logic
            file_path = "Attendance/Attendance.csv"
            if not os.path.exists(file_path):
                with open(file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ID","Name","Date","IN","OUT","Duration"])

            with open(file_path, "r") as f:
                data = list(csv.reader(f))
            
            header = data[0]
            rows = data[1:]
            found = False

            for r in rows:
                if r[0] == str(detected_id) and r[2] == date:
                    if type_ == "IN" and r[3] == "":
                        r[3] = timeStamp
                    elif type_ == "OUT":
                        r[4] = timeStamp
                        try:
                            t1 = datetime.datetime.strptime(r[3], "%H:%M:%S")
                            t2 = datetime.datetime.strptime(r[4], "%H:%M:%S")
                            r[5] = str(t2 - t1)
                        except: pass
                    found = True

            if not found and type_ == "IN" and detected_name != "Unknown":
                rows.append([detected_id, detected_name, date, timeStamp, "", ""])

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

        return jsonify({
            "status": "success",
            "name": detected_name,
            "id": detected_id
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})

# ================= VIEW ATTENDANCE =================
@app.route('/attendance')
def attendance():
    try:
        file_path = "Attendance/Attendance.csv"
        if not os.path.exists(file_path):
            return render_template("attendance.html", header=[], data=[])
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            data = list(reader)
        return render_template("attendance.html", header=header, data=data)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)