from flask import Flask, render_template, request
import csv
import os
import cv2
import pandas as pd
import datetime
import time

import Train_Image

app = Flask(__name__)


# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')


# ================= ADD PERSON =================
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')
        Id = request.form.get("id")
        name = request.form.get("name")

        if not file:
            return "No image received ❌"

        if not Id or not name:
            return "ID or Name missing ❌"

        os.makedirs("TrainingImage", exist_ok=True)

        filepath = os.path.join("TrainingImage", f"{name}.{Id}.1.jpg")
        file.save(filepath)

        # Face detect
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Save student
        os.makedirs("StudentDetails", exist_ok=True)
        with open("StudentDetails/StudentDetails.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([Id, name])

        return f"Person Added ✅ | Faces detected: {len(faces)}"

    except Exception as e:
        return f"Upload error: {str(e)}"


# ================= TRAIN =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Model Trained Successfully ✅"
    except Exception as e:
        return f"Training Error: {str(e)}"


# ================= RECOGNIZE (IN / OUT) =================
@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type = request.form.get("type")  # IN / OUT

        if not file:
            return "No image received ❌"

        filepath = "temp.jpg"
        file.save(filepath)

        if not os.path.exists("TrainingImageLabel/Trainner.yml"):
            return "Model not trained ❌"

        if not os.path.exists("StudentDetails/StudentDetails.csv"):
            return "Student data missing ❌"

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")

        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        df = pd.read_csv("StudentDetails/StudentDetails.csv")

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) == 0:
            return "No face detected ❌"

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        file_path = f"Attendance/Attendance_{date}.csv"
        os.makedirs("Attendance", exist_ok=True)

        # Load existing data
        rows = []
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)

        # remove header if exists
        if rows and rows[0][0] == "ID":
            rows = rows[1:]

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                name = df.loc[df['Id'] == Id]['Name'].values
                name = name[0] if len(name) > 0 else "Unknown"
            else:
                name = "Unknown"

            found = False

            for row in rows:
                if row[0] == str(Id) and row[2] == date:
                    if type == "OUT":
                        row[4] = timeStamp

                        # duration calculate
                        try:
                            t1 = datetime.datetime.strptime(row[3], "%H:%M:%S")
                            t2 = datetime.datetime.strptime(row[4], "%H:%M:%S")
                            row[5] = str(t2 - t1)
                        except:
                            row[5] = ""

                        found = True

            # NEW ENTRY (IN)
            if type == "IN" and not found:
                rows.append([Id, name, date, timeStamp, "", ""])

        # SAVE FILE
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID","Name","Date","IN","OUT","Duration"])
            writer.writerows(rows)

        return f"Attendance Marked ✅ ({type})"

    except Exception as e:
        return f"Recognition Error: {str(e)}"


# ================= VIEW =================
@app.route('/attendance')
def attendance():
    try:
        if not os.path.exists("Attendance"):
            return "Attendance folder not found ❌"

        files = os.listdir("Attendance")
        if not files:
            return "No attendance found"

        latest = sorted(files)[-1]

        with open(os.path.join("Attendance", latest)) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            data = list(reader)

        return render_template("attendance.html", header=header, data=data)

    except Exception as e:
        return f"Error loading attendance: {str(e)}"


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)