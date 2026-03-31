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

        # Save student
        os.makedirs("StudentDetails", exist_ok=True)
        with open("StudentDetails/StudentDetails.csv", "a+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([Id, name])

        return f"Person Added ✅"

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


# ================= RECOGNIZE =================
@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type = request.form.get("type")

        if not file:
            return "No image ❌"

        filepath = "temp.jpg"
        file.save(filepath)

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

        file_path = "Attendance/Attendance.csv"
        os.makedirs("Attendance", exist_ok=True)

        # CREATE FILE
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID","Name","Date","IN","OUT","Duration"])

        # LOAD DATA
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        data = rows[1:]

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                name = df.loc[df['Id'] == Id]['Name'].values[0]
            else:
                name = "Unknown"

            found = False

            for row in data:
                if row[0] == str(Id) and row[2] == date:
                    if type == "OUT":
                        row[4] = timeStamp

                        # duration
                        try:
                            t1 = datetime.datetime.strptime(row[3], "%H:%M:%S")
                            t2 = datetime.datetime.strptime(row[4], "%H:%M:%S")
                            row[5] = str(t2 - t1)
                        except:
                            row[5] = ""

                        found = True

            if type == "IN" and not found:
                data.append([Id, name, date, timeStamp, "", ""])

        # SAVE BACK
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        return f"{type} Marked ✅"

    except Exception as e:
        return str(e)


# ================= VIEW =================
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


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)