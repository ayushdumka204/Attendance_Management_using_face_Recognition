from flask import Flask, render_template, request, jsonify
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

        if not file or not Id or not name:
            return "Missing data ❌"

        folder = "TrainingImage"
        os.makedirs(folder, exist_ok=True)

        # 🔥 MULTIPLE IMAGE SAVE FIX
        count = len([f for f in os.listdir(folder) if f.startswith(f"{name}.{Id}")])
        filepath = os.path.join(folder, f"{name}.{Id}.{count+1}.jpg")
        file.save(filepath)

        # 🔥 SAVE STUDENT ONLY ONCE
        os.makedirs("StudentDetails", exist_ok=True)
        file_path = "StudentDetails/StudentDetails.csv"

        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                csv.writer(f).writerow(["Id", "Name"])

        rows = list(csv.reader(open(file_path)))
        exists = any(row[0] == Id for row in rows[1:])

        if not exists:
            with open(file_path, "a", newline="") as f:
                csv.writer(f).writerow([Id, name])

        return "Person Added ✅"

    except Exception as e:
        return str(e)


# ================= TRAIN =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Model Trained Successfully ✅"
    except Exception as e:
        return str(e)


# ================= RECOGNIZE =================
@app.route('/recognize_upload', methods=['POST'])
def recognize_upload():
    try:
        file = request.files.get('image')
        type = request.form.get("type")

        if not file:
            return jsonify({"status": "error", "msg": "No image"})

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
            return jsonify({"status": "error", "msg": "No face detected"})

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        file_path = "Attendance/Attendance.csv"
        os.makedirs("Attendance", exist_ok=True)

        # CREATE FILE
        if not os.path.exists(file_path):
            with open(file_path, "w", newline="") as f:
                csv.writer(f).writerow(["ID","Name","Date","IN","OUT","Duration"])

        rows = list(csv.reader(open(file_path)))
        header, data = rows[0], rows[1:]

        detected_name = "Unknown"
        detected_id = ""

        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:
                name = df.loc[df['Id'] == Id]['Name'].values[0]
            else:
                name = "Unknown"

            detected_name = name
            detected_id = Id

            found = False

            for row in data:
                if row[0] == str(Id) and row[2] == date:

                    # ✅ CLOCK IN only once
                    if type == "IN" and row[3] == "":
                        row[3] = timeStamp

                    # ✅ CLOCK OUT update
                    elif type == "OUT":
                        row[4] = timeStamp
                        try:
                            t1 = datetime.datetime.strptime(row[3], "%H:%M:%S")
                            t2 = datetime.datetime.strptime(row[4], "%H:%M:%S")
                            row[5] = str(t2 - t1)
                        except:
                            row[5] = ""

                    found = True

            # ✅ NEW ENTRY
            if not found and type == "IN":
                data.append([Id, name, date, timeStamp, "", ""])

        # SAVE
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        return jsonify({
            "status": "success",
            "name": detected_name,
            "id": detected_id
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)})


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