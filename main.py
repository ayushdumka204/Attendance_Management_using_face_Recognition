from flask import Flask, render_template, request, redirect, url_for
import csv
import os
import cv2

# Existing modules
import Capture_Image
import Train_Image
import Recognize

app = Flask(__name__)


# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')


# ================= BROWSER CAMERA UPLOAD (NEW 🔥) =================
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get('image')

        if not file:
            return "No image received ❌"

        os.makedirs("Uploads", exist_ok=True)
        filepath = os.path.join("Uploads", "capture.jpg")
        file.save(filepath)

        # 🔥 Face detection
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        return f"Image received ✅ Faces detected: {len(faces)}"

    except Exception as e:
        return f"Upload error: {str(e)}"


# ================= CAPTURE (OLD - DISABLED ON SERVER) =================
@app.route('/capture', methods=['POST'])
def capture():
    return "Use browser camera instead ❌"


# ================= TRAIN =================
@app.route('/train')
def train():
    try:
        Train_Image.TrainImages()
        return "Training Done ✅"
    except Exception as e:
        return f"Error in training: {str(e)}"


# ================= RECOGNIZE =================
@app.route('/recognize')
def recognize():
    try:
        return Recognize.recognize_attendence()
    except Exception as e:
        return f"Error in recognize: {str(e)}"


# ================= ATTENDANCE =================
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
            header = next(reader)
            data = list(reader)

        return render_template("attendance.html", header=header, data=data)

    except Exception as e:
        return f"Error loading attendance: {str(e)}"


# ================= RUN APP =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)