from flask import Flask, render_template, request, redirect, url_for
import csv
import os

# 🔴 IMPORTANT: ye imports tabhi kaam karenge jab tu GUI functions hata de
# warna Render pe crash ho sakta hai
import Capture_Image
import Train_Image
import Recognize

app = Flask(__name__)


# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')


# ================= CAPTURE =================
@app.route('/capture', methods=['POST'])
def capture():
    try:
        Id = request.form.get('id')
        name = request.form.get('name')

        msg = Capture_Image.takeImages(Id, name)
        return str(msg)

    except Exception as e:
        return f"Error in capture: {str(e)}"


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
        Recognize.recognize_attendence()
        return "Recognition Done ✅"

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