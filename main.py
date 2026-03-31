from flask import Flask, render_template, request, redirect, url_for
import csv, os

import Capture_Image
import Train_Image
import Recognize

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture():
    Id = request.form['id']
    name = request.form['name']
    msg = Capture_Image.takeImages(Id, name)
    return msg


@app.route('/train')
def train():
    Train_Image.TrainImages()
    return redirect(url_for('home'))


@app.route('/recognize')
def recognize():
    Recognize.recognize_attendence()
    return redirect(url_for('home'))


@app.route('/attendance')
def attendance():
    files = os.listdir("Attendance")
    if not files:
        return "No attendance found"

    latest = sorted(files)[-1]

    with open(os.path.join("Attendance", latest)) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)

    return render_template("attendance.html", header=header, data=data)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)