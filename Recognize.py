import datetime
import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import pandas as pd


def recognize_attendence():
    root = tk.Tk()
    root.title("Mark Attendance")
    root.geometry("400x300")

    # ===== Lecture duration input =====
    lecture = simpledialog.askstring("Input", "Enter Lecture Duration (HH:MM:SS)", parent=root)
    if not lecture:
        messagebox.showerror("Error", "Lecture duration required!")
        return

    # ===== Load models =====
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel" + os.sep + "Trainner.yml")

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    df = pd.read_csv("StudentDetails" + os.sep + "StudentDetails.csv")

    col_names = ['Id', 'Name', 'Date', 'Clock IN Time', 'Clock OUT Time', 'Duration', 'Status']
    attendance = pd.DataFrame(columns=col_names)

    cam = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    Id = None
    aa = None

    def save_attendance():
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = "Attendance" + os.sep + f"Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
        attendance.to_csv(fileName, index=False)
        messagebox.showinfo("Success", "Attendance Saved")

    def clock_in():
        nonlocal Id, aa, attendance
        if Id is None:
            messagebox.showwarning("Warning", "No face detected")
            return

        if messagebox.askyesno("Confirm", f"{aa} Clock IN?"):
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            attendance.loc[len(attendance)] = [Id, aa, date, timeStamp, '-', '-', '-']

    def clock_out():
        nonlocal attendance, Id, aa
        if Id is None:
            messagebox.showwarning("Warning", "No face detected")
            return

        if messagebox.askyesno("Confirm", f"{aa} Clock OUT?"):
            ts = time.time()
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            attendance.loc[attendance['Id'] == Id, 'Clock OUT Time'] = timeStamp

    def back():
        if messagebox.askyesno("Exit", "Save attendance before exit?"):
            save_attendance()
        cam.release()
        cv2.destroyAllWindows()
        root.destroy()

    # ===== Buttons =====
    tk.Button(root, text="Clock IN", width=20, command=clock_in).pack(pady=5)
    tk.Button(root, text="Clock OUT", width=20, command=clock_out).pack(pady=5)
    tk.Button(root, text="Save Attendance", width=20, command=save_attendance).pack(pady=5)
    tk.Button(root, text="Back", width=20, bg="red", fg="white", command=back).pack(pady=10)

    # ===== Camera Loop =====
    def update_frame():
        nonlocal Id, aa

        ret, im = cam.read()
        if not ret:
            root.after(10, update_frame)
            return

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            Id_pred, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 100:
                Id = Id_pred
                aa = df.loc[df['Id'] == Id]['Name'].values[0]
                label = f"{Id}-{aa}"
            else:
                Id = None
                aa = None
                label = "Unknown"

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(im, label, (x, y - 10), font, 0.8, (255, 255, 255), 2)

        cv2.imshow("Camera", im)

        if cv2.waitKey(1) == 27:  # ESC key
            back()
            return

        root.after(10, update_frame)

    update_frame()
    root.mainloop()

# end