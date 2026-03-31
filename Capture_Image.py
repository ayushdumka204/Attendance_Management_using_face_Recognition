import csv
import tkinter as tk
from tkinter import messagebox
import cv2
import os


# ===== check number =====
def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


# ===== main function =====
def takeImages():
    root = tk.Tk()
    root.title("Student Details")
    root.geometry("400x250")

    tk.Label(root, text="ID:", font=("Helvetica", 12)).pack(pady=5)
    id_entry = tk.Entry(root)
    id_entry.pack()

    tk.Label(root, text="Name:", font=("Helvetica", 12)).pack(pady=5)
    name_entry = tk.Entry(root)
    name_entry.pack()

    def submit():
        Id = id_entry.get()
        name = name_entry.get()

        if not (is_number(Id) and name.isalpha()):
            messagebox.showerror("Error", "Enter valid ID and Name")
            return

        root.destroy()

        # ===== Start Camera =====
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        sampleNum = 0

        os.makedirs("TrainingImage", exist_ok=True)
        os.makedirs("StudentDetails", exist_ok=True)

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                sampleNum += 1

                cv2.imwrite(
                    "TrainingImage" + os.sep + name + "." + Id + "." + str(sampleNum) + ".jpg",
                    gray[y:y + h, x:x + w]
                )

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Capture Face", img)

            # press q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if sampleNum >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        # ===== Save to CSV =====
        with open("StudentDetails" + os.sep + "StudentDetails.csv", 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])

        messagebox.showinfo("Success", f"Images saved for ID: {Id}, Name: {name}")

    tk.Button(root, text="Submit", command=submit, width=15).pack(pady=10)
    tk.Button(root, text="Cancel", command=root.destroy, width=15).pack()

    root.mainloop()


# end