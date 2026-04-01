import cv2
import os
import time
import datetime
import pandas as pd


def recognize_attendence():
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        return "Model not trained ❌"

    if not os.path.exists("StudentDetails/StudentDetails.csv"):
        return "Student data not found ❌"

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 🟢 FIX 1: ID ko string ki tarah load karo taaki matching me error na aaye
        df = pd.read_csv("StudentDetails/StudentDetails.csv", dtype={'Id': str})

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return "Camera not available ❌"

        col = ['Id', 'Name', 'Date', 'Time']
        # Hum set use karenge taaki ek hi bande ki attendance baar-baar duplicate na ho
        found_students = set() 
        attendance_list = []

        while True:
            ret, img = cam.read()
            if not ret: break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                predict_id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 100:
                    # 🟢 FIX 2: String matching for ID
                    row = df.loc[df['Id'] == str(predict_id)]
                    if not row.empty:
                        name = row['Name'].values[0]
                        student_id = str(predict_id)
                    else:
                        name = "Unknown"
                        student_id = "Unknown"
                else:
                    name = "Unknown"
                    student_id = "Unknown"

                # 🟢 FIX 3: Duplicate entry rokne ke liye
                if student_id != "Unknown" and student_id not in found_students:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance_list.append([student_id, name, date, timeStamp])
                    found_students.add(student_id)

                # GUI update
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Recognizing (Press ESC to save)", img)
            if cv2.waitKey(1) == 27: break

        cam.release()
        cv2.destroyAllWindows()

        # 🟢 Save logic
        if attendance_list:
            os.makedirs("Attendance", exist_ok=True)
            # Purani attendance file check karo agar aaj ki file pehle se hai
            today_date = datetime.datetime.now().strftime('%Y-%m-%d')
            file_name = f"Attendance/Attendance_{today_date}.csv"
            
            new_df = pd.DataFrame(attendance_list, columns=col)
            new_df.to_csv(file_name, index=False)
            return f"Attendance saved ✅ ({file_name})"
        else:
            return "No attendance recorded ❌"

    except Exception as e:
        return f"Error: {str(e)}"