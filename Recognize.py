import cv2
import os
import time
import datetime
import pandas as pd


def recognize_attendence():
    # 🔴 Check: model exist karta hai ya nahi
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        return "Model not trained ❌"

    if not os.path.exists("StudentDetails/StudentDetails.csv"):
        return "Student data not found ❌"

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")

        faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        df = pd.read_csv("StudentDetails/StudentDetails.csv")

        # 🔴 Try opening camera
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            return "Camera not available ❌ (Server pe run ho raha hai)"

        col = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=col)

        while True:
            ret, img = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 100:
                    name = df.loc[df['Id'] == Id]['Name'].values
                    name = name[0] if len(name) > 0 else "Unknown"
                else:
                    name = "Unknown"

                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                attendance.loc[len(attendance)] = [Id, name, date, timeStamp]

                # 🟢 LOCAL me hi GUI chalega
                try:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, str(name), (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("Recognizing", img)
                except:
                    pass  # server pe ignore

            # Exit key (LOCAL)
            if cv2.waitKey(1) == 27:
                break

        cam.release()

        try:
            cv2.destroyAllWindows()
        except:
            pass

        # 🔴 Save attendance
        os.makedirs("Attendance", exist_ok=True)
        file_name = f"Attendance/Attendance_{date}.csv"
        attendance.to_csv(file_name, index=False)

        return f"Attendance saved ✅ ({file_name})"

    except Exception as e:
        return f"Error: {str(e)}"
# end