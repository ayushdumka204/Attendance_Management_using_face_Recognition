import csv
import cv2
import os


def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


def takeImages(Id, name):
    # 🔴 Validation
    if not (is_number(Id) and name.isalpha()):
        return "Invalid Input ❌"

    try:
        # 🔴 Try opening camera
        cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            return "Camera not available ❌ (Server pe run ho raha hai)"

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

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
                    f"TrainingImage/{name}.{Id}.{sampleNum}.jpg",
                    gray[y:y+h, x:x+w]
                )

                # 🟢 LOCAL me rectangle dikhega
                try:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("Capture Face", img)
                except:
                    pass  # server pe ignore

            # Exit condition
            if sampleNum > 100:
                break

            try:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass  # server pe ignore

        cam.release()

        try:
            cv2.destroyAllWindows()
        except:
            pass

        # 🔴 Save student details
        with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([Id, name])

        return f"Images Saved Successfully ✅ (Total: {sampleNum})"

    except Exception as e:
        return f"Error in capture: {str(e)}"
# end