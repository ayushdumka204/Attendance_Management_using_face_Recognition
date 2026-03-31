import csv, cv2, os

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def takeImages(Id, name):
    if not (is_number(Id) and name.isalpha()):
        return "Invalid Input"

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
                f"TrainingImage/{name}.{Id}.{sampleNum}.jpg",
                gray[y:y+h, x:x+w]
            )

            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow("Capture Face", img)

        if cv2.waitKey(1) & 0xFF == ord('q') or sampleNum > 100:
            break

    cam.release()
    cv2.destroyAllWindows()

    with open("StudentDetails/StudentDetails.csv", 'a+', newline='') as f:
        csv.writer(f).writerow([Id, name])

    return "Images Saved Successfully"
# end