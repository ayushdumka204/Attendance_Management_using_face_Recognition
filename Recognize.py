import cv2, os, time, datetime
import pandas as pd

def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)

    col = ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns=col)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,1.2,5)

        for(x,y,w,h) in faces:
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

            if conf < 100:
                name = df.loc[df['Id']==Id]['Name'].values[0]
            else:
                name = "Unknown"

            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            attendance.loc[len(attendance)] = [Id,name,date,timeStamp]

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,str(name),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        cv2.imshow("Recognizing", img)

        if cv2.waitKey(1)==27:
            break

    cam.release()
    cv2.destroyAllWindows()

    os.makedirs("Attendance", exist_ok=True)
    attendance.to_csv(f"Attendance/Attendance_{date}.csv", index=False)
# end