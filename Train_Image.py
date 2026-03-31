import cv2, os, numpy as np
from PIL import Image

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces, Ids = [], []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage,'uint8')

        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces:
        print("No images found")
        return

    recognizer.train(faces, np.array(Ids))

    os.makedirs("TrainingImageLabel", exist_ok=True)
    recognizer.save("TrainingImageLabel/Trainner.yml")

    print("Training Done")
# end