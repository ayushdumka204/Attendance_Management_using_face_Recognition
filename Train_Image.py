import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox


# ===== Get images and labels =====
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')

        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces.append(imageNp)
        Ids.append(Id)

    return faces, Ids


# ===== Train images =====
def TrainImages():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        faces, Ids = getImagesAndLabels("TrainingImage")

        if len(faces) == 0:
            messagebox.showerror("Error", "No images found for training")
            return

        recognizer.train(faces, np.array(Ids))

        os.makedirs("TrainingImageLabel", exist_ok=True)
        recognizer.save("TrainingImageLabel" + os.sep + "Trainner.yml")

        messagebox.showinfo("Success", "All Images Trained Successfully")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ===== Optional standalone test =====
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    TrainImages()

# end