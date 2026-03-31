import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import Capture_Image
import Train_Image
import Recognize
import view_attendance


def open_attendance():
    path = os.path.realpath("Attendance")
    os.startfile(path)


def open_students():
    path = os.path.realpath("StudentDetails")
    os.startfile(path)


def add_person():
    Capture_Image.takeImages()


def train_images():
    Train_Image.TrainImages()


def mark_attendance():
    Recognize.recognize_attendence()


def view_att():
    view_attendance.vcsv()


def mainMenu():
    root = tk.Tk()
    root.title("Face Attendance Recognition Program")
    root.geometry("900x700")
    root.configure(bg="white")

    # ===== MENU =====
    menu_bar = tk.Menu(root)

    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open Attendance Folder", command=open_attendance)
    file_menu.add_command(label="Open Student Records", command=open_students)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    menu_bar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menu_bar)

    # ===== TITLE =====
    title = tk.Label(
        root,
        text="Attendance Capture System Using Face Recognition",
        font=("Helvetica", 20, "bold"),
        bg="white"
    )
    title.pack(pady=20)

    # ===== IMAGE =====
    try:
        img = Image.open("Images/Facial_Recognition_logo.png")
        img = img.resize((500, 300))
        photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(root, image=photo, bg="white")
        img_label.image = photo
        img_label.pack(pady=10)
    except:
        tk.Label(root, text="Image not found", bg="white").pack()

    # ===== BUTTONS =====
    tk.Button(root, text="Mark Attendance", width=30, height=2, bg="#303030", fg="white",
              command=mark_attendance).pack(pady=10)

    tk.Button(root, text="Add Person", width=30, height=2, bg="#303030", fg="white",
              command=add_person).pack(pady=5)

    tk.Button(root, text="Train Images", width=30, height=2, bg="#303030", fg="white",
              command=train_images).pack(pady=5)

    tk.Button(root, text="View Attendance", width=30, height=2, bg="#303030", fg="white",
              command=view_att).pack(pady=5)

    tk.Button(root, text="Quit", width=30, height=2, bg="red", fg="white",
              command=root.quit).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    mainMenu()

# end