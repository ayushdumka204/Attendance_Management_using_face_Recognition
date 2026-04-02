# 🎯 Attendance System Using Face Recognition (Web-Based)

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0.1-green.svg)](https://opencv.org/)

A modern, full-stack web application that automates attendance tracking using **Real-time Facial Recognition**. This project replaces traditional desktop GUIs with a sleek, responsive **Web Interface** accessible via browsers.

---

## 📺 Project Overview
This system captures live video feeds through the browser, processes them using **OpenCV** on the backend, and identifies registered users. It maintains a secure log of **Clock-In** and **Clock-Out** timings, calculating total duration automatically.

---

## 📖 Key Functionalities

### 1. Web-Based Dashboard
* **Dynamic UI:** Built with HTML5 and CSS3 for a responsive experience on both Desktop and Mobile.
* **Live Camera Feed:** Integrated using JavaScript `navigator.mediaDevices` API.

### 2. Smart Attendance Logic
* **Clock IN/OUT:** Recognizes face and logs entry/exit time in a CSV database.
* **Duration Calculation:** Automatically calculates the time spent based on IN and OUT logs.
* **IST Time Synchronization:** Configured to record time in Indian Standard Time, even on cloud servers like Render.

### 3. User Management
* **Fast Registration:** Add new users by capturing photos directly from the UI.
* **Duplicate Protection:** Prevents registering the same face or ID twice.
* **AI Training:** One-click training that uses the **LBPH (Local Binary Pattern Histogram)** algorithm.

---

## 🛠 Tech Stack

| Category | Tools / Technologies |
| :--- | :--- |
| **Backend** | Python (Flask Framework) |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **Computer Vision** | OpenCV (Haar Cascades, LBPH) |
| **Data Storage** | Pandas, CSV |
| **Time Management** | Pytz (IST Support) |
| **Deployment** | Render / Localhost |

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.9+
* Webcam access

### 2. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt