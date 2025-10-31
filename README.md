# 🚗 Drowsiness Alerting System

This project detects driver drowsiness and yawning using **OpenCV**, **Dlib**, and **facial landmarks**.  
If the driver appears sleepy or yawns frequently, the system triggers an **alarm sound** and sends a signal to **Arduino** to activate buzzer and LED alerts.

---

## 🧠 Features
- Real-time **eye and mouth monitoring**
- **Sound alarm** and **Arduino buzzer** integration
- Works with any standard webcam
- Adjustable thresholds for EAR and yawn detection

---

## 🧩 Requirements

Before running the project, make sure you have the following installed:

- **Python 3.7 or higher**
- **OpenCV**
- **dlib**
- **imutils**
- **playsound**
- **NumPy**
- **pyserial** (for Arduino communication)
- A **webcam** connected to your system
- **Arduino UNO/Nano** (optional, if you're using hardware alert via buzzer or LED)
- **Alert.wav** sound file for the alarm
- **shape_predictor_68_face_landmarks.dat** file (required for face landmark detection)

### 🔧 Installation

Install the required Python libraries using the following command:

```bash
pip install opencv-python dlib imutils playsound numpy pyserial

```md

## 🧠 Python Code Overview

The main script used for this project is **`drowsiness_yawn.py`**.  
It performs **real-time drowsiness and yawn detection** using the following steps:

1. Captures live video feed using OpenCV.  
2. Detects facial landmarks (eyes and mouth) using dlib’s `shape_predictor_68_face_landmarks.dat`.  
3. Calculates **Eye Aspect Ratio (EAR)** to detect eye closure.  
4. Calculates **Lip Distance** to detect yawning.  
5. Plays an **alarm sound** when drowsiness or yawning is detected.  
6. Sends a signal to **Arduino** (via serial communication) to activate LED or buzzer when drowsiness is detected.

To run the Python script, use the following command:

```bash
python drowsiness_yawn.py --webcam 0

