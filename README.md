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

## ⚙️ Requirements
Install the required libraries before running:
```bash
pip install opencv-python dlib imutils playsound numpy scipy pyserial

