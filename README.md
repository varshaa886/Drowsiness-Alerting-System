#  Real-Time Driver Drowsiness Detection System

This project detects driver drowsiness and yawning using **OpenCV**, **Dlib**, and **facial landmarks**.  
If the driver appears sleepy or yawns frequently, the system triggers an **alarm sound** and sends a signal to **Arduino** to activate buzzer and LED alerts.

---

##  Features
- Real-time **eye and mouth monitoring**
- **Sound alarm** and **Arduino buzzer** integration
- Works with any standard webcam
- Adjustable thresholds for EAR and yawn detection

---

##  Requirements

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
```
```bash
##  Python Code Overview

The main script used for this project is **`drowsiness_yawn.py`**.  
It performs **real-time drowsiness and yawn detection** using the following steps:

1. Captures live video feed using OpenCV.  
2. Detects facial landmarks (eyes and mouth) using dlib’s `shape_predictor_68_face_landmarks.dat`.  
3. Calculates **Eye Aspect Ratio (EAR)** to detect eye closure.  
4. Calculates **Lip Distance** to detect yawning.  
5. Plays an **alarm sound** when drowsiness or yawning is detected.  
6. Sends a signal to **Arduino** (via serial communication) to activate LED or buzzer when drowsiness is detected.

To run the Python script, use the following command:
```
```bash
python drowsiness_yawn.py --webcam 0

---

```md
```
```bash

##  Hardware Connections (Arduino + OV7670 Camera Module)

The system uses **Arduino Uno**, **OV7670 camera module**, and **buzzer/LED** for alerting.

### 🔌 Component Connections:

| Component | Arduino Pin | Description |
|------------|--------------|-------------|
| OV7670 Camera | Digital Pins (D0–D7) | Camera data output |
| Camera XCLK | Pin 9 (PWM) | External clock signal |
| Camera SCCB SCL | A5 | I2C Clock |
| Camera SCCB SDA | A4 | I2C Data |
| Buzzer | D12 | Activates when drowsiness detected |
| LED | D13 | Blinks as visual alert |
| GND | GND | Common ground |
| VCC | 3.3V | Power supply to camera and sensors |


---

###  Working Principle

1. The **Python code** detects drowsiness or yawning using webcam feed.  
2. When detected, the system sends a **serial signal** (via USB) to Arduino.  
3. Arduino activates **buzzer/LED** as an alert mechanism.  
4. OV7670 camera can optionally capture frame data for further analysis.

---

---
```md
```
```bash
## 🔌 Arduino Code Overview

The Arduino acts as an **alert controller** that receives signals from the Python program (via serial communication).  
When it receives the signal `'1'`, it activates the **buzzer** and **LED** to alert the driver.

---

###  Arduino Code

```cpp
// Drowsiness Alert - Arduino Code
int buzzer = 12;
int led = 13;
char data = 0;

void setup() {
  pinMode(buzzer, OUTPUT);
  pinMode(led, OUTPUT);
  Serial.begin(9600);  // Match with Python serial baud rate
}

void loop() {
  if (Serial.available() > 0) {
    data = Serial.read();
    if (data == '1') {           // Drowsiness detected
      digitalWrite(buzzer, HIGH);
      digitalWrite(led, HIGH);
      delay(1000);
      digitalWrite(buzzer, LOW);
      digitalWrite(led, LOW);
    }
  }
}

---
```md
```
```bash
##  Hardware Setup and Connections

This project involves both **software (Python)** and **hardware (Arduino + sensors)** components.  
Follow these steps for proper connections:

###  Components Used
- Arduino Uno  
- OV7670 Camera Module *(optional)*  
- Buzzer  
- LED  
- Jumper Wires  
- USB Cable (for serial communication)

---

### ⚙️ Basic Connections

| Component | Arduino Pin | Description |
|------------|-------------|-------------|
| Buzzer     | D12         | Activates when drowsiness detected |
| LED        | D13         | Blinks as visual alert |
| GND        | GND         | Common ground connection |

---

###   OV7670 Camera Connection

If you’re using the OV7670 camera module with Arduino:  
- Connect **SDA → A4**  
- Connect **SCL → A5**  
- Connect **VCC → 3.3V**  
- Connect **GND → GND**  

The camera can be used to test image capture functionality or for future upgrades.

---

###  System Integration

1. Run the Python script (`drowsiness_yawn.py`) on your computer.  
2. Connect the Arduino via USB — it will automatically listen for the `'1'` signal.  
3. When drowsiness or yawning is detected:  
   - The **alarm sound** plays on your computer.  
   - The **LED and buzzer** activate via Arduino.

---

```md
```
```bash
---

##  Project Workflow

This project integrates **computer vision**, **machine learning**, and **embedded systems** to detect driver drowsiness and issue timely alerts.

###  End-to-End Flow

1. **Camera Capture (Webcam / OV7670)**  
   - Captures real-time video frames of the driver’s face.

2. **Facial Landmark Detection (Python + dlib)**  
   - The Python script processes frames using `shape_predictor_68_face_landmarks.dat`.  
   - Eye Aspect Ratio (EAR) and Lip Distance are computed.

3. **Drowsiness / Yawn Detection**  
   - When the EAR falls below a threshold for a set duration → eyes are considered *closed*.  
   - When the Lip Distance exceeds the threshold → *yawning* is detected.

4. **Alert Generation (Software + Hardware)**  
   - A **sound alarm** is triggered through the computer using `playsound`.  
   - A **signal (‘1’)** is sent to Arduino via serial communication.

5. **Arduino Activation**  
   - Arduino receives the signal and activates the **buzzer** and **LED** for alerting.

6. **System Reset**  
   - After the alert, both Python and Arduino return to monitoring mode for continuous operation.

---


###  Key Advantages
- Works in **real-time** with low latency.  
- Combines **AI + IoT + Embedded Systems**.  
- Modular design — easy to extend for advanced features (e.g., GSM alert, cloud logging).

---

