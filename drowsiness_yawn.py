
# Usage: python drowsiness_yawn_bench.py --webcam 0 --port COM11 --baud 9600
from scipy.spatial import distance as dist #computes Euclidean distance between 2 points – used for EAR, MAR, etc.
from imutils.video import VideoStream #wrapper around OpenCV’s VideoCapture for easier webcam use.
from imutils import face_utils #helper utilities for face landmarks
from collections import deque
import numpy as np
import argparse #For parsing command-line arguments --webcame--port
import imutils #simple function to resize frames while preserving aspect ratio.
import time
import dlib #Face landmark predictor using
import cv2 #eads webcam video, converts each frame to grayscale, detects faces using a Haar cascade, draws outlines around eyes and mouth, adds text labels, and shows the processed frame on the screen.
import serial
import os
import sys
import statistics

# Utility functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) #vertical distances between eyelid landmark pairs.
    B = dist.euclidean(eye[2], eye[4]) # vertical distances between eyelid landmark pairs.
    C = dist.euclidean(eye[0], eye[3]) # horizontal distance across the eye
    return (A + B) / (2.0 * C + 1e-6)

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # total 68 landmarks extracted from both
    leftEye  = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR  = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0 #Computes EAR for each eye and averages them.
    return (ear, leftEye, rightEye)

# --- Yawn helpers --- # we need eye related values also for mouth. why?  Because people’s faces appear bigger or smaller depending on how close they are to the camera.If we only use mouth size directly, the system gets confused. This makes the yawn measurement camera-distance independent.
def interpupil_distance(shape_np):
    left  = np.mean(shape_np[36:42], axis=0)   # left eye 36-41
    right = np.mean(shape_np[42:48], axis=0)   # right eye 42-47
    return float(np.linalg.norm(left - right)) 

def mouth_aspect_ratio(shape_np): #same as we did for ear but we do for mouth
    A = dist.euclidean(shape_np[50], shape_np[58])  # vertical distances (upper lip–lower lip).
    B = dist.euclidean(shape_np[52], shape_np[56])
    C = dist.euclidean(shape_np[48], shape_np[54])  # horizontal distance (corner to corner).
    return (A + B) / (2.0 * C + 1e-6)

def normalized_yawn_metric(shape_np): #normalised yawn = mouth opening/distance between eyes
    top_lip = shape_np[50:53]
    top_lip = np.concatenate((top_lip, shape_np[61:64])) ##Picks several upper lip points
    low_lip = shape_np[56:59]
    low_lip = np.concatenate((low_lip, shape_np[65:68])) #Picks several lower lip points
    top_mean = np.mean(top_lip, axis=0) #avg of all points upper lip
    low_mean = np.mean(low_lip, axis=0) #avg of all points lower lip
    raw = abs(top_mean[1] - low_mean[1]) #vertical gap between lips is calculated
    ipd = interpupil_distance(shape_np)
    return float(raw / (ipd + 1e-6))

def level_style(level):
    if level == 0: return ((0,255,0),  "LEVEL 0: NORMAL")
    if level == 1: return ((0,255,255),"LEVEL 1: SLIGHT DROWSY")
    return ((0,0,255),  "LEVEL 2: VERY DROWSY")

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("--port", type=str, default="COM11", help="Arduino serial port (e.g., COM11 or /dev/ttyUSB0)") #change comport here for arduino
ap.add_argument("--baud", type=int, default=9600, help="Arduino baud rate")
ap.add_argument("--frame-width", type=int, default=640, help="resize width for processing")
args = vars(ap.parse_args())


# Tunable thresholds
# Eyes 
EAR_L1_THRESH   = 0.27
EAR_L2_THRESH   = 0.22
EAR_L1_FRAMES   = 12
EAR_L2_FRAMES   = 20

# Yawn 
YAWN_NORM_L1      = 0.32
YAWN_NORM_L2      = 0.38
YAWN_L1_MIN_SEC   = 0.5
YAWN_L2_MIN_SEC   = 0.8

# Talking filter
TALK_WINDOW_SEC      = 0.4
TALK_TRANSITIONS_MAX = 4

# Arduino refresh
ALERT_COOLDOWN_SEC = 3.0

# Arduino connection
arduino = None
try:
    arduino = serial.Serial(args["port"], args["baud"])
    time.sleep(2.0)  # give Arduino time to reset
    print(f"[INFO] Arduino connected on {args['port']} @ {args['baud']}")
except Exception as e:
    print(f"[WARN] Could not open Arduino on {args['port']}: {e}")
    arduino = None

#avoid spamming Arduino.
last_sent_level = 0
last_sent_time  = 0.0
def send_level_to_arduino(level, ser=None):
    global last_sent_level, last_sent_time
    now = time.time()
    if ser is None:
        return
    if level == last_sent_level:
        if level == 2 and (now - last_sent_time) > ALERT_COOLDOWN_SEC:
            ser.write(b'2')
            last_sent_time = now
        return
    if   level == 0: ser.write(b'0')
    elif level == 1: ser.write(b'1')
    else:            ser.write(b'2')
    last_sent_level = level
    last_sent_time  = now

# Load detectors 
print("-> Loading the predictor and detector...")
cascade_path = "haarcascade_frontalface_default.xml" #A file that contains a pre-trained face detection model made by OpenCV 
             #OpenCV trained it using thousands of images of faces and non-faces.They used a machine-learning method called Haar Cascade Classifier.                                    
shape_path   = "shape_predictor_68_face_landmarks.dat" #A file that contains a trained model that predicts 68 face landmarks
                #Dlib trained this model using thousands of manually labeled faces.ach face image had 68 landmarks marked by human annotators.A machine-learning algorithm called ensemble regression trees learned how to predict these points.
if not os.path.exists(cascade_path):
    print(f"[ERROR] Missing {cascade_path}"); sys.exit(1)
if not os.path.exists(shape_path):
    print(f"[ERROR] Missing {shape_path}"); sys.exit(1)

detector  = cv2.CascadeClassifier(cascade_path) #OpenCV Haar cascade for frontal face detection.
predictor = dlib.shape_predictor(shape_path)#dlib 68-landmark model for facial landmarks.

# Video + State
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# State for closest person counters
ear_counter_L1 = 0
ear_counter_L2 = 0

# Yawn persistence & talking state
yawn_above_L1_since = None
yawn_above_L2_since = None
talk_hist = deque()  # tuples of (t, yawn_norm)

# ---- BENCHMARK STATE ----
frame_times = []             # per-frame ms
last_print  = time.time()

# Raw onsets for yawn only (what you asked to measure)
yawn_L1_onset = None
yawn_L2_onset = None

# the part of the code that measures how fast and how accurately your system works.
bench = {
    "yawn_L1_delays": [],
    "yawn_L2_delays": []
}
def ms(x): return int(round(x*1000))
def mean_ms(v): 
    return (sum(v)/len(v)*1000) if v else 0.0

try: # ensures clean up of all after i press Q key
    while True:
        t0 = time.time()  # ---- start frame timer

        frame = vs.read()
        if frame is None:
            print("[WARN] Empty frame; reconnecting camera might help.")
            send_level_to_arduino(0, arduino)
            continue

        frame = imutils.resize(frame, width=args["frame_width"])
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        level = 0
        chosen = None #code fpr closest face detection
        if len(rects) > 0:
            areas = [(w*h, (x,y,w,h)) for (x,y,w,h) in rects] # face area calculated and stored in decending order
            areas.sort(reverse=True, key=lambda z: z[0])
            chosen = areas[0][1]

        for (x, y, w, h) in rects:
            if chosen is not None and (x,y,w,h) == chosen: 
                continue
            cv2.rectangle(frame, (x,y), (x+w,y+h), (200,200,200), 1)

        if chosen is not None:
            (x, y, w, h) = chosen
            rect  = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            yawn_norm = normalized_yawn_metric(shape)
            mar       = mouth_aspect_ratio(shape)  # optional display

            # EAR counters (unchanged)
            if ear < EAR_L2_THRESH:
                ear_counter_L2 += 1; ear_counter_L1 += 1
            elif ear < EAR_L1_THRESH:
                ear_counter_L2 = 0;  ear_counter_L1 += 1
            else:
                ear_counter_L1 = 0;  ear_counter_L2 = 0

            # Talking filter (oscillation on yawn_norm)
            now_ts = time.time()
            talk_hist.append((now_ts, yawn_norm))
            while talk_hist and (now_ts - talk_hist[0][0]) > TALK_WINDOW_SEC:
                 talk_hist.popleft()
            mid = YAWN_NORM_L1
            cross = 0
            for i in range(1, len(talk_hist)):
                p0 = talk_hist[i-1][1] - mid
                p1 = talk_hist[i][1]   - mid
                if p0 == 0 or p1 == 0: 
                    continue
                if (p0 > 0 and p1 < 0) or (p0 < 0 and p1 > 0):
                    cross += 1
            is_talking_like = (cross > TALK_TRANSITIONS_MAX)

            # Yawn persistence timers (only advance if NOT talking-like)
            if yawn_norm >= YAWN_NORM_L2 and not is_talking_like:
                if yawn_above_L2_since is None: yawn_above_L2_since = now_ts
            else:
                yawn_above_L2_since = None

            if yawn_norm >= YAWN_NORM_L1 and not is_talking_like:
                if yawn_above_L1_since is None: yawn_above_L1_since = now_ts
            else:
                yawn_above_L1_since = None

            # -------- BENCHMARK: record raw yawn onset times --------
            # (the moment mouth opens beyond thresholds, not the alert)
            if yawn_norm >= YAWN_NORM_L2 and not is_talking_like:
                if yawn_L2_onset is None: yawn_L2_onset = now_ts
                if yawn_L1_onset is None: yawn_L1_onset = now_ts
            elif yawn_norm >= YAWN_NORM_L1 and not is_talking_like:
                if yawn_L1_onset is None: yawn_L1_onset = now_ts
                yawn_L2_onset = None
            else:
                yawn_L1_onset = None
                yawn_L2_onset = None

            # Flags for sustained yawns (these cause the actual alerts)
            yawn_L2 = (yawn_above_L2_since is not None) and ((now_ts - yawn_above_L2_since) >= YAWN_L2_MIN_SEC)
            yawn_L1 = (yawn_above_L1_since is not None) and ((now_ts - yawn_above_L1_since) >= YAWN_L1_MIN_SEC)

            # -------- BENCHMARK: compute detection delay when alerts fire --------
            if yawn_L2 and yawn_L2_onset is not None:
                bench["yawn_L2_delays"].append(now_ts - yawn_L2_onset)
                yawn_L2_onset = None   # avoid double count
            elif yawn_L1 and yawn_L1_onset is not None:
                bench["yawn_L1_delays"].append(now_ts - yawn_L1_onset)
                yawn_L1_onset = None

            # ---- Final level decision (same as your code)
            if ear_counter_L2 >= EAR_L2_FRAMES or yawn_L2:
                level = 2
            elif ear_counter_L1 >= EAR_L1_FRAMES or yawn_L1:
                level = 1
            else:
                level = 0

            # Draw ONLY for closest person
            color, text = level_style(level)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            leftEyeHull  = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull],  -1, color, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, color, 1)
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, color, 1)

            # HUD - Heads-Up Display.
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"YAWN(norm): {yawn_norm:.2f} | MAR:{mar:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if is_talking_like:
                cv2.putText(frame, "Talking detected (yawn suppressed)", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame, text, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # No face → reset
            ear_counter_L1 = 0
            ear_counter_L2 = 0
            yawn_above_L1_since = None
            yawn_above_L2_since = None
            talk_hist.clear()
            # Also reset yawn onsets (no valid measurement while no face)
            yawn_L1_onset = None
            yawn_L2_onset = None
            cv2.putText(frame, "No face detected", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # Send level to Arduino
        send_level_to_arduino(level, arduino)

        # ---- Frame timing end + rolling stats
        t1 = time.time()
        frame_times.append(t1 - t0)
        if len(frame_times) > 120:
            frame_times.pop(0)
        avg_ms = (sum(frame_times)/len(frame_times)*1000) if frame_times else 0.0
        fps    = (1000.0/avg_ms) if avg_ms > 0 else 0.0

        # ---- BENCH HUD line (bottom)
        hud = f"{avg_ms:4.0f} ms | {fps:4.1f} FPS | " \
              f"Y1:{mean_ms(bench['yawn_L1_delays']):.0f}  Y2:{mean_ms(bench['yawn_L2_delays']):.0f} ms"
        cv2.putText(frame, hud, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Optional: print every ~5s
        if (time.time() - last_print) > 5.0:
            last_print = time.time()
            print("[BENCH]", hud)
            # OPTIONAL CSV log (uncomment to enable):
            # with open("yawn_bench.csv","a") as f:
            #     f.write(f"{time.time()},{avg_ms:.1f},{fps:.1f},{mean_ms(bench['yawn_L1_delays']):.1f},{mean_ms(bench['yawn_L2_delays']):.1f}\n")

        cv2.imshow("Drowsiness/Yawn (Closest Person Only) + BENCH", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    if arduino:
        try:
            arduino.write(b'0'); arduino.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
    vs.stop()


#NOTES: MAR is the raw mouth-opening measurement, so it changes if you move closer or farther from the camera.
#Normalized yawn metric divides mouth opening by the eye distance, making it stable even if your face size in the frame changes.
#MAR is only shown on screen, but normalized yawn is what the system actually uses to detect real yawns.
