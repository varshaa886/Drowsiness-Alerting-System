// ===========================
// DROWSINESS ALERT SYSTEM
// Arduino Code for Buzzer + LED Signal
// ===========================

// Pin Assignments
int buzzer = 8;        // Buzzer connected to digital pin 8
int redLED = 9;        // Red LED - Drowsy alert
int yellowLED = 10;    // Yellow LED - Optional (warning or transition)
int greenLED = 11;     // Green LED - Normal / Awake
char data;             // Variable to store incoming serial data from Python

void setup() {
  // Configure pins as outputs
  pinMode(buzzer, OUTPUT);
  pinMode(redLED, OUTPUT);
  pinMode(yellowLED, OUTPUT);
  pinMode(greenLED, OUTPUT);
  
  // Start serial communication with laptop (Python)
  Serial.begin(9600);
  
  // Initial state: Normal / Awake
  digitalWrite(greenLED, HIGH);
  digitalWrite(redLED, LOW);
  digitalWrite(yellowLED, LOW);
  noTone(buzzer);
}

void loop() {
  // Check if Python has sent any data (via USB Serial)
  if (Serial.available() > 0) {
    data = Serial.read();   // Read the character ('1' or '0')

    if (data == '1') {
      // Drowsy detected
      digitalWrite(redLED, HIGH);    // Turn ON red LED
      digitalWrite(greenLED, LOW);   // Turn OFF green LED
      digitalWrite(yellowLED, LOW);
      tone(buzzer, 1000);            // Activate buzzer with 1kHz tone
    } 
    
    else if (data == '0') {
      // Normal state
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, HIGH);  // Turn ON green LED
      digitalWrite(yellowLED, LOW);
      noTone(buzzer);                // Stop buzzer
    } 
    
    else if (data == '2') {
      // Optional: Intermediate warning
      digitalWrite(redLED, LOW);
      digitalWrite(greenLED, LOW);
      digitalWrite(yellowLED, HIGH); // Turn ON yellow LED
      tone(buzzer, 500);             // Softer warning tone
    }
  }
}