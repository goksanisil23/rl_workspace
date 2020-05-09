#define ENCODEROUTPUT 663

const int HALLSEN_A = 2; // Hall sensor A connected to pin 3 (external interrupt)
const int HALLSEN_B = 3; // Hall sensor A connected to pin 3 (external interrupt)
//const int MOTOR1A = 10;
//const int MOTOR1B = 12;

//The sample code for driving one way motor encoder
volatile long encoderValue = 0;

int interval = 1000;
long previousMillis = 0;
long currentMillis = 0;

int rpm = 0;
boolean measureRpm = false;
//int motorPwm = 0;

void setup() {

  Serial.begin(57600);//Initialize the serial port
  EncoderInit();//Initialize the module
  
//   pinMode( MOTOR1A , OUTPUT);
//   pinMode( MOTOR1B , OUTPUT);
//
//   digitalWrite(MOTOR1A,HIGH);
//   digitalWrite(MOTOR1B,LOW);

   encoderValue = 0;
   previousMillis = millis();
}

void loop() {
  // put your main code here, to run repeatedly:

 

  // Update RPM value on every second
  currentMillis = millis();
  if (currentMillis - previousMillis > interval) {
    previousMillis = currentMillis;

   

    // Revolutions per minute (RPM) =
    // (total encoder pulse in 1s / motor encoder output) x 60s
    rpm = (float)(encoderValue * 60 / ENCODEROUTPUT);
      

    Serial.print(encoderValue);
    Serial.print(" pulse \n");
  }

}

void EncoderInit()
{
 // Attach interrupt at hall sensor A on each rising signal
  attachInterrupt(digitalPinToInterrupt(HALLSEN_A), updateEncoder, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(HALLSEN_B), updateEncoder, RISING);
  //attachInterrupt(digitalPinToInterrupt(HALLSEN_A), updateEncoder, FALLING);
  attachInterrupt(digitalPinToInterrupt(HALLSEN_B), updateEncoder, CHANGE);  
}


void updateEncoder()
{
  // Add encoderValue by 1, each time it detects rising signal
  // from hall sensor A
  encoderValue++;
}
