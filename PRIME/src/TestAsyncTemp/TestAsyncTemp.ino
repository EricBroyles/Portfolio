/*
Test the async code I added to the max sensors
*/

#include <array>  
#include <Wire.h>
#include <MyAsync_Adafruit_MAX31865.h>

void setup_i2c();
void setup_serial();
void setup_temperature();
std::array<float, 3> get_temperature();
std::array<float, 3> async_get_temperature();

const int SERIAL_PORT = 57600; // not really a port its a baud or smthng
const int TEMP1_PORT = 9; 
const int TEMP2_PORT = 4; 
const int TEMP3_PORT = 33; 

const float TEMP1_RREF = 430.0;    //TEMP1 uses Rref 431
const float TEMP2_RREF = 430.0;    //TEMP2 uses Rref 431
const float TEMP3_RREF = 4300.0;   //TEMP3 uses Rref 432
const float TEMP_RNOMINAL = 100.0;

MyAsync_Adafruit_MAX31865        temp1 = MyAsync_Adafruit_MAX31865(TEMP1_PORT); 
MyAsync_Adafruit_MAX31865        temp2 = MyAsync_Adafruit_MAX31865(TEMP2_PORT); 
MyAsync_Adafruit_MAX31865        temp3 = MyAsync_Adafruit_MAX31865(TEMP3_PORT);

unsigned long prev_millis     = 0;

void setup() {
  setup_i2c();
  setup_serial();
  Serial.println("----------------------------------");
  Serial.println("START TESTASYNCTEMP");
}

void loop() {
  unsigned long curr_millis = millis();
  unsigned long delta = curr_millis - prev_millis;
  prev_millis = curr_millis;
  Serial.print("LOOP: "); Serial.println(delta);

  std::array<float, 3> reading = get_temperature();
  Serial.println(reading[0]);
  Serial.println(reading[1]);
  Serial.println(reading[2]);
}

void setup_i2c() {
  /*
    Begins the I2C bus on Arduino. 
    This allows communication with sensors (pressure, temp, rtc, ...)
  */
  Wire.begin();
}

void setup_serial() {
  /*
    Begins the Serial so that print statments can be made to Serial Moinitor
    Needed for debugging only and is not needed for the actual launch.
  */
  Serial.begin(SERIAL_PORT);
  while (!Serial){
    delay(10); // wait 10 ms if not ready
  }
}

void setup_temperature() {
  /*
    Begin all 3 temperature sensors (MAX31865)
    WARNING: I am uncertain about the digitalWrite being used to prevent SPI contention, does that need to be set on power on?
  */
  // Ensure MAX31865 CS lines idle high (avoid SPI contention)
  pinMode(TEMP1_PORT, OUTPUT); digitalWrite(TEMP1_PORT, HIGH);
  pinMode(TEMP2_PORT, OUTPUT); digitalWrite(TEMP2_PORT, HIGH);
  pinMode(TEMP3_PORT, OUTPUT); digitalWrite(TEMP3_PORT, HIGH);

  bool did_begin1 = temp1.begin(MAX31865_3WIRE);
  bool did_begin2 = temp2.begin(MAX31865_3WIRE);
  bool did_begin3 = temp3.begin(MAX31865_3WIRE);

  // TEMP1_BEGIN_FAILED = !did_begin1;
  // TEMP2_BEGIN_FAILED = !did_begin2;
  // TEMP3_BEGIN_FAILED = !did_begin3;
}

std::array<float, 3> get_temperature() {
  /*
  Reads all 3 of the temperature sensors
  Note: Takes 75 ms * number of temp sensors
  Units: C
  Returns: {temp1, temp2, temp3}
  */
  std::array<float, 3> temps_reading = {0.0, 0.0, 0.0};
  Serial.print("Hi");
  temps_reading[0] = temp1.temperature(TEMP_RNOMINAL, TEMP1_RREF);
  Serial.print("Hii");
  temps_reading[1] = temp2.temperature(TEMP_RNOMINAL, TEMP2_RREF);
  Serial.print("Hiii");
  temps_reading[2] = temp3.temperature(TEMP_RNOMINAL, TEMP3_RREF);
  Serial.print("Hiiii");
  return temps_reading;
}

std::array<float, 3> async_get_temperature() {
  /*
  Reads all 3 of the temperature sensors
  Note: Takes 75 ms 
  Units: C
  Returns: {temp1, temp2, temp3}
  */
  std::array<float, 3> temps_reading = {0.0, 0.0, 0.0};
  temp1.readRTDPart1();
  temp2.readRTDPart1();
  temp3.readRTDPart1();
  delay(10);
  temp1.readRTDPart2();
  temp2.readRTDPart2();
  temp3.readRTDPart2();
  delay(65);
  temps_reading[0] = temp1.calculateTemperature(temp1.readRTDPart3(), TEMP_RNOMINAL, TEMP1_RREF);
  temps_reading[1] = temp2.calculateTemperature(temp2.readRTDPart3(), TEMP_RNOMINAL, TEMP2_RREF);
  temps_reading[2] = temp3.calculateTemperature(temp3.readRTDPart3(), TEMP_RNOMINAL, TEMP3_RREF);
  return temps_reading;

}

