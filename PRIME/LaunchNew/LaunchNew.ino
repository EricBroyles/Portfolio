/*
  SPACE JAM: Launch.ino
  Author: Eric Broyles

  Improvements from LaunchStable.ino
  - (Done) Additional check to ensure each loop iteration is at least 30 ms to prevent excessive power draw (somewhat redundant as reading temp sensors should always take 75ms at least)
  - (Done) Added LAUNCH_UPLOAD_VERIFICATION_ID: a unique value set by the user before launch that shows up in the MAIN files ouput to the SD card & Serial.
    This allows the user to verify that the latest code was successfully uploaded to the Arduino if they dont have Serial ouput.
    This is neccisary as failing to properly upload without any error due to a COM mismatch has occured before. And it can be hard to notice small changes.
  - (Done) Changed the way data is stored so it gets added to RUN## folder on SD Card
  - (Get test script to work first) Added the code to read multiple temperature sensors, reading temp sensors should take 75 ms not 75 *3 ms

  Notes:
  - run_time: means from the arduino power on.
  - mission_time: means from when we think the launch is underway.
  - RTC is 8 min behind.

  Pre Launch Instructions
  * Clear the SD Card of any files (good idea to do anyway, not absolutly neccisary)
  * modify the launch sequence constants as desired (ensure dummy numbers are commented out or removed)
  * if you have modified the code, modify the launch_upload_verification_id to somthing unique
  * If the code is modified reupload it to the arduino.
  * if you have Serial access the launch_upload_verification_id should show up in Serial Monitor
  * if you dont have Serial access the launch_upload_verification_id will show up in any MAIN.txt on the SD card
  * verify the launch_upload_verification_id is the one you specified.
*/

#include <array>  
#include <cmath>
#include <Wire.h>
#include <RTClib.h>
#include <SD.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_MAX31865.h>
#include "Adafruit_MPRLS.h"

void setup();
void loop();
void run_setup_test();
void run_test_loop();
void run_setup_launch_seq();
void run_launch_seq_loop();
void setup_i2c();
void setup_serial();
void setup_rtclock();
void setup_sd();
void setup_pressure();
void setup_temperature();
void setup_acceleration();
void setup_lights();
void setup_camera();
void setup_motors();
void shutdown_lights();
void shutdown_camera();
void shutdown_motors();
std::array<float, 3> get_temperature();
float get_accel_mag_g();
float calc_accel_mag_g(std::array<float, 3> accel_xyz);
std::array<float, 3> get_accel_xyz();
float get_pressure();
float get_pressure_psi();
void create_main_file();
void create_data_file();
void data_file_write();
void get_file_name_time(char* buffer, size_t len);

/* 
  CONVERSIONS
*/
const float g1 = 9.81;                  // m/s2
const float HPA_PER_PSI = 68.947572932; // hPa/psi

/* 
  SENSORS
  Notes:
  - RTC (Clock) and MPRLS (Pressure) do not need a port number
*/
const int SERIAL_PORT = 57600; // not really a port its a baud or smthng
const int SD_PORT = 10;        //SD Card CS port
const int TEMP1_PORT = 9; 
const int TEMP2_PORT = 4; 
const int TEMP3_PORT = 33; 
const int ACCEL_PORT = 29;    // ADXL 345 (port was hex 0x1D is just 29)
const int MOTORS_PORT = 38;
const int CAMERA_PORT = 40;
const int LIGHTS_PORT = 36;

const float TEMP1_RREF = 430.0;    //TEMP1 uses Rref 431
const float TEMP2_RREF = 430.0;    //TEMP2 uses Rref 431
const float TEMP3_RREF = 4300.0;   //TEMP3 uses Rref 432
const float TEMP_RNOMINAL = 100.0;
const int ACCEL_UNIQUE_ID = 12345; // arbitrary id needed becuase we are using adxl345_U 

/*
  LAUNCH SEQUENCE
  1. System is Powered on
  Power on Due & RTC & SD Card
  2. WAIT 25 min (no data being read so can halt the program with delay)
  turn on Pressure, temperature, accel
  3. CHECK g > 2gs for 2 seconds (use gx^2 + gy^2 + gz^2 >= 4g^2)
  start MissionTime  (The Flight has begun)
  4. WAIT 30 seconds (we have been flying for 30 seconds)
  turn on lights and camera
  5. CHECK g < .01 (for 1 second I HAVE ADDED THIS IS THIS NEEDED?)
  turn motor on (we have started to drain the tanks)
  6. WAIT 30 seconds (time to drain tanks)
  turn motors off (tank is drained fully)
  7. WAIT 30 seconds
  turn camera and lights off
  8. WAIT 10 min
  turn everything off
  9. DONE. all systems shutdown or idle
*/
// const float SEQ2_WAIT           = 25 * 60 * 1000.0; // ms
// const float SEQ3_CHECK_G        = 2;                // g
// const float SEQ3_CHECK_DURATION = 2000.0;           // ms
// const float SEQ4_WAIT           = 30 * 1000.0;      // ms
// const float SEQ5_CHECK_G        = .01;              // g
// const float SEQ5_CHECK_DURATION = 1000.0;           // ms
// const float SEQ6_WAIT           = 30 * 1000.0;      // ms
// const float SEQ7_WAIT           = 30 * 1000.0;      // ms
// const float SEQ8_WAIT           = 10 * 60 * 1000.0; // ms

/*
DUMMY LAUNCH SEQ (for easier testing)
____ REPLACE for LAUNCH ___
*/
const float SEQ2_WAIT           = 30 * 1000.0; // ms
const float SEQ3_CHECK_G        = .9;                // g
const float SEQ3_CHECK_DURATION = 2000.0;           // ms
const float SEQ4_WAIT           = 30 * 1000.0;      // ms
const float SEQ5_CHECK_G        = 1.1;              // g
const float SEQ5_CHECK_DURATION = 1 * 1000.0;           // ms
const float SEQ6_WAIT           = 30 * 1000.0;      // ms
const float SEQ7_WAIT           = 30 * 1000.0;      // ms
const float SEQ8_WAIT           = 30 * 1000.0; // ms


/*
Set to unique value when uploading to allow you to see on Serial and MAIN.txt that the upload
was successful
____ REPLACE for LAUNCH ___
*/
const int LAUNCH_UPLOAD_VERIFICATION_ID = 0;

DateTime run_start_datetime;             // recorded from RTC Clock when the system is powered on (SEQ1)
DateTime mission_start_datetime;         // recorded from RTC Clock when the flight has begun (SEQ3)

const int MIN_TIME_STEP       = 30;      // (ms) min time between loop iterations
const int DATA_FILE_INTERVAL  = 30*1000; // (ms) time between creating new datafiles
unsigned long data_file_timer = 0;       // (ms) time passed since last datafile created
unsigned long seq3_timer      = 0;       // (ms) time passed while meeting seq3 criterion
unsigned long seq4_timer      = 0;       // (ms) time passed during seq4
unsigned long seq5_timer      = 0;       // (ms) time passed while meeting seq3 criterion
unsigned long seq6_timer      = 0;       // (ms) time passed during seq6
unsigned long seq7_timer      = 0;       // (ms) time passed during seq7
unsigned long seq8_timer      = 0;       // (ms) time passed during seq8
unsigned long mission_start   = 0;       // (ms) the millis() when the mission starts (makes it easier to read the data file and see when the mission started)

unsigned long prev_millis     = 0;

int launch_seq = 1; // The current Launch Sequence Step the system is on. Change to the next step at the end of the current.

RTC_PCF8523              rtclock; 
Adafruit_MPRLS           pressure; 
Adafruit_MAX31865        temp1 = Adafruit_MAX31865(TEMP1_PORT); 
Adafruit_MAX31865        temp2 = Adafruit_MAX31865(TEMP2_PORT); 
Adafruit_MAX31865        temp3 = Adafruit_MAX31865(TEMP3_PORT); 
Adafruit_ADXL345_Unified accel; 
File                     curr_main_file; 
File                     curr_data_file; 

const char* TOP_LEVEL_PATH = "/root";
char run_folder_path[32];

/*
  LAUNCH SEQUENCE ERRORS
  Notes:
  - This exists to record any errors experienced during the LAUNCH SEQUENCE so they can be saved to CSV.
*/
bool CLOCK_BEGIN_FAILED    = false; // the system likley failed to find the rtc
bool CLOCK_NOT_INITIALIZED = false; // the rtc lost backup pwr or is not initialized. RTC clock is set to the last time the code was complied.
bool SD_BEGIN_FAILED       = false; // prevents any writing to file. check wiring/format/cs pins for the sd card
bool PRESSURE_BEGIN_FAILED = false; // cannot find mprls. check wiring/format/pin
bool TEMP1_BEGIN_FAILED    = false; // cannot find max1. check wiring/format/pin
bool TEMP2_BEGIN_FAILED    = false; // cannot find max2. check wiring/format/pin
bool TEMP3_BEGIN_FAILED    = false; // cannot find max3. check wiring/format/pin
bool ACCEL_BEGIN_FAILED    = false; // cannot find accelerometor. check wiring/format/pin
bool MAIN_FILE_FAILED      = false; // failed to create a main file
bool DATA_FILE_FAILED      = false; // failed to create a data file

void setup() {
  run_setup_launch_seq();
}

void loop() {
  run_launch_seq_loop();
}

void run_setup_launch_seq() {
  // SEQ1: runs on power up
  setup_i2c();
  setup_serial();
  setup_rtclock();
  setup_sd();
  run_start_datetime = rtclock.now();
  launch_seq = 2; // Move onto next launch sequence step
  setup_run_folder();
  
  create_main_file();
  Serial.println("----------------------------------");
  Serial.println("START SEQ2");
  Serial.println(run_folder_path);
  Serial.print("LAUNCH_UPLOAD_VERIFICATION_ID: ");
  Serial.println(LAUNCH_UPLOAD_VERIFICATION_ID);
}

void run_launch_seq_loop() {
  unsigned long curr_millis = millis();
  unsigned long delta = curr_millis - prev_millis;
  prev_millis = curr_millis;

  Serial.print("LOOP: "); Serial.print(delta); Serial.print(" "); Serial.println(launch_seq);

  if (launch_seq == 2) {
    delay(SEQ2_WAIT); // This halts the program due to not needing to read/write any data yet
    setup_pressure();
    setup_temperature();
    setup_acceleration();
    launch_seq = 3;
    create_main_file();
    Serial.println("END SEQ2");
  }

  else if (launch_seq == 3) {
    if (get_accel_mag_g() >= SEQ3_CHECK_G) {
      seq3_timer += delta;
    }
    else{
      seq3_timer = 0;
    }
    if (seq3_timer >= SEQ3_CHECK_DURATION) {
      // I have passed both the g check and wait for SEQ3. 
      mission_start_datetime = rtclock.now();
      mission_start = millis();
      launch_seq = 4;
      create_main_file();
      Serial.println("END SEQ3");
    }
  }

  else if (launch_seq == 4) {
    if (seq4_timer >= SEQ4_WAIT) {
      // the timer is complete: turn on lights and camera
      setup_lights();
      setup_camera();
      launch_seq = 5;
      create_main_file();
      Serial.println("END SEQ4");
    }
    else {
      //Serial.print(seq4_timer); Serial.print(" "); Serial.println(delta);
      seq4_timer += delta;
    }
  }

  else if (launch_seq == 5) {
    if (get_accel_mag_g() <= SEQ5_CHECK_G) {
      seq5_timer += delta;
    }
    else{
      Serial.println("RESET seq5_timer");
      seq5_timer = 0;
    }
    if (seq5_timer >= SEQ5_CHECK_DURATION) {
      // I have passed both the g check and wait for SEQ5. 
      setup_motors();
      launch_seq = 6;
      create_main_file();
      Serial.println("END SEQ5");
    }
  }

  else if (launch_seq == 6) {
    if (seq6_timer >= SEQ6_WAIT) {
      // the timer is complete: turn off motors
      shutdown_motors();
      launch_seq = 7;
      create_main_file();
      Serial.println("END SEQ6");
    }
    else {
      //Serial.print("SEQ6_TIMER: "); Serial.println(seq6_timer);
      seq6_timer += delta;
    }
  }

  else if (launch_seq == 7) {
    if (seq7_timer >= SEQ7_WAIT) {
      // the timer is complete: turn off cameras and lights
      shutdown_camera();
      shutdown_lights();
      launch_seq = 8;
      create_main_file();
      Serial.println("END SEQ7");
    }
    else {
      seq7_timer += delta;
    }
  }

  else if (launch_seq == 8) {
    if (seq8_timer >= SEQ8_WAIT) {
      // the timer is complete: turn everything off
      // There is not a good way to turn off a senosors. I can just add a way to do a final write to sd card then cause a while loop with large delays?
      launch_seq = 9;
      create_main_file();
      Serial.println("END SEQ8");
      Serial.print("DONE "); Serial.println(millis());
    }
    else {
      seq8_timer += delta;
    }
  }

  if (2 <= launch_seq && launch_seq <= 8) {

    if (data_file_timer >= DATA_FILE_INTERVAL) {
      create_data_file();
      data_file_timer = 0;
    }
    else {
      data_file_write();
      data_file_timer += delta;
    }
  }

  if (delta < MIN_TIME_STEP) {
    delay(MIN_TIME_STEP);
  }
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

void setup_rtclock() {
  /*
    Begins the Real Time Clock & Starts time keeping
    Needed for run_time and mission_time
  */
  bool did_begin = rtclock.begin(); 
  CLOCK_BEGIN_FAILED = !did_begin;
  if (!rtclock.initialized() || rtclock.lostPower()) {
    CLOCK_NOT_INITIALIZED = true;
    rtclock.adjust(DateTime(F(__DATE__), F(__TIME__))); //clocks time becomes time of last code compile
  }
  rtclock.start(); // begins time keeping
}

void setup_sd() {
  /*
    Begins the SD Card
    Needed to write data.
  */
  pinMode(SD_PORT, OUTPUT);
  bool did_begin = SD.begin(SD_PORT);
  SD_BEGIN_FAILED = !did_begin;
}

void setup_run_folder() {
  /*
  Creates a folder called RUN01 inside root
  assigns the folder name to run_folder_path
  */
  int run_count = get_run_count();
  if (run_count < 0) {
    Serial.println("Error counting runs.");
    return;
  }
  get_folder_path(run_count + 1, run_folder_path, sizeof(run_folder_path));
  if (create_folder(run_folder_path)) {
    Serial.print("Created new run folder: ");
    Serial.println(run_folder_path);
  } else {
    Serial.println("Failed to create new run folder.");
  }
}

void setup_pressure() {
  /*
    Begin the Pressure Sensor (MPRLS)
  */
  bool did_begin = pressure.begin();
  PRESSURE_BEGIN_FAILED = !did_begin;
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

  TEMP1_BEGIN_FAILED = !did_begin1;
  TEMP2_BEGIN_FAILED = !did_begin2;
  TEMP3_BEGIN_FAILED = !did_begin3;
}

void setup_acceleration() {
  /*
    Begin acceleration sensor (ADXL)
  */
  accel = Adafruit_ADXL345_Unified(ACCEL_UNIQUE_ID);
  bool did_begin = accel.begin(ACCEL_PORT);
  ACCEL_BEGIN_FAILED = !did_begin;
  accel.setRange(ADXL345_RANGE_8_G); // set max accel to 8g 
}

void setup_lights() {
  /*
    Turns on the lights.
  */
  pinMode(LIGHTS_PORT, OUTPUT);
  digitalWrite(LIGHTS_PORT, HIGH);
}

void setup_camera() {
  /*
    Turns on the camera.
  */
  pinMode(CAMERA_PORT, OUTPUT);
  digitalWrite(CAMERA_PORT, HIGH);
}

void setup_motors() {
  /*
    Turns on the motors.
  */
  pinMode(MOTORS_PORT, OUTPUT); 
  digitalWrite(MOTORS_PORT, HIGH); 
}

void shutdown_lights() {
  /*
    Turns off the lights.
  */
  digitalWrite(LIGHTS_PORT, LOW);
}

void shutdown_camera() {
  /*
    Turns off the camera.
  */
  digitalWrite(CAMERA_PORT, LOW);
}

void shutdown_motors() {
  /*
    Turns off the motors.
  */
  digitalWrite(MOTORS_PORT, LOW); 
}

std::array<float, 3> get_temperature() {
  /*
  Reads all 3 of the temperature sensors (slow takes 75ms for each)
  Note: Takes 75 ms * number of temp sensors
  Units: C
  Returns: {temp1, temp2, temp3}
  */
  std::array<float, 3> temps_reading = {0.0, 0.0, 0.0};
  temps_reading[0] = temp1.temperature(TEMP_RNOMINAL, TEMP1_RREF);
  temps_reading[1] = temp2.temperature(TEMP_RNOMINAL, TEMP2_RREF);
  temps_reading[2] = temp3.temperature(TEMP_RNOMINAL, TEMP3_RREF);
  return temps_reading;
}

float get_accel_mag_g() {
  /*
  Gets the acceleration magnitude from current accel sensor reading
  Units: unitless
  */
  return calc_accel_mag_g(get_accel_xyz());
}

float calc_accel_mag_g(std::array<float, 3> accel_xyz) {
  /*
  Calculates the acceleration magnitude from passed accel sensor reading
  Notes:
  - This exists seperate get_accel_mag_g() so when I write the accle_xyz and mag_g to data file I can use one sensor call and avoid drift.
  Units: unitless
  */
  std::array<float, 3> xyz = get_accel_xyz();
  float gx = xyz[0]/g1;
  float gy = xyz[1]/g1;
  float gz = xyz[2]/g1;
  return sqrtf(gx*gx + gy*gy + gz*gz);
}

std::array<float, 3> get_accel_xyz() {
  /*
  Gets the acceleration sensor data 
  Units: m/s2
  Returns: {x_accel, y_accel, z_accel} 
  */
  std::array<float, 3> accel_reading = {0.0, 0.0, 0.0};
  sensors_event_t event;
  accel.getEvent(&event);
  accel_reading[0] = event.acceleration.x;
  accel_reading[1] = event.acceleration.y;
  accel_reading[2] = event.acceleration.z;
  return accel_reading;
}

float get_pressure() {
  /*
    Gets the pressure sensor data 
    Units: Pa
    Returns: pressure
  */
  float pressure_hPa = pressure.readPressure(); // sensor returns hPa
  return pressure_hPa * 100;                
}

float get_pressure_psi() {
  /*
    Gets the pressure sensor data 
    Units: psi
    Returns: pressure
  */
  float pressure_hPa = pressure.readPressure(); // sensor returns hPa
  return pressure_hPa / HPA_PER_PSI;                
}

void create_main_file() {
  /*
  Creates a new main_file called MainHHMMSS.txt
  Used to store launch sequence information that rarely changes.
  Notes:
  - if a file exists with the same name delete it then create the new file.
  */

  if (curr_main_file) {
    curr_main_file.flush();
    curr_main_file.close();
  }

  char timebuf[7];     // HHMMSS + null
  get_file_name_time(timebuf, sizeof(timebuf));

  char filename[40];
  snprintf(filename, sizeof(filename), "%s/D%s.csv", run_folder_path, timebuf); //Name required by SD card to be 8char.csv
  
  // delete old if exists
  if (SD.exists(filename)) {
    SD.remove(filename);
  }
  
  curr_main_file = SD.open(filename, FILE_WRITE);

  if (curr_main_file) {
    curr_main_file.println("=== Main ===");
    curr_main_file.print("LAUNCH_UPLOAD_VERIFICATION_ID: ");
    curr_main_file.println(LAUNCH_UPLOAD_VERIFICATION_ID);
    curr_main_file.print("run_start_datetime: ");
    curr_main_file.println((run_start_datetime.year() == 2000) ? "None": run_start_datetime.timestamp());
    curr_main_file.print("mission_start_datetime: ");
    curr_main_file.println((mission_start_datetime.year() == 2000) ? "None": mission_start_datetime.timestamp());
    curr_main_file.print("mission_start: ");
    curr_main_file.println(mission_start ? String(mission_start) : "None");
    curr_main_file.print("launch_seq: "); curr_main_file.println(launch_seq);
    curr_main_file.print("CLOCK_BEGIN_FAILED: ");
    curr_main_file.println(CLOCK_BEGIN_FAILED);
    curr_main_file.print("CLOCK_NOT_INITIALIZED: ");
    curr_main_file.println(CLOCK_NOT_INITIALIZED);
    curr_main_file.print("SD_BEGIN_FAILED: ");
    curr_main_file.println(SD_BEGIN_FAILED);
    curr_main_file.print("PRESSURE_BEGIN_FAILED: ");
    curr_main_file.println(PRESSURE_BEGIN_FAILED);
    curr_main_file.print("TEMP1_BEGIN_FAILED: ");
    curr_main_file.println(TEMP1_BEGIN_FAILED);
    curr_main_file.print("TEMP2_BEGIN_FAILED: ");
    curr_main_file.println(TEMP2_BEGIN_FAILED);
    curr_main_file.print("TEMP3_BEGIN_FAILED: ");
    curr_main_file.println(TEMP3_BEGIN_FAILED);
    curr_main_file.print("ACCEL_BEGIN_FAILED: ");
    curr_main_file.println(ACCEL_BEGIN_FAILED);
    curr_main_file.print("MAIN_FILE_FAILED: ");
    curr_main_file.println(MAIN_FILE_FAILED);
    curr_main_file.print("DATA_FILE_FAILED: ");
    curr_main_file.println(DATA_FILE_FAILED);
    curr_main_file.flush();
  }
  else {
    MAIN_FILE_FAILED = true;
  }
}

void create_data_file() {
  /*
  Creates a new data_file called DataHHMMSS.csv, adds the headers
  Used to store sensor data.
  Notes:
  - WARNING: if the SD card is not cleared and there is a DataHHMMSS.csv with the same name this will write the headers again and then write data to the old file.
  */

  if (curr_data_file) {
    curr_data_file.flush();
    curr_data_file.close();
  }

  char timebuf[7];     // HHMMSS + null
  get_file_name_time(timebuf, sizeof(timebuf));

  char filename[40];
  snprintf(filename, sizeof(filename), "%s/D%s.csv", run_folder_path, timebuf); //Name required by SD card to be 8char.csv
  
  curr_data_file = SD.open(filename, FILE_WRITE);

  if (curr_data_file) {
    curr_data_file.println("Run Time (ms), Pressure (Pa), Temp1 (C), Temp2 (C), Temp3 (C), AccelX (m/s2), AccelY (m/s2), AccelZ (m/s2), G");
    curr_data_file.flush();
  }
  else {
    DATA_FILE_FAILED = true;
  }
}

void data_file_write() {
  /*
    Write to the curr_data_file
  */
  if (!curr_data_file) return;
  unsigned long       ms = millis();
  float                P = get_pressure();
  std::array<float, 3> T = get_temperature();
  std::array<float, 3> A = get_accel_xyz();
  float                G = calc_accel_mag_g(A);
  curr_data_file.print(ms);   curr_data_file.print(", ");
  curr_data_file.print(P);    curr_data_file.print(", ");
  curr_data_file.print(T[0]); curr_data_file.print(", ");
  curr_data_file.print(T[1]); curr_data_file.print(", ");
  curr_data_file.print(T[2]); curr_data_file.print(", ");
  curr_data_file.print(A[0]); curr_data_file.print(", ");
  curr_data_file.print(A[1]); curr_data_file.print(", ");
  curr_data_file.print(A[2]); curr_data_file.print(", ");
  curr_data_file.println(G);  
  curr_data_file.flush();
}

void get_file_name_time(char* buffer, size_t len) {
  unsigned long ms = millis();
  unsigned long seconds = (ms / 1000) % 60;
  unsigned long minutes = (ms / 60000) % 60;
  unsigned long hours   = (ms / 3600000) % 24;

  snprintf(buffer, len, "%02lu%02lu%02lu", hours, minutes, seconds);
}


int get_run_count() {
  // Returns the number of RUN folders in the top-level directory, or -1 on failure
  File dir = SD.open(TOP_LEVEL_PATH);

  if (!dir) {
    Serial.print("Failed to open directory: ");
    Serial.println(TOP_LEVEL_PATH);
    return -1;
  }
  if (!dir.isDirectory()) {
    Serial.print("Not a directory: ");
    Serial.println(TOP_LEVEL_PATH);
    dir.close();
    return -1;
  }

  int folder_count = 0;

  File entry = dir.openNextFile();
  while (entry) {
    if (entry.isDirectory()) {
      // Optional: Only count folders that start with "RUN"
      String name = entry.name();
      if (name.startsWith("RUN")) {
        folder_count++;
      }
    }
    entry.close();
    entry = dir.openNextFile();
  }
  dir.close();

  return folder_count;
}

void get_folder_path(int run_num, char* buffer, size_t buffer_size) {
  // Writes the full folder path for a given run number into buffer
  // e.g. buffer = "/Toplevel/RUN02"
  // Note is not just limited to 00 to 99, can do larger numbers until buffer overflow RUN9999 etc.
  snprintf(buffer, buffer_size, "%s/RUN%02d", TOP_LEVEL_PATH, run_num);
}

// Creates a folder at the given path, returns true if successful
bool create_folder(const char* folder_path) {
  Serial.print("Creating folder: ");
  Serial.println(folder_path);

  if (SD.mkdir(folder_path)) {
    Serial.println("Folder created successfully.");
    return true;
  } else {
    Serial.println("Failed to create folder.");
    return false;
  }
}
















