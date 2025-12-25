# SpaceJam Project Notes

**Author:** Eric Broyles

## Important Files

### LaunchStable.ino
Use this as the working launch code for next semester.

## Setup LaunchStable

1. Download and install the Arduino IDE.
2. The `Arduino` folder is usually located in your `Documents` folder
3. Place the entire `PRIME` folder under `Arduino` folder
4. Move `MyAsync_Adafruit_MAX31865_library` under `Arduino/libraries`
5. Open `LaunchStable.ino`
6. For each `#include` statement at the top of the file:
> * Open the Libraries menu (left side of the Arduino editor)
> * Search for the required library
> * Install it  
7. Press **Verify** (top-left checkmark).
- The code should compile successfully
8. If errors occur, check that all required libraries are installed properly.

## Launch Instructions (LaunchStable.ino)

1. Clear Arduino SD card of previous runs. 
> * If this is infeasible this step may be skipped. Data.csv files will be appended to existing data.csv files of the same timestamp. (Main.txt override each other)
> * If you choose to improve this code: DO NOT clear the SD card on setup. This will delete all data if the Arduino is powered on after the launch. 
2. Ensure camera SD card has enough storage left. 
> * Clearing the camera SD card also clears the uploaded QR code that tells the camera to turn on/off and record from USB power. (more info on box)
3. Modify launch sequence constants as desired. 
4. Modify `LAUNCH_UPLOAD_VERIFICATION_ID` with a unique integer
> * This is a unique value that gets output to Serial Monitor on `setup` and saved to the SD card. 
> * It is used to ensure that the latest code you created has been successfully uploaded without needing to run the entire system.
5. If at any point the code was modified reupload it to the Arduino
> * You do not need to reupload the code if you only cleared either SD card.
6. Start the program, check Serial Monitor for the correct `LAUNCH_UPLOAD_VERIFICATION_ID`. 
> * If you don't have access to Serial Monitor check the SD card for the `LAUNCH_UPLOAD_VERIFICATION_ID`.
> * Clear the SD card afterward.

## Troubleshooting

- If the temperature sensors read garbage values (e.g., `-242` or `242`):
- Check the wiring and ensure all connections are secure.
- If the issue persists:
 - Verify the wire order.
 - Compare against the PCB layout on the box drive.

## Incomplete Files
### LaunchNew.ino
New code that was in progress but not finished.

- Uses `MyAsync_Adafruit_MAX31865_library` to fix slow temperature sensor reading.
- Uses a folder system for each run to prevent needing to clear SD card (untested)

### TestAsyncTemp.ino
Used to test the new `MyAsync_Adafruit_MAX31865_library`.

### MyAsync_Adafruit_MAX31865_library
Move this entire folder into your Arduino `libraries` directory.
- Problem: each temperature sensor has a built in delay of 75 ms from the code provided by Adafruit. Calling 3 temperature sensors has a total delay of 225 ms. 
- Only two delays of 10 ms and 65 ms are needed to safely read each sensor.
- Attempted Fix: split `readRTD` into 3 parts. Only use one 10 ms delay and 65 ms delay for each temperature sensor.
- Unresolved: Program halts and waits with no error after calling any `readRTD` parts.
- Problem likely from my library code trying to look for a sensor with invalid information. Add new serial.print() to isolate issue. 
- Or: add async step functions into Adafruit_MAX31865_library and test this.






