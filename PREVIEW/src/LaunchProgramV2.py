import time
import board
import busio
import adafruit_max31855
import digitalio
import adafruit_adxl37x
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from datetime import datetime
import csv
import os
import sys

# Initialize I2C for ADXL375
i2c = busio.I2C(board.SCL, board.SDA)
accelerometer = adafruit_adxl37x.ADXL375(i2c)

# Initialize SPI for MAX31855
spi = board.SPI()
28
cs = digitalio.DigitalInOut(board.D5)
max31855 = adafruit_max31855.MAX31855(spi, cs)

# Initialize ADS1115 for pressure transducers
ads = ADS1115(i2c)
ads.gain = 1 # ±4.096 V input range
pt1 = AnalogIn(ads, ADS.P0) # 30 PSI sensor
pt2 = AnalogIn(ads, ADS.P1) # 100 PSI sensor

# Initialize Camera
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder(bitrate=10000000)
picam2.start()

# Prepare Logging Directory
session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_dir = f"/media/aae418/09B6-7233/session_{session_start}"
os.makedirs(session_dir, exist_ok=True)

# CSV Logging
csv_path = os.path.join(session_dir, "data_log.csv")
last_video_file = os.path.join(session_dir, ".last_video.txt")

try: 
  with open(csv_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
    "timestamp", "ax_m_s2", "ay_m_s2", "az_m_s2", "temp_C",
    "volt_pt1", "pressure_pt1",
    "volt_pt2", "pressure_pt2"
    ])
    print("Recording started...")

    # Start first video recording
    video_index = 1
    video_start_time = time.monotonic()
    current_video = os.path.join(session_dir, f"video_part_{video_index}.h264")
    picam2.start_recording(encoder, current_video)
    with open(last_video_file, "w") as f:
        f.write(current_video)
        
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x, y, z = accelerometer.acceleration
        temp_c = max31855.temperature

        # PT1 (30 PSI sensor)
        volt1 = pt1.voltage / 0.89397084
        pressure1 = (volt1 - 0.5) * (30.0 / 4.0) * 1.101761 # 0.5–4.5 V → 0–30 PSI
        #This second adjust factor is for battery adjustment

        # PT2 (100 PSI sensor)
        volt2 = pt2.voltage / 0.89397084
        pressure2 = (volt2 - 0.5) * (100.0 / 4.0) * 1.154501 # 0.5–4.5 V → 0–100 PSI
        # Log to CSV
        writer.writerow([
        now, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{temp_c:.2f}",
        f"{volt1:.3f}", f"{pressure1:.2f}",
        f"{volt2:.3f}", f"{pressure2:.2f}"
        ])
        csv_file.flush()

        f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}, Temp={temp_c:.2f} °C | "
        f"PT1: {volt1:.3f} V → {pressure1:.2f} psi | "
        f"PT2: {volt2:.3f} V → {pressure2:.2f} psi")
        # Rotate video every 100 seconds

        if time.monotonic() - video_start_time >= 100:
        picam2.stop_recording()
        video_index += 1
        video_start_time = time.monotonic()
        current_video = os.path.join(session_dir,
        f"video_part_{video_index}.h264")
        picam2.start_recording(encoder, current_video)
        with open(last_video_file, "w") as f:
            f.write(current_video)
        # time.sleep(0.05)

except KeyboardInterrupt:
    print("Interrupted by user.")
except Exception as e:
    print(f"Error occurred: {e}")
    sys.exit(1)
finally:
    try:
        picam2.stop_recording()
    except Exception:
        pass
    try:
        cs.deinit()
    except:
        pass
    try:
        spi.deinit()
    except:
        pass
    print("Cleanup complete.")