# PREVIEW  
*Purdue Rocket Experimental Video in Educational Work*

## Project Overview

PREVIEW is a student-led aerospace project developing a self-contained payload capable of collecting sensor data and capturing video footage of a rocket’s exterior during hypersonic flight. The launch provider for this mission is PLUTO Aerospace. During flight, the payload is designed to withstand impulse accelerations of up to 150 g.

During fall 2025, our team manufactured the aluminum wedge structures, fabricated, and assembled the printed circuit board (PCB) through OSH Park, and developed the launch and data acquisition software. A temporary 3D printed version of the rocket body is used to verify the system can be assembled. 

## Technical Description
The payload is built around a Raspberry Pi Zero microcontroller paired with a Pi Camera, enabling video capture of the rocket’s exterior surface. The sensor suite includes:

- Two pressure transducers to measure at the body and wedge, enabling Mach number estimation  
- An ADXL375 accelerometer for high-G acceleration measurements  
- A K-type thermocouple with a MAX31855 breakout board for internal temperature monitoring  

Synchronized sensor data and video footage are used to evaluate the performance of a material applied to the rocket’s exterior under hypersonic conditions.

All electronic components are housed within a circular structural frame. The system stacks batteries, the microcontroller, and sensors across multiple disks separated by spacers, reducing mechanical stress during high-G launch events. Power is supplied by two 9V NiMH batteries, regulated down to 5V for onboard electronics.


## My Role  
*January 2025 – October 2025*
- System design considerations for electronics.
- Improved Solidworks CAD protoypes.
- PCB protoyping with breadboard and aided with Fusion 360 PCB layouts.
- Improved Python scripts for launch sequence control and sensor data acquisition.
- Configured Raspberry Pi to enable WiFi power management and auto-run launch software on power-up using systemd service files.






| | | |
|---|---|---|
| ![Camera Front Wedge](assets/camera_wedge_front.png)<br>Camera Front Wedge | ![Camera Bottom Wedge](assets/camera_wedge_bottom.png)<br>Camera Bottom Wedge | ![CAD Camera Front Wedge](assets/CAD_camera_wedge_front.png)<br>CAD Camera Front Wedge |
| ![Toggle Front Wedge](assets/toggle_wedge_front.png)<br>Toggle Front Wedge | ![Toggle Bottom Wedge](assets/toggle_wedge_bottom.png)<br>Toggle Bottom Wedge | ![CAD Toggle Front Wedge](assets/CAD_toggle_wedge_front.png)<br>CAD Toggle Front Wedge |
| ![Payload Frame Inside](assets/pt_frame_inside.png)<br>Payload Frame Inside | ![CAD Payload Frame Outside](assets/CAD_pt_frame_outside.png)<br>CAD Payload Frame Outside | ![CAD Battery Layout](assets/CAD_battery_layout.png)<br>CAD Battery Layout |
| ![Early Prototype](assets/breadboard_early_prototype_temp_accel.jpg)<br>Early Breadboard Prototype | ![Final Prototype](assets/breadboard_final_protoype_temp_accel_pressure_batteries.jpg)<br>Final Breadboard Prototype | ![PCB Design](assets/PCB_design.png)<br>PCB Design |
