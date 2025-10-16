# Volume and Brightness Control Using Hand Gestures
<h3>Project:</h3> Volume and Brightness Control Using Hand Gestures
<h3>Author:</h3> Akash Khanra 
<h3>Status:</h3> Prototype / Demo

<h1>Overview</h1>
A real-time desktop application that uses a webcam to control system volume and screen brightness using hand gestures. The project uses MediaPipe for hand landmark detection, OpenCV for video capture & visualization, pycaw to control Windows audio, and screen_brightness_control to set brightness.

The app includes a 3‑second "Shaka" trigger (thumb + pinky open, other fingers folded) — when the Shaka sign is shown, the controller becomes active for 3 seconds, and during that window:

--Left hand (thumb + index) controls screen brightness.

--Right hand (thumb + index) controls volume.

A small cooldown prevents accidental retriggers.

<h1>Features</h1>
Shaka sign (thumb + pinky) to activate a 3-second control window

Left hand: brightness control (thumb–index distance)

Right hand: volume control (thumb–index distance)

Visual feedback with lines, % values, and a countdown timer on the camera feed

Safety cooldown to avoid accidental re-triggers

<h1>Requirements</h1>
This project was developed and tested on Windows. 
<h3>Expected working environment:</h3>

--Python 3.8+ (3.10 recommended)

--A webcam (built-in or phone-as-webcam)

Windows OS (for pycaw). screen_brightness_control has cross-platform support but may require additional permissions.

<h3>Python packages (installable via pip):</h3>

opencv-python
mediapipe
numpy
screen_brightness_control
pycaw
comtypes


<h3>optionally for packaging/run</h3>
python-dotenv
I recommend creating a venv.

<h1>Installation (Quick)</h1>
<h3>1. Clone the repository:</h3>
git clone https://github.com/khanra321/VaB.git

cd VaB
<h3>2. Create and activate a virtual environment:</h3>
python -m venv .venv
<h4>Windows</h4> 
.venv\Scripts\activate
<h4>macOS / Linux</h4>
source .venv/bin/activate

<h3>3. Install dependencies</h3>
pip install -r requirements.txt
<h3>4. Run the code:</h3>
python shaka_controller.py

(Replace the script name if your main file differs.)

<h3>5. Create exe file </h3>
Copy and save this code in a new file. Open a terminal and create an exe file. After that, you use it. 
<h3>6. Run the app:</h3>
Open the exe file. It takes permission from you.

<h1>Usage</h1>
The webcam window will open.

Stand so your hands are visible to the camera.

Show the Shaka sign (thumb + pinky extended, other fingers folded) to activate a 3-second control window.

During the active window:

Use your left hand: move the thumb and index finger to change brightness. The app maps distance to % (0–100).

Use your right hand: move thumb and index to change volume. The app maps distance to dB/percentage.

The HUD shows current Brightness, Volume, and remaining active seconds.

Press q to exit.

<h1>Controls & Tuning</h1>
active_until / 3 seconds — duration of the active control window. Change to extend/reduce control time.

cooldown — seconds to wait between triggers (default 1.5).

is_shaka() — detection thresholds (thumb distance threshold, folded finger checks) can be tuned for different camera setups.

np.interp(dist, [50, 220], [0, 100]) — maps measured pixel distance to 0–100; adjust [50, 220] to match your camera distance and resolution.

<h1>Troubleshooting</h1>
Camera not opening: Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1) or other indexes. If using a phone-as-webcam app, ensure it is connected and recognized as a video device.

Volume control not working: pycaw works on Windows only. Ensure comtypes is installed and you run on Windows with proper audio drivers.

Brightness control not working: screen_brightness_control requires support from the OS and display drivers. On multi-monitor setups, you can specify the display index.

Shaka not detected reliably: Tweak the thresholds in is_shaka() — especially the thumb_dist > 0.20 * w threshold and the finger fold checks. Also, adjust min_detection_confidence.

##  Dual-Camera Version "Update V&B"
<h3>This version includes:</h3>

Support for **two cameras** (e.g., laptop + phone webcam).

- Press "P" to switch between the Laptop and Phone camera instantly.
- 
- Works exactly like the base version otherwise.
- 
- Useful for testing with different angles or higher-quality phone cameras.

