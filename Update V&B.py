# Hand-Gesture 3-Second Controller (Shaka Trigger + Dual Camera)
import cv2
import time
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Utility: distance between two landmarks
def distance_px(lm, id1, id2, w, h):
    x1, y1 = int(lm.landmark[id1].x * w), int(lm.landmark[id1].y * h)
    x2, y2 = int(lm.landmark[id2].x * w), int(lm.landmark[id2].y * h)
    return hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

# Detect Shaka gesture (Thumb + Pinky out)
def is_shaka(hand_lm, w, h):
    tips = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
    pips = {'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}

    folded = sum(hand_lm.landmark[tips[f]].y > hand_lm.landmark[pips[f]].y
                 for f in ('index', 'middle', 'ring'))
    fingers_folded = folded == 3
    pinky_open = hand_lm.landmark[tips['pinky']].y < hand_lm.landmark[pips['pinky']].y

    wrist = hand_lm.landmark[0]
    thumb_tip = hand_lm.landmark[4]
    thumb_dist = hypot((thumb_tip.x - wrist.x) * w, (thumb_tip.y - wrist.y) * h)
    thumb_open = thumb_dist > 0.20 * w

    return fingers_folded and pinky_open and thumb_open

def main():
    # Pycaw volume setup
    interface = AudioUtilities.GetSpeakers().Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    minVol, maxVol, _ = volume.GetVolumeRange()

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=0.75,
                           min_tracking_confidence=0.75)
    draw = mp.solutions.drawing_utils

    # Camera setup
    cam_laptop = cv2.VideoCapture(0)  # Laptop camera
    cam_phone = cv2.VideoCapture(1)   # Phone Link camera
    current_cam = cam_laptop

    print("Camera started — Press 'p' to switch Laptop/ Phone")
    print("Show Shaka gesture to activate control (3 seconds)")

    active = False
    active_until = 0
    last_trigger = 0
    cooldown = 1.5

    while True:
        ok, frame = current_cam.read()
        if not ok:
            # Camera temporarily unavailable → skip frame
            print("⚠️ Frame not available, retrying...")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        h, w, _ = frame.shape
        now = time.time()

        if active and now > active_until:
            active = False

        # Detect Shaka gesture
        if res.multi_hand_landmarks:
            for hlm in res.multi_hand_landmarks:
                if is_shaka(hlm, w, h) and not active and (now - last_trigger) > cooldown:
                    active = True
                    active_until = now + 3
                    last_trigger = now
                    print("🟢 ACTIVE — next 3 seconds")

                draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        # Gesture control
        if active and res.multi_hand_landmarks and res.multi_handedness:
            left_lm, right_lm = None, None
            for h_info, hlm in zip(res.multi_handedness, res.multi_hand_landmarks):
                if h_info.classification[0].label == 'Left':
                    left_lm = hlm
                else:
                    right_lm = hlm

            # Brightness control — Left hand
            if left_lm:
                dist, p1, p2 = distance_px(left_lm, 4, 8, w, h)
                b_val = np.interp(dist, [50, 220], [0, 100])
                sbc.set_brightness(int(b_val))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f'Brightness: {int(b_val)}%', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

            # Volume control — Right hand
            if right_lm:
                dist, p1, p2 = distance_px(right_lm, 4, 8, w, h)
                vol_db = np.interp(dist, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol_db, None)
                vol_pct = np.interp(dist, [50, 220], [0, 100])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f'Volume: {int(vol_pct)}%', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

            cv2.putText(frame, f'ACTIVE — {int(active_until - now)}s',
                        (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'PAUSED — show Shaka sign to start',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 255), 2)

        cv2.imshow('Shaka-Trigger Hand Controller', frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit program
        if key == ord('q'):
            print("Program exited")
            break

        # Manual camera switch
        elif key == ord('p'):
            if current_cam == cam_laptop:
                current_cam = cam_phone
                print("Switched to Phone Camera")
            else:
                current_cam = cam_laptop
                print("Switched to Laptop Camera")

    cam_laptop.release()
    cam_phone.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
