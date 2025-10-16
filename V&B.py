# Hand‑Gesture 3‑Second Controller (Shaka Trigger)

import cv2
import time
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

#  Utility: Get distance between two landmarks

def distance_px(lm, id1, id2, w, h):
    x1, y1 = int(lm.landmark[id1].x * w), int(lm.landmark[id1].y * h)
    x2, y2 = int(lm.landmark[id2].x * w), int(lm.landmark[id2].y * h)
    return hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

#  Detect Shaka Sign (Thumb + Pinky open, other 3 folded)

def is_shaka(hand_lm, w, h):
    # Tip and PIP landmark indices
    tips = { 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20 }
    pips = { 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18 }

    # 1) Check if index, middle, ring fingers are folded
    folded = 0
    for finger in ('index', 'middle', 'ring'):
        if hand_lm.landmark[tips[finger]].y > hand_lm.landmark[pips[finger]].y:
            folded += 1
    fingers_folded = folded == 3

    # 2) Pinky is up
    pinky_open = hand_lm.landmark[tips['pinky']].y < hand_lm.landmark[pips['pinky']].y

    # 3) Thumb is out (based on wrist-tip distance)
    wrist = hand_lm.landmark[0]
    thumb_tip = hand_lm.landmark[4]
    thumb_dist = hypot((thumb_tip.x - wrist.x) * w, (thumb_tip.y - wrist.y) * h)
    thumb_open = thumb_dist > 0.20 * w

    return fingers_folded and pinky_open and thumb_open

#  Main Function

def main():
    #  Pycaw volume control setup
    interface = AudioUtilities.GetSpeakers().Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    minVol, maxVol, _ = volume.GetVolumeRange()

    #  Mediapipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=.75,
                           min_tracking_confidence=.75)
    draw = mp.solutions.drawing_utils

    # Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera"); return
    print("Camera started — show Shaka sign to begin")

    # Timer control variables
    active = False
    active_until = 0
    last_trigger = 0
    cooldown = 1.5  # Prevent accidental rapid re-trigger

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)                        # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        h, w, _ = frame.shape
        now = time.time()

        #  Deactivate after timeout
        if active and now > active_until:
            active = False

        #  Check for Shaka Sign
        if res.multi_hand_landmarks:
            for hlm in res.multi_hand_landmarks:
                if is_shaka(hlm, w, h) and not active and (now - last_trigger) > cooldown:
                    active = True
                    active_until = now + 3
                    last_trigger = now
                    print(" ACTIVE — next 3 seconds")

                draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        #  If active, control brightness and volume
        if active and res.multi_hand_landmarks and res.multi_handedness:
            left_lm, right_lm = None, None
            for h_info, hlm in zip(res.multi_handedness, res.multi_hand_landmarks):
                if h_info.classification[0].label == 'Left':
                    left_lm = hlm
                else:
                    right_lm = hlm

            #  Brightness — Left hand Thumb + Index
            if left_lm:
                dist, p1, p2 = distance_px(left_lm, 4, 8, w, h)
                b_val = np.interp(dist, [50, 220], [0, 100])
                sbc.set_brightness(int(b_val))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f'Brightness: {int(b_val)}%', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

            #  Volume — Right hand Thumb + Index
            if right_lm:
                dist, p1, p2 = distance_px(right_lm, 4, 8, w, h)
                vol_db = np.interp(dist, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol_db, None)
                vol_pct = np.interp(dist, [50, 220], [0, 100])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f'Volume: {int(vol_pct)}%', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)

            #  Show countdown timer
            cv2.putText(frame, f'ACTIVE — {int(active_until - now)}s',
                        (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (255, 255, 0), 2)

        else:
            cv2.putText(frame, 'PAUSED — show Shaka sign to start',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7,
                        (0, 255, 255), 2)

        #  Show live window
        cv2.imshow('Shaka‑Trigger Hand Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Program exited"); break

    cap.release()
    cv2.destroyAllWindows()



#  Run main

if __name__ == '__main__':
    main()