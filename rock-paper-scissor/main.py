
import cv2
import mediapipe as mp
import os
import random
import time
import numpy as np
from dotenv import load_dotenv
import tkinter as tk

load_dotenv()


# Get screen height using tkinter
root = tk.Tk()
root.withdraw()
screen_height = root.winfo_screenheight()
screen_width = root.winfo_screenwidth()
root.destroy()

cam = int(os.environ.get("cam", 0))
cap = cv2.VideoCapture(cam)
hands = mp.solutions.hands.Hands(max_num_hands=1)
draws = mp.solutions.drawing_utils

# Get default camera size
ret, temp_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")
orig_h, orig_w = temp_frame.shape[:2]

# Set new height to 80% of screen height, keep aspect ratio
target_h = int(screen_height * 0.8 // 2)  # since canvas is 2x frame height
scale = target_h / orig_h
target_w = int(orig_w * scale)

# Set camera resolution (if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)


# Load PNGs for moves using absolute paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
move_imgs = {
    "rock": cv2.imread(os.path.join(script_dir, "rock.png"), cv2.IMREAD_UNCHANGED),
    "paper": cv2.imread(os.path.join(script_dir, "paper.png"), cv2.IMREAD_UNCHANGED),
    "scissor": cv2.imread(
        os.path.join(script_dir, "scissor.png"), cv2.IMREAD_UNCHANGED
    ),
}
move_names = ["rock", "paper", "scissor"]


def overlay_png(bg, fg, pos, border=False, max_height=None):
    """Overlay fg PNG with alpha onto bg at pos (x, y). Optionally add border. Optionally scale fg to max_height."""
    x, y = pos
    h, w = fg.shape[:2]
    # Scale fg if max_height is set and fg is too tall
    if max_height is not None and h > max_height:
        scale_factor = max_height / h
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        fg = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = fg.shape[:2]
    if y + h > bg.shape[0] or x + w > bg.shape[1]:
        return bg  # Don't draw if out of bounds
    if border:
        # Draw a simple black border
        cv2.rectangle(bg, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 0), 6)
    if fg.shape[2] == 4:
        alpha_fg = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y : y + h, x : x + w, c] = (
                alpha_fg * fg[:, :, c] + (1 - alpha_fg) * bg[y : y + h, x : x + w, c]
            )
    else:
        bg[y : y + h, x : x + w] = fg
    return bg


def get_move_from_fingers(fingers):
    # [0,0,0,0,0] = rock, [1,1,1,1,1] = paper, [0,1,1,0,0] = scissor
    if fingers == [0, 0, 0, 0, 0]:
        return "rock"
    if fingers == [1, 1, 1, 1, 1]:
        return "paper"
    if fingers == [0, 1, 1, 0, 0]:
        return "scissor"
    return None


def fingers_up(hand_landmarks):
    # Detect thumb+fingers up for both left and right hands
    tips = [4, 8, 12, 16, 20]
    lm = hand_landmarks.landmark
    fingers = []
    # Try to infer handedness from landmark positions
    # If thumb tip is to the left of wrist, it's likely right hand, else left hand
    is_right = lm[4].x < lm[0].x
    # Thumb
    if is_right:
        fingers.append(1 if lm[tips[0]].x < lm[tips[0] - 1].x else 0)
    else:
        fingers.append(1 if lm[tips[0]].x > lm[tips[0] - 1].x else 0)
    # Other fingers (same for both hands)
    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[tips[i] - 2].y else 0)
    return fingers


scores = [0, 0]  # [AI, Player]
stateResult = False
startGame = False
timer = 0
result = ""
player_move = None
ai_move = None


font = cv2.FONT_HERSHEY_SIMPLEX
base_h = 240  # typical webcam height



while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Resize frame to target size if needed
    frame = cv2.resize(frame, (target_w, target_h))
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_obj = hands.process(frameRGB)

    # Prepare a blank canvas: top for AI, bottom for player
    h, w = frame.shape[:2]
    font_scale = h / base_h
    # Clamp font_scale to a reasonable range (e.g., 0.7 to 1.2)
    font_scale = max(0.7, min(font_scale, 1.2))
    def scale(val):
        return int(val * font_scale)
    canvas = 255 * np.ones((h * 2, w, 3), dtype=np.uint8)
    # Place player camera on bottom
    canvas[h : h * 2, 0:w] = frame

    # Draw horizontal dividing line
    cv2.line(canvas, (0, h), (w, h), (180, 180, 180), scale(2))

    # Draw hand skeleton and box with handedness label on player side (bottom)
    if hand_obj.multi_hand_landmarks:
        # Use MediaPipe's handedness detection
        handedness_list = (
            hand_obj.multi_handedness if hasattr(hand_obj, "multi_handedness") else None
        )
        for idx, hand_landmarks in enumerate(hand_obj.multi_hand_landmarks):
            draws.draw_landmarks(
                canvas[h:, :], hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )
            # Get bounding box
            lm = hand_landmarks.landmark
            xs = [int(pt.x * w) for pt in lm]
            ys = [int(pt.y * h) for pt in lm]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # Use handedness from MediaPipe if available
            if handedness_list and idx < len(handedness_list):
                hand_label = handedness_list[idx].classification[0].label
            else:
                hand_label = "Hand"
            cv2.rectangle(canvas[h:, :], (x_min, y_min), (x_max, y_max), (0, 0, 0), scale(3))
            # Put handedness text above the box
            (label_w, label_h), _ = cv2.getTextSize(hand_label, font, font_scale, scale(2))
            label_x = x_min + (x_max - x_min) // 2 - label_w // 2
            label_y = max(y_min - scale(10), label_h + scale(5))
            cv2.putText(
                canvas[h:, :], hand_label, (label_x, label_y), font, font_scale, (0, 0, 0), scale(2)
            )

    # Centered score and result (top center)
    score_text = f"Score: AI {scores[0]} – You {scores[1]}"
    (score_w, score_h), _ = cv2.getTextSize(score_text, font, font_scale * 1.2, scale(3))
    cv2.putText(
        canvas, score_text, (w // 2 - score_w // 2, scale(40)), font, font_scale * 1.2, (0, 0, 0), scale(3)
    )

    if startGame:
        if not stateResult:
            timer = time.time() - initialTime
            prompt = f"Show your move: {2-int(timer)}"
            (prompt_w, _), _ = cv2.getTextSize(prompt, font, font_scale, scale(2))
            # Place prompt in the AI area (top half), below the score
            cv2.putText(
                canvas,
                prompt,
                (w // 2 - prompt_w // 2, scale(90)),
                font,
                font_scale,
                (0, 0, 255),
                scale(2),
            )
            if timer > 2:
                stateResult = True
                timer = 0
                player_move = None
                if hand_obj.multi_hand_landmarks:
                    hand_landmarks = hand_obj.multi_hand_landmarks[0]
                    fingers = fingers_up(hand_landmarks)
                    player_move = get_move_from_fingers(fingers)
                ai_move = random.choice(move_names)
                # Decide winner
                if player_move and ai_move:
                    if player_move == ai_move:
                        result = "Draw"
                    elif (
                        (player_move == "rock" and ai_move == "scissor")
                        or (player_move == "paper" and ai_move == "rock")
                        or (player_move == "scissor" and ai_move == "paper")
                    ):
                        result = "You Win!"
                        scores[1] += 1
                    else:
                        result = "AI Wins!"
                        scores[0] += 1
                else:
                    result = "Invalid move"
        else:
            # Show AI move on top, player move on bottom, with border
            if ai_move in move_imgs:
                # Limit hand image height to 30% of frame height
                max_img_h = int(h * 0.3)
                overlay_png(canvas, move_imgs[ai_move], (scale(50), scale(150)), border=False, max_height=max_img_h)
                cv2.putText(canvas, "AI", (scale(50), scale(140)), font, font_scale, (0, 0, 0), scale(2))
                # Move label
                if ai_move:
                    # Use the same max_img_h for label offset
                    cv2.putText(
                        canvas,
                        ai_move.capitalize(),
                        (scale(50), scale(150) + max_img_h + scale(40)),
                        font,
                        font_scale,
                        (0, 0, 0),
                        scale(2),
                    )
            if player_move in move_imgs:
                max_img_h = int(h * 0.3)
                overlay_png(canvas, move_imgs[player_move], (scale(50), h + scale(150)), border=False, max_height=max_img_h)
                cv2.putText(canvas, "You", (scale(50), h + scale(140)), font, font_scale, (0, 0, 0), scale(2))
                # Move label
                if player_move:
                    cv2.putText(
                        canvas,
                        player_move.capitalize(),
                        (scale(50), h + scale(150) + max_img_h + scale(40)),
                        font,
                        font_scale,
                        (0, 0, 0),
                        scale(2),
                    )
            # Centered result, larger and colored (top center)
            (res_w, _), _ = cv2.getTextSize(result, font, font_scale * 2, scale(4))
            color = (
                (0, 128, 0)
                if "Win" in result
                else ((0, 0, 255) if "AI" in result else (0, 0, 0))
            )
            cv2.putText(canvas, result, (w // 2 - res_w // 2, scale(110)), font, font_scale * 2, color, scale(4))

    else:
        # Centered prompt (bottom center)
        prompt = "Press 's' to start, 'q' to quit"
        (prompt_w, _), _ = cv2.getTextSize(prompt, font, font_scale, scale(2))
        cv2.putText(
            canvas, prompt, (w // 2 - prompt_w // 2, h * 2 - scale(40)), font, font_scale, (0, 0, 0), scale(2)
        )

    if stateResult and startGame:
        next_prompt = "Press 's' for next round"
        (next_w, _), _ = cv2.getTextSize(next_prompt, font, font_scale, scale(2))
        cv2.putText(
            canvas,
            next_prompt,
            (w // 2 - next_w // 2, h * 2 - scale(10)),
            font,
            font_scale,
            (0, 0, 0),
            scale(2),
        )

    cv2.imshow("Rock Paper Scissor", canvas)
    key = cv2.waitKey(1)
    if key == ord("s") and not startGame:
        startGame = True
        initialTime = time.time()
        stateResult = False
        result = ""
    if key == ord("s") and startGame and stateResult:
        startGame = True
        initialTime = time.time()
        stateResult = False
        result = ""
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
