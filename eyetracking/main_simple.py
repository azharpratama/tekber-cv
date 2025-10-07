import cv2
import mediapipe as mp
import time
import math
from playsound import playsound

# --- KONFIGURASI DAN THRESHOLD ---
# Mata
EYE_CLOSED_THRESHOLD = 0.015
PERCLOS_WINDOW = 20
PERCLOS_THRESHOLD_WARN = 0.20
PERCLOS_THRESHOLD_DANGER = 0.30

# Menguap
YAWN_THRESHOLD = 0.05
YAWN_COUNT_WARN = 3

# Kepala
HEAD_NOD_THRESHOLD = 0.04
HEAD_NOD_TIME_DANGER = 1.5

# Inisialisasi MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Inisialisasi variabel state
perclos_data = []
yawn_counter = 0
head_nod_start_time = None
alert_level = 0
alarm_triggered_for_event = False # Penanda agar alarm hanya berbunyi sekali per kejadian

def play_alarm():
    """Fungsi untuk memainkan suara alarm."""
    try:
        playsound('tekber-cv/eyetracking/alarm.wav', block=False)
    except Exception as e:
        print(f"Error memainkan suara: {e}. Pastikan file 'alarm.wav' ada di folder yang sama.")

# Buka webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 1. Analisis Mata (PERCLOS)
            top_lid_y = face_landmarks.landmark[159].y
            bottom_lid_y = face_landmarks.landmark[145].y
            eye_height_ratio = abs(top_lid_y - bottom_lid_y)
            is_eye_closed = eye_height_ratio < EYE_CLOSED_THRESHOLD
            perclos_data.append(is_eye_closed)
            if len(perclos_data) > PERCLOS_WINDOW * 30: perclos_data.pop(0)
            current_perclos = sum(perclos_data) / len(perclos_data) if perclos_data else 0

            # 2. Analisis Mulut (Menguap)
            top_lip_y = face_landmarks.landmark[13].y
            bottom_lip_y = face_landmarks.landmark[14].y
            mouth_height_ratio = abs(top_lip_y - bottom_lip_y)
            if mouth_height_ratio > YAWN_THRESHOLD:
                yawn_counter += 0.1
            
            # 3. Analisis Kepala (Anggukan)
            nose_y = face_landmarks.landmark[1].y
            chin_y = face_landmarks.landmark[152].y
            head_tilt_ratio = abs(nose_y - chin_y)
            is_head_nodding = head_tilt_ratio < HEAD_NOD_THRESHOLD
            
            head_nod_duration = 0
            if is_head_nodding:
                if head_nod_start_time is None:
                    head_nod_start_time = time.time()
                head_nod_duration = time.time() - head_nod_start_time
            else:
                head_nod_start_time = None

            # --- TENTUKAN LEVEL PERINGATAN (LOGIKA BARU) ---
            # Cek kondisi bahaya pada frame saat ini
            is_in_danger = head_nod_duration > HEAD_NOD_TIME_DANGER or current_perclos > PERCLOS_THRESHOLD_DANGER
            is_in_warning = current_perclos > PERCLOS_THRESHOLD_WARN or int(yawn_counter) >= YAWN_COUNT_WARN
            
            if is_in_danger:
                alert_level = 2
                if not alarm_triggered_for_event:
                    play_alarm()
                    alarm_triggered_for_event = True # Set penanda agar alarm tidak berulang
            elif is_in_warning:
                alert_level = 1
                alarm_triggered_for_event = False # Reset penanda alarm jika turun ke level waspada
            else:
                alert_level = 0
                alarm_triggered_for_event = False # Reset penanda alarm jika kondisi aman
                if int(yawn_counter) >= YAWN_COUNT_WARN:
                    yawn_counter = 0

            # --- GAMBAR TAMPILAN VISUAL ---
            cv2.putText(frame, f"PERCLOS (20s): {current_perclos:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Menguap: {int(yawn_counter)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if alert_level == 1:
                cv2.putText(frame, "WASPADA", (w // 3, h // 2), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)
            elif alert_level == 2:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
                cv2.putText(frame, "BAHAYA! ANDA MENGANTUK!", (w // 6, h // 2), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 3)

        cv2.imshow('Advanced Drowsiness Detection System', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()