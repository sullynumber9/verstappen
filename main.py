import cv2
import torch
import pygame
import threading

# === SETTINGS ===
SHOW_FEED = False  # Set to True to show camera feed, False to hide
MP3_FILE = "verstappen.mp3"

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(MP3_FILE)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Classes to detect
BIKE_CLASSES = ['bicycle', 'motorbike']

# Prevent repeated playback
playing = False

def play_sound():
    global playing
    playing = True
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass  # Wait until playback is done
    playing = False

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print("Bike detection running... Press Ctrl+C to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for *box, conf, cls in results.xyxy[0]:
            class_name = model.names[int(cls)]
            if class_name in BIKE_CLASSES:
                print(f"Detected {class_name} with confidence {conf:.2f}")

                # Draw rectangle and label if showing feed
                if SHOW_FEED:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Play sound if not already playing
                if not playing:
                    threading.Thread(target=play_sound).start()

        # Show camera feed if enabled
        if SHOW_FEED:
            cv2.imshow("Bike Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
