import cv2
import csv
import time
import os
import io
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp

# -------------------- CONFIG --------------------
FOLDER_ID         = os.environ.get("FOLDER_ID", "1j6llniO3bjkBWIevzJk3xlIjO4UhH0cc")
CREDS_FILE        = os.environ.get("CREDS_FILE", "service_account.json")
OUTPUT_CSV        = "attention_results.csv"
LATEST_IMAGE_FILE = "latest_frame.jpg"

POLL_SECONDS      = 5
MAX_CSV_ROWS      = 2000
MAX_DRIVE_FILES   = 100  

# -------------------- INIT MODELS --------------------
print("Initializing Enhanced Classroom AI (Face + Pose)...")
model = YOLO("yolov8n-face-lindevs.pt")

# n_init=1 is critical for Drive snapshots to show results immediately
tracker = DeepSort(max_age=30, n_init=1) 

# Mediapipe for Face and Pose (to detect hands)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=20, refine_landmarks=True)
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# -------------------- DRIVE AUTH --------------------
if not os.path.exists(CREDS_FILE):
    print(f"Error: {CREDS_FILE} not found!")
    exit()

creds = service_account.Credentials.from_service_account_file(
    CREDS_FILE, scopes=["https://www.googleapis.com/auth/drive"]
)
drive = build("drive", "v3", credentials=creds)

# -------------------- CSV SETUP --------------------
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "student_id", "attention_score", "head_turned", "eyes_closed", "hand_raised"])

# -------------------- ANALYTICS LOGIC --------------------

def analyze_engagement(frame, face_bbox):
    """
    Returns (score, head_turned, eyes_closed, hand_raised)
    """
    x, y, w, h = face_bbox
    
    # 1. Pose Analysis (Check for Hand Raising)
    # Crop a wider area around the person to see arms
    pad = 120
    cy1, cx1 = max(0, y - pad), max(0, x - pad)
    cy2, cx2 = min(frame.shape[0], y + h + pad), min(frame.shape[1], x + w + pad)
    body_crop = frame[cy1:cy2, cx1:cx2]
    
    hand_raised = False
    if body_crop.size > 0:
        rgb_body = cv2.cvtColor(body_crop, cv2.COLOR_BGR2RGB)
        pose_res = pose_detector.process(rgb_body)
        if pose_res.pose_landmarks:
            pm = pose_res.pose_landmarks.landmark
            # Check if wrists (15, 16) are above the nose (0)
            if pm[15].y < pm[0].y or pm[16].y < pm[0].y:
                hand_raised = True

    # 2. Face Analysis
    face_img = frame[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
    if face_img.size == 0: return 50, False, False, False

    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_res = face_mesh.process(rgb_face)
    
    score = 75 # Starting Base Score
    head_turned, eyes_closed, mouth_participation = False, False, False

    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0].landmark
        
        # Head Pose (Lenient: 0.18 threshold for classroom focus)
        nose_x = lm[1].x
        if abs(nose_x - 0.5) > 0.18:
            head_turned = True
            score -= 30
        
        # Eyes (EAR proxy)
        ear = (abs(lm[159].y - lm[145].y) + abs(lm[386].y - lm[374].y)) / 2.0
        if ear < 0.0065:
            eyes_closed = True
            score -= 40
            
        # Mouth Participation (Mouth open = talking/answering)
        mouth_gap = abs(lm[13].y - lm[14].y)
        if mouth_gap > 0.02:
            mouth_participation = True
            score += 15

    # Hand Raising Bonus
    if hand_raised:
        score += 40
        
    return min(100, max(0, int(score))), head_turned, eyes_closed, hand_raised

def preprocess_frame(frame):
    """Enhance image for better detection in low light/distant shots"""
    h, w = frame.shape[:2]
    if w < 1000: # Upscale small images
        frame = cv2.resize(frame, (1280, int(h * 1280/w)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# -------------------- CORE PROCESSING --------------------

def process_image(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None: return

    frame = preprocess_frame(frame)
    
    # YOLO Face Detection
    results = model(frame, verbose=False, conf=0.20)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, _ = r
        # Format: [left, top, w, h]
        detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(conf), "face"))

    # Update Tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logged_count = 0
    for track in tracks:
        # We allow 'tentative' to solve the "Tracked 0" issue on single images
        if not (track.is_confirmed() or track.is_tentative()):
            continue
            
        tid = track.track_id
        ltwh = list(map(int, track.to_ltwh()))
        x, y, w, h = ltwh
        
        # Comprehensive Engagement Analysis
        score, head, eyes, hand = analyze_engagement(frame, ltwh)
        logged_count += 1

        # Annotation
        color = (0, 255, 0) if score > 75 else (0, 165, 255) if score > 45 else (0, 0, 255)
        if hand: color = (255, 255, 0) # Cyan for hand raised
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"ID:{tid} {score}%"
        if hand: label += " [HAND]"
        cv2.putText(frame, label, (x, max(0, y-10)), 0, 0.6, color, 2)

        # Write to CSV
        with open(OUTPUT_CSV, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, tid, score, int(head), int(eyes), int(hand)])

    print(f"[{timestamp}] YOLO: {len(detections)} | Tracker: {logged_count} active IDs")
    cv2.imwrite(LATEST_IMAGE_FILE, frame)


# -------------------- MAIN LOOP --------------------
processed_files = set()
file_queue = deque()

print("Watcher started. Detecting Hands, Participation, and Attention.")
print(f"Monitoring Folder: {FOLDER_ID}\n")

while True:
    try:
        # IMPROVED QUERY: Look for any image type, not just jpeg
        query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed=false"
        
        results = drive.files().list(
            q=query, 
            fields="files(id, name, mimeType)",
            pageSize=10 # Look for the 10 most recent
        ).execute()
        
        files = results.get("files", [])
        
        # Debug: Uncomment the line below to see EVERY file the API finds
        # print(f"DEBUG: API found {len(files)} total images in folder")

        new_files = [f for f in files if f["id"] not in processed_files]

        if new_files:
            # Process oldest to newest among the new batch
            for f in reversed(new_files): 
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing: {f['name']} ({f['mimeType']})")
                
                request = drive.files().get_media(fileId=f["id"])
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                
                process_image(fh.getvalue())
                processed_files.add(f["id"])
                file_queue.append(f["id"])
                
                # Cleanup Drive (Keep last 100 images)
                if len(file_queue) > MAX_DRIVE_FILES:
                    old_id = file_queue.popleft()
                    try: 
                        drive.files().delete(fileId=old_id).execute()
                        print(f"Cleaned up old image ID: {old_id}")
                    except: pass
        else:
            # Show a pulsing scan so you know it's alive
            print(f"Scanning... {datetime.now().strftime('%H:%M:%S')} | Total Processed: {len(processed_files)}", end="\r")

    except Exception as e:
        print(f"\nDrive Error: {e}")
        time.sleep(10)
    
    time.sleep(POLL_SECONDS)