import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def detect_shot_type(avg_angle):
    if avg_angle > 158: return "Cover / Straight Drive"
    elif 140 <= avg_angle <= 158: return "Defensive Stroke"
    else: return "Pull / Cut Shot"

def process_video(video_path, output_path, report_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Write to a temporary file first (Browsers can't play raw OpenCV output)
    temp_output = output_path.replace(".mp4", "_temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    angles = []
    frames_processed = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frames_processed += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                try:
                    lm = results.pose_landmarks.landmark
                    s = [lm[11].x, lm[11].y]
                    e = [lm[13].x, lm[13].y]
                    w = [lm[15].x, lm[15].y]
                    angles.append(calculate_angle(s, e, w))
                except: pass
            out.write(image)
            
    cap.release()
    out.release()

    # --- CRITICAL: FFmpeg Conversion for Mobile/Web Playback ---
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', temp_output, 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', 
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(temp_output): os.remove(temp_output)
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        os.rename(temp_output, output_path)

    avg_angle = int(np.mean(angles)) if angles else 0
    shot_name = detect_shot_type(avg_angle)
    is_compact = avg_angle > (145 if "Drive" in shot_name else 120)
    
    feedback = [
        f"Detected Shot: {shot_name}",
        f"Average Elbow Extension: {avg_angle}°",
        f"Posture Status: {'COMPACT' if is_compact else 'COLLAPSED'}",
        f"Total Frames Analyzed: {frames_processed}"
    ]
    feedback.append("COACHING: Excellent form." if is_compact else "CORRECTION: Lead arm is winging.")
    
    generate_pdf(report_path, feedback, avg_angle, frames_processed)
    return bool(is_compact), feedback

def generate_pdf(path, feedback, avg_angle, frames):
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 20); c.drawString(100, 750, "Cricket AI Report")
    c.setFont("Helvetica", 12); c.drawString(100, 720, f"Frames: {frames} | Angle: {avg_angle}°")
    y = 650
    for item in feedback:
        c.drawString(110, y, f"• {item}"); y -= 25
    c.save()