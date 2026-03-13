import numpy as np

def calculate_angle(a, b, c):
    """Calculates the 3D angle between three landmarks."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 2)

def get_posture_feedback(landmarks):
    """
    Identifies technical deviations in specific body segments as per abstract goals.
    Provides instant visual/textual feedback for head, knees, and balance.
    """
    feedback = []
    
    # 1. Head Position Check (Abstract Requirement: head segment correction)
    # Landmark 0: Nose
    head = landmarks[0]
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    if head.y > shoulder_y:
        feedback.append("Correction: Keep your head level; it is dropping below shoulder level.")

    # 2. Knee & Stance Balance (Abstract Requirement: actionable improvement)
    # Landmarks 25: L_Knee, 26: R_Knee
    knee_diff = abs(landmarks[25].y - landmarks[26].y)
    if knee_diff > 0.12:
        feedback.append("Correction: Balance your weight; your lead knee is collapsing.")

    return feedback if feedback else ["Technique is stable. Maintain this posture."]

def classify_shot(metrics):
    """Classifies the shot based on average elbow angle."""
    avg_elbow = metrics.get("average_elbow_angle", 0)
    if avg_elbow > 150:
        return "Full Extension (Lofted Shot / Drive)"
    elif 110 < avg_elbow <= 150:
        return "Balanced Stance (Ideal Defensive/Push)"
    else:
        return "Compact Posture (Forward Defense)"