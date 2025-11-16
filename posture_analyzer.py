"""Posture analysis logic"""
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from posture_metrics import PostureMetrics, PostureIssues


class PostureAnalyzer:
    """Analyzes body posture from MediaPipe pose landmarks"""
    
    # Thresholds for posture detection
    FORWARD_HEAD_THRESHOLD = 0.05  # Normalized distance threshold
    HEAD_TOO_CLOSE_THRESHOLD = -0.9  # Z-depth threshold (closer = more negative)
    SLOUCH_ANGLE_THRESHOLD = 160  # Degrees
    LEANING_THRESHOLD = 0.03  # Normalized height difference
    HOLLOW_BACK_MIN_ANGLE = 165  # Minimum angle for normal back
    HOLLOW_BACK_MAX_ANGLE = 195  # Maximum angle for normal back
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """
        Calculate angle at point2 formed by three points
        Returns angle in degrees
        """
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_forward_head(self, ear: Tuple[float, float], 
                            shoulder: Tuple[float, float]) -> float:
        """
        Measure forward head posture
        Returns horizontal distance between ear and shoulder (normalized)
        """
        return abs(ear[0] - shoulder[0])
    
    def analyze_slouching(self, shoulder: Tuple[float, float], 
                         hip: Tuple[float, float], 
                         knee: Tuple[float, float]) -> float:
        """
        Measure back angle for slouching detection
        Returns angle in degrees (should be close to 180° for good posture)
        """
        return self.calculate_angle(shoulder, hip, knee)
    
    def analyze_leaning(self, left_shoulder: Tuple[float, float], 
                       right_shoulder: Tuple[float, float]) -> float:
        """
        Measure shoulder height difference for leaning detection
        Returns normalized height difference
        """
        return abs(left_shoulder[1] - right_shoulder[1])
    
    def analyze_hollow_back(self, shoulder: Tuple[float, float], 
                           hip: Tuple[float, float], 
                           ankle: Tuple[float, float]) -> float:
        """
        Measure hip-shoulder angle for hollow back detection
        Returns angle in degrees (should be ~180° for normal posture)
        """
        return self.calculate_angle(shoulder, hip, ankle)
    
    def get_landmarks_coords(self, landmarks, landmark_ids: list) -> list:
        """Extract coordinates for specified landmark IDs"""
        coords = []
        for lid in landmark_ids:
            lm = landmarks.landmark[lid]
            coords.append((lm.x, lm.y, lm.z))
        return coords
    
    def analyze_posture(self, landmarks) -> Tuple[Optional[PostureMetrics], PostureIssues]:
        """
        Analyze all posture metrics from pose landmarks
        Returns tuple of (metrics, issues)
        """
        if not landmarks:
            return None, PostureIssues()
        
        # Extract key landmark coordinates (x, y, z)
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Use average of left and right landmarks for calculations
        ear = ((left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2)
        ear_z = (left_ear.z + right_ear.z) / 2  # Z-depth for distance from camera
        shoulder = ((left_shoulder.x + right_shoulder.x) / 2, 
                   (left_shoulder.y + right_shoulder.y) / 2)
        hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        knee = ((left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2)
        ankle = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)
        
        # Calculate metrics
        forward_head = self.analyze_forward_head(ear, shoulder)
        head_distance = ear_z  # Normalized z-depth (more negative = closer to camera)
        back_angle = self.analyze_slouching(shoulder, hip, knee)
        shoulder_height_diff = self.analyze_leaning(
            (left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)
        )
        hip_shoulder_angle = self.analyze_hollow_back(shoulder, hip, ankle)
        
        # Create metrics object
        metrics = PostureMetrics(
            forward_head=forward_head,
            head_distance=head_distance,
            back_angle=back_angle,
            shoulder_height_diff=shoulder_height_diff,
            hip_shoulder_angle=hip_shoulder_angle
        )
        
        # Detect issues
        issues = PostureIssues(
            forward_head_posture=forward_head > self.FORWARD_HEAD_THRESHOLD,
            too_close_to_monitor=head_distance < self.HEAD_TOO_CLOSE_THRESHOLD,
            slouching=back_angle < self.SLOUCH_ANGLE_THRESHOLD,
            leaning=shoulder_height_diff > self.LEANING_THRESHOLD,
            hollow_back=(hip_shoulder_angle < self.HOLLOW_BACK_MIN_ANGLE or 
                        hip_shoulder_angle > self.HOLLOW_BACK_MAX_ANGLE)
        )
        
        return metrics, issues
