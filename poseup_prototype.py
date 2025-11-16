import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PostureMetrics:
    """Stores calculated posture metrics"""
    forward_head: float  # Distance in pixels (normalized)
    back_angle: float  # Degrees
    shoulder_angle: float  # Degrees
    shoulder_height_diff: float  # Normalized difference
    hip_shoulder_angle: float  # Degrees
    

@dataclass
class PostureIssues:
    """Flags for detected posture problems"""
    forward_head_posture: bool = False
    slouching: bool = False
    rounded_shoulders: bool = False
    leaning: bool = False
    hollow_back: bool = False


class PostureAnalyzer:
    """Analyzes body posture from MediaPipe pose landmarks"""
    
    # Thresholds for posture detection
    FORWARD_HEAD_THRESHOLD = 0.05  # Normalized distance threshold
    SLOUCH_ANGLE_THRESHOLD = 160  # Degrees
    ROUNDED_SHOULDER_THRESHOLD = 30  # Degrees from vertical
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
    
    def analyze_rounded_shoulders(self, shoulder: Tuple[float, float], 
                                  elbow: Tuple[float, float]) -> float:
        """
        Measure shoulder rounding
        Returns angle from vertical (0° = perfect vertical)
        """
        # Calculate angle from vertical
        vertical_point = (shoulder[0], shoulder[1] + 0.1)
        angle = self.calculate_angle(vertical_point, shoulder, elbow)
        return abs(90 - angle)  # Deviation from 90° (perpendicular)
    
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
        shoulder = ((left_shoulder.x + right_shoulder.x) / 2, 
                   (left_shoulder.y + right_shoulder.y) / 2)
        hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        knee = ((left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2)
        ankle = ((left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2)
        
        # Calculate metrics
        forward_head = self.analyze_forward_head(ear, shoulder)
        back_angle = self.analyze_slouching(shoulder, hip, knee)
        shoulder_angle_left = self.analyze_rounded_shoulders(
            (left_shoulder.x, left_shoulder.y), (left_elbow.x, left_elbow.y)
        )
        shoulder_angle_right = self.analyze_rounded_shoulders(
            (right_shoulder.x, right_shoulder.y), (right_elbow.x, right_elbow.y)
        )
        shoulder_angle = (shoulder_angle_left + shoulder_angle_right) / 2
        shoulder_height_diff = self.analyze_leaning(
            (left_shoulder.x, left_shoulder.y), (right_shoulder.x, right_shoulder.y)
        )
        hip_shoulder_angle = self.analyze_hollow_back(shoulder, hip, ankle)
        
        # Create metrics object
        metrics = PostureMetrics(
            forward_head=forward_head,
            back_angle=back_angle,
            shoulder_angle=shoulder_angle,
            shoulder_height_diff=shoulder_height_diff,
            hip_shoulder_angle=hip_shoulder_angle
        )
        
        # Detect issues
        issues = PostureIssues(
            forward_head_posture=forward_head > self.FORWARD_HEAD_THRESHOLD,
            slouching=back_angle < self.SLOUCH_ANGLE_THRESHOLD,
            rounded_shoulders=shoulder_angle > self.ROUNDED_SHOULDER_THRESHOLD,
            leaning=shoulder_height_diff > self.LEANING_THRESHOLD,
            hollow_back=(hip_shoulder_angle < self.HOLLOW_BACK_MIN_ANGLE or 
                        hip_shoulder_angle > self.HOLLOW_BACK_MAX_ANGLE)
        )
        
        return metrics, issues


class PoseUpApp:
    """Main application for real-time posture monitoring"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.analyzer = PostureAnalyzer()
        self.cap = None
        
    def draw_posture_info(self, frame: np.ndarray, 
                         metrics: Optional[PostureMetrics], 
                         issues: PostureIssues) -> np.ndarray:
        """Draw posture information on the frame"""
        h, w, _ = frame.shape
        y_offset = 30
        line_height = 30
        
        # Draw title
        cv2.putText(frame, "Posture Analysis", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        if metrics is None:
            cv2.putText(frame, "No pose detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return frame
        
        # Display metrics and warnings
        warnings = []
        
        if issues.forward_head_posture:
            warnings.append("Forward Head Posture")
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        cv2.putText(frame, f"Head Distance: {metrics.forward_head:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
        
        if issues.slouching:
            warnings.append("Slouching")
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f"Back Angle: {metrics.back_angle:.1f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
        
        if issues.rounded_shoulders:
            warnings.append("Rounded Shoulders")
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f"Shoulder Angle: {metrics.shoulder_angle:.1f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
        
        if issues.leaning:
            warnings.append("Leaning")
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f"Shoulder Diff: {metrics.shoulder_height_diff:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
        
        if issues.hollow_back:
            warnings.append("Hollow Back")
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, f"Hip-Shoulder: {metrics.hip_shoulder_angle:.1f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += line_height
        
        # Draw warnings
        if warnings:
            y_offset += 10
            cv2.putText(frame, "WARNINGS:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += line_height
            for warning in warnings:
                cv2.putText(frame, f"• {warning}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += line_height
        else:
            y_offset += 10
            cv2.putText(frame, "Posture: GOOD", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Run the main application loop"""
        self.cap = cv2.VideoCapture(0)  # 0 = default webcam
        
        print("PoseUp - Real-time Posture Monitoring")
        print("Press 'q' to quit")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            # Analyze posture
            metrics, issues = self.analyzer.analyze_posture(results.pose_landmarks)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Draw posture information
            frame = self.draw_posture_info(frame, metrics, issues)
            
            # Show output
            cv2.imshow("PoseUp - Real Time Posture Monitoring", frame)
            
            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()


if __name__ == "__main__":
    app = PoseUpApp()
    app.run()