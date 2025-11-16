"""Main application class for PoseUp"""
import cv2
import mediapipe as mp
import numpy as np
from posture_analyzer import PostureAnalyzer
from ui_renderer import UIRenderer


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
        self.ui_renderer = UIRenderer()
        self.cap = None
        self.running = True
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_width = param
            # Since frame is flipped, flip x coordinate back
            flipped_x = frame_width - x
            if self.ui_renderer.check_button_click(flipped_x, y, frame_width):
                self.running = False
        
    def run(self):
        """Run the main application loop"""
        self.cap = cv2.VideoCapture(0)  # 0 = default webcam
        
        if not self.cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Get frame dimensions for button positioning
        ret, test_frame = self.cap.read()
        if ret:
            h, w, _ = test_frame.shape
        else:
            w = 640  # Default width
        
        # Create window and set mouse callback
        window_name = "PoseUp - Real Time Posture Monitoring"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback, w)
        
        print("PoseUp - Real-time Posture Monitoring")
        print("Press 'Q' to quit or click the CLOSE button")
        
        while self.cap.isOpened() and self.running:
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
            frame = self.ui_renderer.draw_posture_info(frame, metrics, issues)
            
            # Show output
            cv2.imshow(window_name, frame)
            
            # Press "q" to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
