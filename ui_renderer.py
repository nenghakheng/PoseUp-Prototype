"""UI rendering utilities for posture feedback"""
import cv2
import numpy as np
from typing import Optional
from posture_metrics import PostureMetrics, PostureIssues


class UIRenderer:
    """Handles drawing posture information on video frames"""
    
    @staticmethod
    def draw_background_panel(frame: np.ndarray, x: int, y: int, 
                             width: int, height: int, alpha: float = 0.7):
        """Draw semi-transparent background panel"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
    
    @staticmethod
    def draw_button(frame: np.ndarray, x: int, y: int, 
                   width: int, height: int, text: str, 
                   bg_color=(200, 50, 50), text_color=(255, 255, 255)):
        """Draw a clickable button"""
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
        
        # Center text in button
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 
                   font_scale, text_color, thickness)
        return frame
    
    @staticmethod
    def draw_posture_info(frame: np.ndarray, 
                         metrics: Optional[PostureMetrics], 
                         issues: PostureIssues) -> np.ndarray:
        """Draw posture information on the frame"""
        h, w, _ = frame.shape
        
        # Draw semi-transparent background panel for better visibility
        panel_width = 400
        panel_height = 365
        frame = UIRenderer.draw_background_panel(frame, 0, 0, panel_width, panel_height, 0.75)
        
        y_offset = 40
        line_height = 35
        x_margin = 15
        
        # Draw title with larger font
        cv2.putText(frame, "POSTURE ANALYSIS", (x_margin, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        y_offset += line_height + 5
        
        if metrics is None:
            cv2.putText(frame, "No pose detected", (x_margin, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            return frame
        
        # Display metrics and warnings
        warnings = []
        font_scale = 0.65
        thickness = 2
        
        # Status indicators with colored circles
        circle_radius = 8
        circle_x = x_margin + 10
        text_x = x_margin + 30
        
        if issues.forward_head_posture:
            warnings.append("Forward Head Posture")
            color = (0, 0, 255)  # Red
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        else:
            color = (0, 255, 0)  # Green
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        cv2.putText(frame, f"Head: {metrics.forward_head:.3f}", 
                   (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        if issues.too_close_to_monitor:
            warnings.append("Too Close to Monitor")
            color = (0, 0, 255)  # Red
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        else:
            color = (0, 255, 0)  # Green
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        cv2.putText(frame, f"Distance: {metrics.head_distance:.3f}", 
                   (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        if issues.slouching:
            warnings.append("Slouching")
            color = (0, 0, 255)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        else:
            color = (0, 255, 0)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        cv2.putText(frame, f"Back: {metrics.back_angle:.1f}°", 
                   (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        if issues.leaning:
            warnings.append("Leaning")
            color = (0, 0, 255)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        else:
            color = (0, 255, 0)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        cv2.putText(frame, f"Balance: {metrics.shoulder_height_diff:.3f}", 
                   (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        if issues.hollow_back:
            warnings.append("Hollow Back")
            color = (0, 0, 255)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        else:
            color = (0, 255, 0)
            cv2.circle(frame, (circle_x, y_offset - 5), circle_radius, color, -1)
        cv2.putText(frame, f"Alignment: {metrics.hip_shoulder_angle:.1f}°", 
                   (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        y_offset += line_height
        
        # Draw warnings section
        y_offset += 15
        cv2.line(frame, (x_margin, y_offset - 10), (panel_width - x_margin, y_offset - 10), 
                (255, 255, 255), 1)
        
        if warnings:
            cv2.putText(frame, "ALERTS:", (x_margin, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 3)
            y_offset += line_height
            for warning in warnings:
                cv2.putText(frame, f"! {warning}", (x_margin + 5, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                y_offset += line_height - 5
        else:
            cv2.putText(frame, "STATUS: EXCELLENT", (x_margin, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 3)
        
        # Draw close button in top-right corner
        button_width = 120
        button_height = 50
        button_x = w - button_width - 20
        button_y = 20
        frame = UIRenderer.draw_button(frame, button_x, button_y, 
                                      button_width, button_height, 
                                      "CLOSE", (200, 50, 50), (255, 255, 255))
        
        # Draw instructions at bottom
        instruction_y = h - 30
        cv2.putText(frame, "Press 'Q' to quit or click CLOSE button", 
                   (x_margin, instruction_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
