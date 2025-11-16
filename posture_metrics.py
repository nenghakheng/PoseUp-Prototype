"""Data classes for posture metrics and issues"""
from dataclasses import dataclass


@dataclass
class PostureMetrics:
    """Stores calculated posture metrics"""
    forward_head: float  # Distance in pixels (normalized)
    head_distance: float  # Distance from camera (z-depth, normalized)
    back_angle: float  # Degrees
    shoulder_height_diff: float  # Normalized difference
    hip_shoulder_angle: float  # Degrees
    

@dataclass
class PostureIssues:
    """Flags for detected posture problems"""
    forward_head_posture: bool = False
    too_close_to_monitor: bool = False
    slouching: bool = False
    leaning: bool = False
    hollow_back: bool = False
