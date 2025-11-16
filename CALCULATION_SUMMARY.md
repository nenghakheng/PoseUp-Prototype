# PoseUp Posture Calculation Summary

This document explains how each posture metric is calculated using MediaPipe pose landmarks.

## Overview

The application uses **MediaPipe Pose** to detect 33 body landmarks in real-time from webcam video. Each landmark has normalized coordinates:
- **x, y**: 2D position (0.0 to 1.0, relative to frame dimensions)
- **z**: Depth from camera (negative values = closer to camera)

## Posture Metrics Calculated

### 1. Forward Head Posture
**What it detects:** Head positioned too far forward relative to shoulders

**Calculation:**
```python
ear_x = average of left_ear.x and right_ear.x
shoulder_x = average of left_shoulder.x and right_shoulder.x
forward_head_distance = abs(ear_x - shoulder_x)
```

**Detection Rule:**
- Alert if `forward_head_distance > 0.05` (normalized units)
- Measures horizontal (x-axis) distance between ear and shoulder midpoints
- Larger value = head is more forward

**Why it matters:** Forward head posture strains neck muscles and can cause pain

---

### 2. Too Close to Monitor
**What it detects:** User leaning too close to the screen

**Calculation:**
```python
ear_z = average of left_ear.z and right_ear.z
head_distance = ear_z  # More negative = closer to camera
```

**Detection Rule:**
- Alert if `head_distance < -0.3` (normalized depth)
- Uses z-coordinate (depth) from MediaPipe
- More negative values indicate closer proximity to camera

**Why it matters:** Sitting too close to monitor causes eye strain and promotes poor posture

---

### 3. Slouching (Back Angle)
**What it detects:** Rounded or hunched back while sitting

**Calculation:**
```python
# Using 3 points: shoulder, hip, knee
shoulder = midpoint of left_shoulder and right_shoulder
hip = midpoint of left_hip and right_hip
knee = midpoint of left_knee and right_knee

# Calculate angle at hip vertex
back_angle = angle_between(shoulder, hip, knee)
```

**Angle Calculation Method:**
```python
# Vector from hip to shoulder
vector_a = shoulder - hip
# Vector from hip to knee  
vector_b = knee - hip

# Dot product formula
cos_angle = dot(vector_a, vector_b) / (||vector_a|| * ||vector_b||)
angle_degrees = arccos(cos_angle) * 180/π
```

**Detection Rule:**
- Alert if `back_angle < 160°`
- Ideal posture: ~180° (straight line from shoulder through hip to knee)
- Lower angles indicate slouching/hunching

**Why it matters:** Slouching compresses spine and reduces lung capacity

---

### 4. Rounded Shoulders
**What it detects:** Shoulders rolling forward/inward

**Calculation:**
```python
# For each shoulder:
# Create vertical reference point below shoulder
vertical_point = (shoulder.x, shoulder.y + 0.1)

# Calculate angle: vertical_reference -> shoulder -> elbow
angle_from_vertical = angle_between(vertical_point, shoulder, elbow)

# Measure deviation from perpendicular (90°)
rounded_shoulder_deviation = abs(90 - angle_from_vertical)

# Average both shoulders
shoulder_angle = (left_deviation + right_deviation) / 2
```

**Detection Rule:**
- Alert if `shoulder_angle > 30°` (deviation from vertical)
- Ideal: shoulders aligned vertically above hips
- Higher values = more rounded/hunched shoulders

**Why it matters:** Rounded shoulders cause upper back pain and restrict breathing

---

### 5. Leaning Left/Right
**What it detects:** Uneven shoulder height (body tilting sideways)

**Calculation:**
```python
left_shoulder_y = left_shoulder.y
right_shoulder_y = right_shoulder.y

shoulder_height_difference = abs(left_shoulder_y - right_shoulder_y)
```

**Detection Rule:**
- Alert if `shoulder_height_difference > 0.03` (normalized units)
- Measures vertical (y-axis) difference between shoulders
- Shoulders should be level when sitting properly

**Why it matters:** Leaning causes asymmetric muscle strain and spinal curvature

---

### 6. Hollow Back (Lordosis)
**What it detects:** Excessive arching of lower back

**Calculation:**
```python
# Using 3 points: shoulder, hip, ankle
shoulder = midpoint of left_shoulder and right_shoulder
hip = midpoint of left_hip and right_hip
ankle = midpoint of left_ankle and right_ankle

# Calculate angle at hip vertex
hip_shoulder_angle = angle_between(shoulder, hip, ankle)
```

**Detection Rule:**
- Alert if `hip_shoulder_angle < 165°` OR `hip_shoulder_angle > 195°`
- Normal range: 165° - 195°
- Too low = excessive forward tilt
- Too high = excessive backward tilt

**Why it matters:** Abnormal hip-shoulder alignment indicates pelvic tilt and lower back issues

---

## Technical Implementation

### Geometric Angle Calculation
All angles use the **dot product formula**:

```python
def calculate_angle(point1, point2, point3):
    """Calculate angle at point2 (vertex)"""
    # Convert to numpy arrays
    a = np.array(point1)
    b = np.array(point2)  # vertex
    c = np.array(point3)
    
    # Create vectors from vertex
    ba = a - b
    bc = c - b
    
    # Dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Convert to degrees
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_radians)
```

### Coordinate Normalization
- MediaPipe provides **normalized coordinates** (0.0 to 1.0)
- All distance calculations use these normalized values
- No conversion to pixels needed for thresholds
- Works across different camera resolutions

### Bilateral Averaging
For symmetrical body parts:
```python
# Average left and right sides for stability
ear = ((left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2)
shoulder = ((left_shoulder.x + right_shoulder.x) / 2, 
           (left_shoulder.y + right_shoulder.y) / 2)
```

Benefits:
- Reduces noise from single-side detection errors
- Provides more stable measurements
- Accounts for slight body asymmetries

---

## Threshold Values Summary

| Metric | Threshold | Unit | Alert Condition |
|--------|-----------|------|-----------------|
| Forward Head | 0.05 | normalized | > threshold |
| Head Distance | -0.3 | z-depth | < threshold (more negative) |
| Back Angle | 160° | degrees | < threshold |
| Shoulder Rounding | 30° | degrees | > threshold |
| Shoulder Balance | 0.03 | normalized | > threshold |
| Hip-Shoulder Angle | 165° - 195° | degrees | outside range |

---

## Data Flow

1. **Capture Frame** → Webcam video (640x480 or higher)
2. **MediaPipe Processing** → Detect 33 pose landmarks with x, y, z coordinates
3. **Landmark Extraction** → Get specific points (ears, shoulders, hips, etc.)
4. **Bilateral Averaging** → Average left/right landmarks for stability
5. **Metric Calculation** → Apply geometric formulas (angles, distances)
6. **Threshold Comparison** → Check if metrics exceed alert thresholds
7. **UI Rendering** → Display metrics with color-coded status indicators
8. **Real-time Feedback** → Show warnings for detected posture issues

---

## Accuracy Considerations

### Strengths
- Uses proven geometric calculations (dot product for angles)
- Normalized coordinates work across resolutions
- Bilateral averaging reduces noise
- Real-time processing (30+ FPS)

### Limitations
- Requires clear view of full upper body
- Lighting affects landmark detection quality
- 2D camera has limited depth perception
- Thresholds are general (not personalized)

### Best Practices for Accurate Detection
1. Sit 2-3 feet from camera
2. Ensure good lighting (front/side)
3. Wear contrasting clothing
4. Keep upper body in frame
5. Avoid busy backgrounds
6. Calibrate thresholds for personal baseline if needed

---

## Future Enhancements

Potential improvements:
- **Personalized baselines**: Calibrate thresholds per user
- **Temporal smoothing**: Average metrics over time to reduce jitter
- **Sitting duration tracking**: Alert after prolonged poor posture
- **Posture score**: Combined metric (0-100) for overall posture quality
- **Historical logging**: Track posture trends over days/weeks
- **Audio alerts**: Voice warnings for critical issues
- **Multiple profiles**: Different thresholds for different activities (gaming, work, etc.)
