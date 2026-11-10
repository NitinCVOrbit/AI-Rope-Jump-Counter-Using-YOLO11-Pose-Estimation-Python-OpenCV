import numpy as np
from collections import deque


class JumpCounter:

    def __init__(
        self,
        threshold=15,
        smoothing_frames=5
    ):

        self.threshold = threshold

        self.history = deque(
            maxlen=smoothing_frames
        )

        self.baseline = None

        self.jump_count = 0

        # States:
        # GROUND
        # RISING
        # FALLING

        self.state = "GROUND"

    # --------------------------------------------------------
    # Calculate body center
    # --------------------------------------------------------

    def get_body_position(self, keypoints):

        # COCO keypoints
        #
        # 11 = left hip
        # 12 = right hip

        left_hip = keypoints[11]
        right_hip = keypoints[12]

        # to make sure both hips are detected.
        if ( left_hip[0] <= 0 or right_hip[0] <= 0 ):
            return None
        
        ## to get the center position of the person's body.
        hip_y = ( left_hip[1] + right_hip[1] ) / 2

        return hip_y

    # --------------------------------------------------------
    # Update jump counter
    # --------------------------------------------------------

    def update(self, keypoints):

        y = self.get_body_position(keypoints)

        if y is None:
            return self.jump_count, self.state

        self.history.append(y)

        # Need enough frames
        if len(self.history) < 3:
            return self.jump_count, self.state

        current_y = np.mean(self.history)

        # Initialize baseline
        if self.baseline is None:

            self.baseline = current_y

            return self.jump_count, self.state
        
        # Y-coordinates (0,0) start from the top-left corner of the image.    
        # When the person moves upward, the current Y decreases so displacement increases.
        # When the person moves downward, the current Y increases, so displacement decreases.
        displacement = self.baseline - current_y

        # print(f'Dis: {displacement}  Baseline: {self.baseline}  Current_y: {current_y}')

        # ----------------------------------------------------
        # PERSON MOVES UP
        # ----------------------------------------------------

        if displacement > self.threshold:

            if self.state == "GROUND":

                self.state = "RISING"

            elif self.state == "FALLING":

                self.state = "RISING"

        # ----------------------------------------------------
        # PERSON RETURNS DOWN
        # ----------------------------------------------------

        elif displacement < self.threshold * 0.3:

            if self.state == "RISING":

                self.state = "FALLING"

            elif self.state == "FALLING":

                self.jump_count += 1

                self.state = "GROUND"

        # Slowly update baseline 
        self.baseline = self.baseline * 0.95 + current_y * 0.05  # Exponential Moving Average (EMA)

        return self.jump_count, self.state



# ==============================================================================
# TABLE: TWO JITTER EVENTS CAUSING DOUBLE COUNTING (Threshold Multiplier = 1.0)
# ==============================================================================
#
# Frame / Stage          | Threshold | Displacement | State Transition       | Count
# -----------------------|-----------|--------------|------------------------|-------
# 1. Takeoff             | 15px      | +18px        | GROUND -> RISING       | 0
# 2. Peak                | 15px      | +28px        | RISING                 | 0
# 3. First Jitter Drop   | 15px      | +14px        | RISING -> FALLING      | 0
# 4. First False Landing | 15px      | +12px        | FALLING -> GROUND      | 1
# 5. Mid-Air Bounce      | 15px      | +16px        | GROUND -> RISING       | 1
# 6. Second Jitter       | 15px      | +13px        | RISING -> FALLING      | 1
# 7. Second False Landing| 15px      | +11px        | FALLING -> GROUND      | 2
#
# ==============================================================================

# ==============================================================================
# TABLE: JUMP COUNTING WITH HYSTERESIS (Threshold Multiplier = 0.3)
# ==============================================================================
#
# Frame / Stage          | Threshold | Displacement | State Transition       | Count
# -----------------------|-----------|--------------|------------------------|-------
# 1. Takeoff             | 15px      | +18px        | GROUND -> RISING       | 0
# 2. Peak                | 4.5px     | +28px        | RISING                 | 0
# 3. First Jitter Drop   | 4.5px     | +14px        | RISING                 | 0
# 4. Mid-Air Bounce      | 4.5px     | +16px        | RISING                 | 0
# 5. Second Jitter       | 4.5px     | +13px        | RISING                 | 0
# 6. Early Descent       | 4.5px     | +11px        | RISING                 | 0
# 7. Near Landing         | 4.5px     | +4px         | RISING -> FALLING      | 0
# 8. Full Landing         | 4.5px     | +2px         | FALLING -> GROUND      | 1
#
# ==============================================================================



# 1. BEFORE FATIGUE (Standing Tall):
#    - Starting Baseline = 300
#    - Normal Standing Y = 300
#    - Jump Y = 250
#    - Displacement: 300 - 250 = 50 (Since 50 > 15 threshold -> Count Works!)
#
# 2. JUMPING WHILE FATIGUED (Jump Y = 290):
#    - Baseline = 300
#    - Jump Peak Y = 290
#    - With a Static Baseline (300): 300 - 290 = 10 (10 < 15 -> JUMP MISSED ❌)



# ============================================================
# FATIGUE SETS IN (BASELINE ADAPTATION AFTER 10 JUMPS)
# ============================================================

# The system waits for the first 3 frames before initializing
# the baseline.

# Frame 1: Hip Y = 300 -> Not enough frames to initialize baseline.
# Frame 2: Hip Y = 300 -> Not enough frames to initialize baseline.
# Frame 3: Hip Y = 300 -> Baseline is initialized to 300.

# Initial Baseline = 300 pixels.

# After performing several jumps, the person gradually gets tired.
# Due to fatigue, the person's resting posture changes.
# Their average resting hip Y-coordinate moves from 300 to 320.

# The baseline continuously adapts using the EMA formula:

# New Baseline = (Old Baseline * 0.95) + (Current Y * 0.05)

# Assuming one resting-position update per jump:

# Frame 4: (300.00 * 0.95) + (320 * 0.05) = 301.00
# Frame 5: (301.00 * 0.95) + (320 * 0.05) = 301.95
# Frame 6: (301.95 * 0.95) + (320 * 0.05) = 302.85
# Frame 7: (302.85 * 0.95) + (320 * 0.05) = 303.71
# Frame 8: (303.71 * 0.95) + (320 * 0.05) = 304.52
# Frame 9: (304.52 * 0.95) + (320 * 0.05) = 305.30
# Frame 10: (305.30 * 0.95) + (320 * 0.05) = 306.03
# Frame 11: (306.03 * 0.95) + (320 * 0.05) = 306.73
# Frame 12: (306.73 * 0.95) + (320 * 0.05) = 307.40
# Frame 13: (307.40 * 0.95) + (320 * 0.05) = 308.03

# However, after 10 jumps, fatigue reduces their jumping ability.

# Now, the person's peak hip Y-coordinate is only 290 pixels.

# Calculate displacement using the updated baseline:

# Displacement = Baseline - Current Y
# Displacement = 308.03 - 290
# Displacement = 18.03 pixels

# Rising Threshold = 15 pixels

# Since displacement (18.03) > threshold (15):
# GROUND -> RISING
