import httpx
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import middleware as mw
import time
import os
import zmq
import threading
import numpy as np

import subprocess

CONFIRMATION_FRAMES = 10
LEAVE_AFTER = 15
FRAME_W = 640
FRAME_H = 480
N_KEYPOINTS = 17

# COCO standard keypoint location
KEYPOINTS_LOCATION = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right_Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle"
]

class BehaviourIdentify:

    def __init__(
        self,
        tracking = "Nose",
    ):

        self.speech = mw.Speech()
        self.speakers = mw.Speakers()
        self.behaviours = mw.Behaviours()
        self.server = mw.Server()
        self.node = mw.Node("behaviour_identify")
        self.onboard = mw.Onboard()
        self.pan = mw.Pan()
        self.tilt = mw.Tilt()

        self.tracking_idx = KEYPOINTS_LOCATION.index(tracking)

        self.initial_pan = self.pan.current_angle
        self.initial_tilt = self.tilt.current_angle
        self.filtered_cx = None
        self.filtered_cy = None

        self.state = "SEARCH" # or "FOLLOW"

        self.connect_to_tpu()
        self.latest_pose = None
        self.lock = threading.Lock()
        self.running = True
        self.consecutive_detection = 0
        self.consecutive_empty = 0

        t = threading.Thread(target=self.reader, daemon=True)
        t.start()

    def connect_to_tpu(self, endpoint: str = "tcp://127.0.0.1:5556"):
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.SUB)
        self.socket.connect(endpoint)

        # Subscribe to all topics
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"Connected to server {endpoint}")

    def reader(self):
        """Read the latest pose estimation from the TPU."""
        while self.running:
            data = self.socket.recv()
            # Reconstruct the pose
            if data is not None:
                self.latest_pose = np.frombuffer(data, dtype=np.float32).reshape((1, 17, 3))


    def calculate_centroid(self, conf_threshold = 0.3, max_jump = 0.2):
        """Calculate the centroid of the keypoint cloud higher than a threshold.
        """

        # Extract the keypoints
        keypoint = self.latest_pose[0][self.tracking_idx]

        y, x, conf = keypoint

        if conf < conf_threshold:
            return None
        
        # Reject sudden jumps
        if hasattr(self, "last_centroid") and self.last_centroid is not None:
            dx = abs(x - self.last_centroid[0])
            dy = abs(y - self.last_centroid[1])

            if dx > max_jump or dy > max_jump:
                return self.last_centroid # ignore the outlier

        self.last_centroid = (x, y)

        return (x, y)
    

    def to_pixel_coords(self, centroid):
        cx = int(centroid[0] * FRAME_W)
        cy = int(centroid[1] * FRAME_H)
        return cx, cy


    def compute_error(self, cx, cy, alpha: float = 0.8):
        
        center_x = FRAME_W // 2
        center_y = FRAME_H // 2

        if self.filtered_cx is None:
            self.filtered_cx = cx
            self.filtered_cy = cy
        
        # Exponential moving average
        self.filtered_cx = alpha * cx + (1 - alpha) * self.filtered_cx
        self.filtered_cy = alpha * cy + (1 - alpha) * self.filtered_cy

        error_x = self.filtered_cx - center_x
        error_y = self.filtered_cy - center_y

        print(error_x, error_y)
        # Dead zone
        if abs(error_x) < 0.03 * FRAME_W:
            error_x = 0
        
        if abs(error_y) < 0.03 * FRAME_H:
            error_y = 0

        return error_x, error_y
    

    def compute_pan_tilt(self, error_x, error_y, gain_pan = 80.0, max_step = 10.0):

        pan_adjust = -error_x * gain_pan
        tilt_adjust = error_y * gain_pan

        # Clamp the movement speed
        pan_adjust = np.clip(pan_adjust, -max_step, max_step)
        tilt_adjust = np.clip(tilt_adjust, -max_step, max_step)

        new_pan = float(self.pan.current_angle) + pan_adjust
        new_tilt = float(self.tilt.current_angle) + tilt_adjust

        new_pan = max(self.pan.min_angle, min(self.pan.max_angle, new_pan))
        new_tilt = max(self.tilt.min_angle, min(self.tilt.max_angle, new_tilt))

        return new_pan, new_tilt
    

    def detect_person(self, detection_threshold = 0.3):
        
        # Calculate the mean confidence from the keypoint detection
        if self.latest_pose is None:
            return False

        keypoint = self.latest_pose[0][self.tracking_idx]

        _, _, conf = keypoint

        if conf >= detection_threshold:
            return True
        
        return False


    def follow_me(self):
        if self.latest_pose is None:
            return False

        centroid = self.calculate_centroid()
        
        if centroid is None:
            return False
        
        # Calculate the centroid in pixel coordinates
        cx, cy = self.to_pixel_coords(centroid)

        # Calculate the difference between the frame center and the pose centroid
        error_x, error_y = self.compute_error(cx, cy)

        # Calculate the new position of the robot
        new_pan, new_tilt = self.compute_pan_tilt(error_x, error_y)


        # Update the position of the robot
        self.pan.angle = new_pan
        self.tilt.angle = new_tilt

        return True
        
        
    def run(self):

        # Enable the motors on startup
        self.pan.enable = True
        self.tilt.enable = True

        try:
            while not self.node.is_shutdown():
                time.sleep(0.02)

                detected = self.detect_person()

                if detected:
                    self.consecutive_detection += 1
                    self.consecutive_empty = 0
                else:
                    self.consecutive_empty += 1
                    self.consecutive_detection = 0
                
                # STATE TRANSITIONS

                # SEARCH -> FOLLOW
                if self.state == "SEARCH":
                    if self.consecutive_detection > CONFIRMATION_FRAMES:
                        self.state = "FOLLOW"
                        self.node.loginfo("Person detected. Following...")
                
                elif self.state == "FOLLOW":
                    if self.consecutive_empty > LEAVE_AFTER:
                        self.state = "SEARCH"
                        self.node.loginfo("Person lost. Going back to initial position...")
                        self.pan.angle = self.initial_pan
                        time.sleep(2)
                
                # STATE ACTIONS

                if self.state == "FOLLOW":
                    self.follow_me()

        except KeyboardInterrupt:
            pass
        
        finally:
            self.running = False
            self.node.shutdown()

if __name__ == "__main__":
    node = BehaviourIdentify()
    node.run()

