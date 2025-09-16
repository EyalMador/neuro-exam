import os
import cv2
import mediapipe as mp
from model import Frame, Video, model_types


class MPModel:
    def __init__(self, model_type, complexity=1, d_confidence=0.5, t_confidence=0.5, save=False):
        self.video = Video()
        self.mp_type = model_type
        self.drawing = mp.solutions.drawing_utils
        self.style = mp.solutions.drawing_styles
        self.model = None

        if model_type == 'pose':
            self.model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=d_confidence,
                min_tracking_confidence=t_confidence
            )
        elif model_type == 'hands':
            self.model = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=complexity,
                min_detection_confidence=d_confidence,
                min_tracking_confidence=t_confidence
            )
        else:
            raise ValueError(f"Unsupported mediapipe type: {model_type}")

    def draw_landmarks(self, result, frame):
        if self.mp_type == 'pose' and result.pose_landmarks:
            self.drawing.draw_landmarks(
                frame, result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.style.get_default_pose_landmarks_style()
            )
        elif self.mp_type == 'hands' and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame, hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.style.get_default_hand_landmarks_style()
                )

    def update_coords(self, result, frame_num):
        # ---- Pose landmarks ----
        if self.mp_type == "pose" and result.pose_landmarks:
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                name = mp.solutions.pose.PoseLandmark(idx).name  # e.g. "LEFT_ELBOW"
                if name not in self.video.coords:
                    self.video.coords[name] = {}
                self.video.coords[name][frame_num] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "v": lm.visibility,
                }

        # ---- Hand landmarks ----
        elif self.mp_type == "hands" and result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get handedness: "Left" or "Right"
                label = result.multi_handedness[hand_idx].classification[0].label.upper()
                for idx, lm in enumerate(hand_landmarks.landmark):
                    name = f"{label}_HAND.{mp.solutions.hands.HandLandmark(idx).name}"  # e.g. "LEFT_HAND.WRIST"
                    if name not in self.video.coords:
                        self.video.coords[name] = {}
                    self.video.coords[name][frame_num] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    }

    

    
    def processing(self, cap, save_video_output=True, output_video_dir=".", output_video_name = "output.mp4"):
        self.video.frame_data = Frame(cap)
        if save_video_output:
            output_path = os.path.join(output_video_dir, output_video_name)
            self.video.output_channel = self.video.out_channel(
                output_path, self.video.frame_data.fps,
                self.video.frame_data.width, self.video.frame_data.height
            )

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.model.process(rgb)

            self.draw_landmarks(result, frame)
            self.update_coords(result, frame_num)

            if save_video_output:
                self.video.output_channel.write(frame)

            frame_num += 1

        cap.release()
        if save_video_output:
            self.video.output_channel.release()
        cv2.destroyAllWindows()
        self.model.close()

        return self.video.coords
