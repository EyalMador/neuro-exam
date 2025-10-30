import os
import cv2
import mediapipe as mp
from .model import Frame, Video


class MPModel:
    def __init__(self, model_type, complexity=2, d_confidence=0.8, t_confidence=0.8, use_world_landmarks=False, save=False):
        self.video = Video()
        self.mp_type = model_type
        self.drawing = mp.solutions.drawing_utils
        self.style = mp.solutions.drawing_styles
        self.model = None
        self.use_world_landmarks = use_world_landmarks

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
        elif model_type == 'holistic':
            self.model = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=False,  # disable fine face mesh, we won’t use face
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
                
        elif self.mp_type == 'holistic':
            if result.pose_landmarks:
                self.drawing.draw_landmarks(
                    frame, result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.style.get_default_pose_landmarks_style()
                )
            if result.left_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame, result.left_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.style.get_default_hand_landmarks_style()
                )
            if result.right_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame, result.right_hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.style.get_default_hand_landmarks_style()
                )
    
    

    
    def _update_pose(self, pose_landmarks, frame_num):
        for idx, lm in enumerate(pose_landmarks.landmark):
            name = mp.solutions.pose.PoseLandmark(idx).name
            if name not in self.video.coords:
                self.video.coords[name] = {}
            self.video.coords[name][frame_num] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "v": lm.visibility,
            }
            
    def _update_hands(self, multi_hand_landmarks, multi_handedness, frame_num):
        for hand_idx, hand_landmarks in enumerate(multi_hand_landmarks):
            label = multi_handedness[hand_idx].classification[0].label.upper()  # "LEFT" or "RIGHT"
            for idx, lm in enumerate(hand_landmarks.landmark):
                name = f"{label}_HAND.{mp.solutions.hands.HandLandmark(idx).name}"
                if name not in self.video.coords:
                    self.video.coords[name] = {}
                self.video.coords[name][frame_num] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                }
                
    def _update_single_hand(self, hand_landmarks, label, frame_num):
        for idx, lm in enumerate(hand_landmarks.landmark):
            name = f"{label}.{mp.solutions.hands.HandLandmark(idx).name}"
            if name not in self.video.coords:
                self.video.coords[name] = {}
            self.video.coords[name][frame_num] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
            }
    
    def _update_coords(self, result, frame_num):
        """Update self.video.coords with landmarks depending on model type."""

        # Pose model only
        if self.mp_type == "pose" and result.pose_landmarks:
            self._update_pose(result.pose_landmarks, frame_num)

        # Hands model only
        if self.mp_type == "hands" and result.multi_hand_landmarks:
            self._update_hands(result.multi_hand_landmarks, result.multi_handedness, frame_num)

        # Holistic model (pose + hands, ignore face)
        if self.mp_type == "holistic":
            if result.pose_landmarks:
                self._update_pose(result.pose_landmarks, frame_num)
            if result.left_hand_landmarks:
                self._update_single_hand(result.left_hand_landmarks, "LEFT_HAND", frame_num)
            if result.right_hand_landmarks:
                self._update_single_hand(result.right_hand_landmarks, "RIGHT_HAND", frame_num)
    
    
    def update_coords(self, result, frame_num):
        """Update self.video.coords with landmarks depending on model type."""
        
        # ---- Pose model only ----
        if self.mp_type == "pose":
            # Use world landmarks if flag is set and available
            if getattr(self, "use_world_landmarks", False) and result.pose_world_landmarks:
                pose_landmarks = result.pose_world_landmarks
            else:
                pose_landmarks = result.pose_landmarks

            if pose_landmarks:
                self._update_pose(pose_landmarks, frame_num)

        # ---- Hands model only ----
        elif self.mp_type == "hands" and result.multi_hand_landmarks:
            self._update_hands(result.multi_hand_landmarks, result.multi_handedness, frame_num)

        # ---- Holistic model ----
        elif self.mp_type == "holistic":
            # Same logic for pose part
            if getattr(self, "use_world_landmarks", False) and result.pose_world_landmarks:
                pose_landmarks = result.pose_world_landmarks
            else:
                pose_landmarks = result.pose_landmarks

            if pose_landmarks:
                self._update_pose(pose_landmarks, frame_num)
            if result.left_hand_landmarks:
                self._update_single_hand(result.left_hand_landmarks, "LEFT_HAND", frame_num)
            if result.right_hand_landmarks:
                self._update_single_hand(result.right_hand_landmarks, "RIGHT_HAND", frame_num)

                
    """
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

        """

    
    def processing(self, cap, save_video_output=True, output_video_dir=".", output_video_name = "output.mp4"):
        self.video.frame_data = Frame(cap)
        if save_video_output:
            output_path = os.path.join(output_video_dir, output_video_name)
            self.video.output_channel = self.video.out_channel(
                output_path, self.video.frame_data.fps,
                self.video.frame_data.width, self.video.frame_data.height
            )

        frame_num = 0
        rotation = getattr(cap, "rotation", 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- Rotate frame if needed ---
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            
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
