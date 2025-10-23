import os
import cv2
import numpy as np
from rtmlib import BodyWithFeet, draw_skeleton
from .model import Frame, Video


class RTMModel:
    """Wrapper around RTMLib Body_with_feet (26 keypoints) with the same API as MPModel."""

    def __init__(self, model_type="body26", device="cpu", backend="onnxruntime", save=False):
        self.video = Video()
        self.model_type = model_type
        self.device = device
        self.backend = backend
        self.save = save

        if model_type != "body26":
            raise ValueError(f"Unsupported RTM model type: {model_type}")

        self.model = BodyWithFeet(
            backend=backend,
            device=device,
            mode="performance"  # options: 'performance', 'lightweight', 'balanced'
        )

    def draw_landmarks(self, frame, keypoints, scores):
        """Draw RTMLib skeleton on the frame."""
        if keypoints is not None and scores is not None:
            draw_skeleton(frame, keypoints, scores, kpt_thr=0.5)
        return frame

    def _update_body(self, results, frame_num):
        """Update self.video.coords with RTMLib results."""
        if results is None:
            return

        # Case 1: single person array (26, 3)
        if isinstance(results, np.ndarray) and results.ndim == 2:
            people = [results]

        # Case 2: multi-person array (N, 26, 3)
        elif isinstance(results, np.ndarray) and results.ndim == 3:
            people = results

        # Case 3: list of dicts (old API)
        elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            people = [p["keypoints"] for p in results]

        else:
            return  # unknown format

        for person_kpts in people:
            for idx, (x, y, score) in enumerate(person_kpts):
                name = f"BODY_{idx}"
                if name not in self.video.coords:
                    self.video.coords[name] = {}
                self.video.coords[name][frame_num] = {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,     
                    "v": float(score),
                }


    def update_coords(self, keypoints, scores, frame_num):
        """Store coordinates (x, y, z=0, v=score)."""
        if keypoints is None or scores is None:
            return

        # Handle multiple people or single person
        if isinstance(keypoints, np.ndarray) and keypoints.ndim == 3:
            people = keypoints
            people_scores = scores
        else:
            people = [keypoints]
            people_scores = [scores]

        for person_kpts, person_scores in zip(people, people_scores):
            for idx, (x, y) in enumerate(person_kpts[:, :2]):
                v = person_scores[idx] if idx < len(person_scores) else 0.0
                name = f"BODY_{idx}"
                if name not in self.video.coords:
                    self.video.coords[name] = {}
                self.video.coords[name][frame_num] = {
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,     # 2D model placeholder
                    "v": float(v),
                }

    def processing(self, cap, save_video_output=True, output_video_dir=".", output_video_name="rtm_output.mp4"):
        """Run inference and save annotated video."""
        self.video.frame_data = Frame(cap)

        if save_video_output:
            output_path = os.path.join(output_video_dir, output_video_name)
            self.video.output_channel = self.video.out_channel(
                output_path,
                self.video.frame_data.fps,
                self.video.frame_data.width,
                self.video.frame_data.height,
            )

        frame_num = 0
        print(f"Starting RTMLib processing...")

        from tqdm.notebook import tqdm
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            keypoints, scores = self.model(frame)

            # visualize
            frame = self.draw_landmarks(frame, keypoints, scores)
            # save coords
            self.update_coords(keypoints, scores, frame_num)

            if save_video_output:
                self.video.output_channel.write(frame)

            frame_num += 1
            pbar.update(1)


        cap.release()
        if save_video_output:
            self.video.output_channel.release()
        cv2.destroyAllWindows()

        return self.video.coords
