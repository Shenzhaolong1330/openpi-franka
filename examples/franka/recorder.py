# recorder.py
import threading
import queue
import yaml
import time
import cv2
import av
import numpy as np
from pathlib import Path

class Recorder:
    """This class handles asynchronous logging and visualization."""

    def __init__(self, log_path: Path, video_path: list, display_fps: int = 15, visualize: bool = False):
        self.visualize = visualize
        # log path    
        self.log_path = log_path

        # video paths
        # self.left_wrist_video = video_path[0]
        self.wrist_video = video_path[0]
        self.exterior_video = video_path[1]

        self.display_fps = display_fps
        
        # safe queues
        self.queue_log = queue.Queue(maxsize=1)
        self.queue_vis = queue.Queue(maxsize=1)

        # store frames for video
        self.frames_ext = []    # exterior camera frames
        # self.frames_left_wrist = []  # left wrist camera frames
        # self.frames_right_wrist = []  # right wrist camera frames
        self.frames_wrist = []  # wrist camera frames

        # start threads
        threading.Thread(target=self._logger_thread, daemon=True).start()
        threading.Thread(target=self._visualizer_thread, daemon=True).start()

    # ====================== Logger Thread ====================== #
    def _logger_thread(self):
        while True:
            data = self.queue_log.get()  # blocking
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)
                    f.write(' ' + '-' * 60 + '\n')
                    f.flush()
            except Exception as e:
                print(f"[Recorder Logger ERROR] {e}")

    # ====================== Visualizer Thread ====================== #
    def _visualizer_thread(self):
        last_frame_time = 0
        while True:
            try:
                obs = self.queue_vis.get()
                # l_wrist, ext, r_wrist = map(self.to_bgr, [obs["observation/left_wrist_image"], obs["observation/exterior_image"], obs["observation/right_wrist_image"]])
                ext, wrist = map(self.to_bgr, [obs["observation/image"], obs["observation/wrist_image"]])
                # save frames in memory (convert to uint8 to save space)
                self.frames_ext.append(ext.astype(np.uint8))
                # self.frames_left_wrist.append(l_wrist.astype(np.uint8))
                # self.frames_right_wrist.append(r_wrist.astype(np.uint8))
                self.frames_wrist.append(wrist.astype(np.uint8))

                if self.visualize:
                    # concatenate and display
                    # combined = np.hstack((l_wrist, ext, r_wrist))
                    # cv2.imshow("Left Wrist | Exterior | Right Wrist", combined)
                    combined = np.hstack((ext, wrist))
                    cv2.imshow("Exterior | Wrist", combined)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # control frame rate
                elapsed = time.time() - last_frame_time
                if elapsed < 1 / self.display_fps:
                    time.sleep(1 / self.display_fps - elapsed)
                last_frame_time = time.time()
            except Exception as e:
                print(f"[VIS ERROR] {e}")
                time.sleep(0.1)
    
    def _encode(self, frames, out_path: Path, vcodec: str = "libx264", crf: int = 23, preset="medium"):
        h, w, _ = frames[0].shape
        container = av.open(str(out_path), "w")
        stream = container.add_stream(vcodec, rate=self.display_fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": str(preset)}

        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        # flush
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"{out_path.name}: {size_mb:.2f} MB")
        return size_mb
        
    # ====================== Save Video ====================== #
    def save_video(self):
        """Save stored frames as two separate MP4 videos using H.264."""
        # if not self.frames_ext or not self.frames_left_wrist or not self.frames_right_wrist:
        if not self.frames_ext or not self.frames_wrist:
            print("No frames to save.")
            return

        print("\nSaving exterior camera video...")
        self._encode(self.frames_ext, self.exterior_video, vcodec="libx264", crf=23, preset="veryslow")

        # print("\nSaving left wrist camera video...")
        # self._encode(self.frames_left_wrist, self.left_wrist_video, vcodec="libx264", crf=23, preset="veryslow")
        print("\nSaving wrist camera video...")
        self._encode(self.frames_wrist, self.wrist_video, vcodec="libx264", crf=23, preset="veryslow")

    # ====================== Save Video ====================== #
    def save_videos_multi_codec(self):
        """Save frames in memory as videos with H.264 / H.265 / AV1."""

        # if not self.frames_ext or not self.frames_left_wrist or not self.frames_right_wrist:
        if not self.frames_ext or not self.frames_wrist:
            print("No frames to save.")
            return

        # ---------------- Helper ---------------- #
        def _encode(frames, out_path: Path, vcodec: str, crf: int = 30, preset="medium"):
            h, w, _ = frames[0].shape
            container = av.open(str(out_path), "w")
            stream = container.add_stream(vcodec, rate=self.display_fps)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"

            # Set options
            if vcodec in ["libx264", "libx265", "libsvtav1"]:
                stream.options = {"crf": str(crf), "preset": str(preset)}

            for frame in frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # flush
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"🎞️ {out_path.name}: {size_mb:.2f} MB")
            return size_mb

        codecs = [
            ("libx264", 18, "veryslow"),
            ("libx265", 28, "slow"),
            ("libsvtav1", 35, "8"),  # preset 8 = slower
        ]

        print("\n🔧 Saving exterior camera videos...")
        for vcodec, crf, preset in codecs:
            out_path = self.exterior_video.parent / f"ext_{vcodec}.mp4"
            _encode(self.frames_ext, out_path, vcodec=vcodec, crf=crf, preset=preset)

        # print("\n🔧 Saving left wrist camera videos...")
        # for vcodec, crf, preset in codecs:
        #     out_path = self.left_wrist_video.parent / f"left_wrist_{vcodec}.mp4"
        #     _encode(self.frames_left_wrist, out_path, vcodec=vcodec, crf=crf, preset=preset)
        print("\n🔧 Saving wrist camera videos...")
        for vcodec, crf, preset in codecs:
            out_path = self.wrist_video.parent / f"wrist_{vcodec}.mp4"
            _encode(self.frames_wrist, out_path, vcodec=vcodec, crf=crf, preset=preset)

    # ====================== Utility Functions ====================== #
    def to_bgr(self, img):
        """Convert RGB → BGR safely."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    # ====================== External Interface ====================== #
    def submit_actions(self, actions, infer_time: int, prompt: str = ""):
        """
        Main thread calls this method to submit action results (required).
        - actions: array-like (N x M) numpy or list
        - infer_time: int (the N-th inference / index)
        - prompt: optional string
        This function tries not to block the main loop (uses put_nowait)
        """
        try:
            actions_list = np.round(np.asarray(actions), 3).tolist()
        except Exception:
            try:
                actions_list = list(actions)
            except Exception:
                actions_list = actions  
        
        delta_list = []
        for i, row in enumerate(actions):
            if i == 0:
                delta = row.copy()  # 第一行，差值就是自己
            else:
                delta = row - actions[i-1]
            delta_list.append(np.round(delta[:6], 4).tolist())

        data = {
            "infer_time": int(infer_time),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": prompt,
            "actions": actions_list,
            "actions_delta": delta_list,
        }

        try:
            self.queue_log.put_nowait(data)
        except queue.Full:
            # If the queue is full, discard this record (do not block the main loop)
            pass

    def submit_obs(self, obs: dict):
        """
        obs dictionary should contain at least:
        - observation/image: exterior camera image (numpy array)
        - observation/wrist_image: wrist camera image (numpy array)
        """
        try:
            self.queue_vis.put_nowait(obs)
        except queue.Full:
            pass