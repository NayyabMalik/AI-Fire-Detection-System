import cv2
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = r"C:\Users\PMLS\Documents\ai fire detection frontend\fire_detection_model_image.h5"

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Global progress data (shared with app.py)
from app import progress_data

def extract_fire_clips(
    video_path,
    output_dir,
    progress_id=None,
    frame_skip=5,
    min_clip_length=5,   # in seconds
    frame_threshold=0.3, # threshold to consider fire
    vote_window=10,      # number of frames in voting window
    vote_majority=3      # minimum fire votes in window to consider fire
):
    """
    Extract fire clips from a video using dynamic frame skipping based on model predictions.
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError("Invalid or corrupted video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Processing video: {video_path}, Total frames: {total_frames}, FPS: {fps}")

        fire_predictions = []
        frame_num = 0

        while frame_num < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_num}")
                break

            frame_resized = cv2.resize(frame, (299, 299))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_norm = frame_rgb.astype(np.float32) / 255.0

            try:
                fire_prob = model.predict(np.expand_dims(frame_norm, axis=0), verbose=0)[0][0]
                logger.debug(f"Frame {frame_num}: Fire probability = {fire_prob:.2f}")
            except Exception as e:
                logger.error(f"Prediction error at frame {  {str(e)}")
                raise

            fire_predictions.append((frame_num, fire_prob))

            # Update progress
            if progress_id and progress_id in progress_data:
                progress = (frame_num / total_frames) * 100
                progress_data[progress_id]['progress'] = progress
                logger.debug(f"Progress for {progress_id}: {progress:.2f}%")

            # Dynamic frame skipping
            if fire_prob > frame_threshold:
                frame_num += 1
            else:
                frame_num += frame_skip

        cap.release()

        # Apply voting
        fire_frames = []
        for i in range(len(fire_predictions) - vote_window + 1):
            window = fire_predictions[i:i + vote_window]
            fire_votes = sum(1 for (_, prob) in window if prob > frame_threshold)

            if fire_votes >= vote_majority:
                mid_frame = window[vote_window // 2][0]
                fire_frames.append(mid_frame)

        fire_frames = sorted(set(fire_frames))

        # Group frames into clips
        clips = []
        current_clip = []

        for i, frame_num in enumerate(fire_frames):
            if not current_clip:
                current_clip.append(frame_num)
            elif frame_num == current_clip[-1] + 1:
                current_clip.append(frame_num)
            else:
                if len(current_clip) / fps >= min_clip_length:
                    clips.append(current_clip)
                current_clip = [frame_num]

        if len(current_clip) / fps >= min_clip_length:
            clips.append(current_clip)

        logger.info(f"Total fire clips found: {len(clips)}")

        # Save clips
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to reopen video for saving clips: {video_path}")
            raise ValueError("Failed to reopen video file")

        for idx, clip in enumerate(clips):
            output_clip_path = os.path.join(output_dir, f"fire_clip_{idx + 1}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_clip_path, fourcc, fps, (299, 299))

            if not out.isOpened():
                logger.error(f"Failed to create output video: {output_clip_path}")
                raise IOError("Failed to create output video")

            for frame_num in clip:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (299, 299))
                    out.write(frame_resized)
                else:
                    logger.warning(f"Failed to read frame {frame_num} for clip {idx + 1}")

            out.release()
            logger.info(f"Saved clip: {output_clip_path}")

        cap.release()
        logger.info(f"Extracted {len(clips)} fire clips to '{output_dir}'")

    except Exception as e:
        logger.error(f"Error in extract_fire_clips: {str(e)}", exc_info=True)
        raise
