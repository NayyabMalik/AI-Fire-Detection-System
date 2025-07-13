from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
import logging
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure key in production
app.permanent_session_lifetime = timedelta(days=7)  # Sessions last 7 days

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017')
    db = client['fire_detection']
    users = db['users']
    logger.info("MongoDB connected successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Directories
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'extracted_clips'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model path
MODEL_PATH = r"C:\Users\PMLS\Documents\ai fire detection frontend\fire_detection_model_image.h5"

# Load the trained model
try:
    model = load_model(MODEL_PATH)
    # Optional: Compile model to suppress warning (unnecessary for inference)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Store progress (simple in-memory storage; use Redis/database in production)
progress_data = {}

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

        print("[INFO] Starting frame analysis with dynamic frame skipping...")
        logger.debug(f"Starting frame analysis for progress_id: {progress_id}")

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
                print(f"Frame {frame_num}: Fire probability = {fire_prob:.2f}")
                logger.debug(f"Frame {frame_num}: Fire probability = {fire_prob:.2f}")
            except Exception as e:
                logger.error(f"Prediction error at frame {frame_num}: {str(e)}")
                raise

            fire_predictions.append((frame_num, fire_prob))

            # Update progress
            if progress_id and progress_id in progress_data:
                progress = (frame_num / total_frames) * 100
                progress_data[progress_id]['progress'] = progress
                logger.debug(f"Progress for {progress_id}: {progress:.2f}%")
            else:
                logger.warning(f"Progress not updated: progress_id {progress_id} not in progress_data")

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

        print(f"[INFO] Total fire clips found: {len(clips)}")
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
            print(f"[SAVED] {output_clip_path}")
            logger.info(f"Saved clip: {output_clip_path}")

        cap.release()
        print(f"[DONE] Extracted {len(clips)} fire clips to '{output_dir}'")
        logger.info(f"Extracted {len(clips)} fire clips to '{output_dir}'")

    except Exception as e:
        logger.error(f"Error in extract_fire_clips: {str(e)}", exc_info=True)
        raise

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        logger.debug("User already logged in, redirecting to upload")
        return redirect(url_for('upload'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        logger.debug("User already logged in, redirecting to upload")
        return redirect(url_for('upload'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.find_one({'email': email})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session.permanent = True  # Make session persistent
            logger.info(f"User {email} logged in successfully")
            return redirect(url_for('upload'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        logger.debug("User already logged in, redirecting to upload")
        return redirect(url_for('upload'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if users.find_one({'email': email}):
            return render_template('register.html', error='Email already exists')
        hashed_password = generate_password_hash(password)
        users.insert_one({
            'username': username,
            'email': email,
            'password': hashed_password
        })
        logger.info(f"User {email} registered successfully")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        logger.debug("No user session, redirecting to login")
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                logger.error("No video file provided in request")
                return jsonify({'error': 'No video file provided'}), 400
            file = request.files['video']
            if file.filename == '':
                logger.error("No selected file")
                return jsonify({'error': 'No selected file'}), 400
            
            # Check if processing is already in progress
            if session.get('processing_id') and session['processing_id'] in progress_data and not progress_data[session['processing_id']]['done']:
                logger.warning("Processing already in progress for this session")
                return jsonify({'error': 'Processing already in progress'}), 400
            
            filename = f"{uuid.uuid4()}_{file.filename}"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            logger.info(f"Video saved to {video_path}")
            
            # Store processing session
            processing_id = str(uuid.uuid4())
            session['processing_id'] = processing_id
            progress_data[processing_id] = {'progress': 0, 'done': False}
            logger.debug(f"Started processing with ID: {processing_id}")
            
            # Process video synchronously
            try:
                extract_fire_clips(
                    video_path=video_path,
                    output_dir=EXTRACTED_FOLDER,
                    progress_id=processing_id,
                    frame_skip=5,
                    min_clip_length=5,
                    frame_threshold=0.3,
                    vote_window=10,
                    vote_majority=3
                )
                progress_data[processing_id]['done'] = True
                logger.info("Video processing completed")
            except Exception as e:
                logger.error(f"Failed to process video with extract_fire_clips: {str(e)}", exc_info=True)
                raise ValueError(f"Video processing failed: {str(e)}")
            
            return jsonify({'message': 'Processing complete', 'redirect': url_for('results')})
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            return jsonify({'error': f"Server error: {str(e)}"}), 500
    return render_template('upload.html')

@app.route('/progress')
def get_progress():
    try:
        processing_id = session.get('processing_id')
        logger.debug(f"Fetching progress for processing_id: {processing_id}")
        if processing_id and processing_id in progress_data:
            logger.debug(f"Progress data: {progress_data[processing_id]}")
            return jsonify(progress_data[processing_id])
        logger.warning("No progress data found for session")
        return jsonify({'progress': 0, 'done': False})
    except Exception as e:
        logger.error(f"Error fetching progress: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/results')
def results():
    if 'user_id' not in session:
        logger.debug("No user session, redirecting to login")
        return redirect(url_for('login'))
    try:
        clips = [
            {'url': f"/extracted_clips/{file}"}
            for file in os.listdir(EXTRACTED_FOLDER)
            if file.endswith('.mp4')
        ]
        logger.info(f"Found {len(clips)} clips in results")
        return render_template('results.html', clips=clips)
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return render_template('results.html', clips=[], error=str(e))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('processing_id', None)
    logger.info("User logged out")
    return render_template('logout.html')

# Serve extracted clips
@app.route('/extracted_clips/<filename>')
def serve_clip(filename):
    try:
        return send_from_directory(EXTRACTED_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving clip {filename}: {str(e)}")
        return jsonify({'error': f"File not found: {str(e)}"}), 404

if __name__ == '__main__':
    app.run()