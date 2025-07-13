# AI Fire Detection

This repository contains an AI-based fire detection system that processes videos to identify and extract clips containing fire using a pre-trained deep learning model. The system uses a Convolutional Neural Network (CNN) to classify video frames as containing fire or not, applies dynamic frame skipping and majority voting for robust detection, and saves detected fire clips as MP4 files. A Flask-based web interface allows users to register, log in, upload videos, monitor processing progress, and view results.

The project is part of my deep learning coursework at the National University of Modern Languages, Islamabad, submitted on October 31, 2024, under the supervision of Mam Iqra Nasem. It builds on concepts from my deep learning labs, particularly CNN-based classification and image preprocessing.

## Features

- **Fire Detection**: Uses a pre-trained CNN model (`fire_detection_model_image.h5`) to detect fire in video frames with a probability threshold.
- **Dynamic Frame Skipping**: Optimizes processing by skipping frames when fire probability is low, reducing computation time.
- **Majority Voting**: Applies a voting mechanism over a window of frames to ensure robust fire detection.
- **Clip Extraction**: Saves fire-containing video segments as MP4 files, ensuring clips meet a minimum length requirement.
- **Web Interface**: A Flask-based frontend for user registration, login, video upload, progress tracking, and result viewing.
- **Logging**: Comprehensive logging for debugging and monitoring the detection process.

## Repository Structure

```
ai-fire-detection/
├── app.py                      # Flask application for the web interface
├── fire_detection.py           # Core script for fire detection and clip extraction
├── fire_detection_model_image.h5 # Pre-trained CNN model
├── templates/                  # HTML templates for the web interface
│   ├── index.html             # Home page
│   ├── login.html             # User login page
│   ├── logout.html            # User logout page
│   ├── progress.html          # Progress tracking page
│   ├── result.html            # Results display page
│   ├── register.html          # User registration page
│   ├── upload.html            # Video upload page
├── static/                    # Static files for the web interface
│   ├── css/                  # CSS stylesheets
│   │   ├── index.css
│   │   ├── login.css
│   │   ├── logout.css
│   │   ├── progress.css
│   │   ├── result.css
│   │   ├── register.css
│   │   ├── upload.css
│   ├── js/                   # JavaScript files
│   │   ├── index.js
│   │   ├── upload.js
├── outputs/                   # Directory for extracted fire clips
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── LICENSE                    # MIT License
```

## Related Coursework

This project applies concepts from my deep learning labs, particularly:

- **Lab 3: CNN Classification** (`lab_manuals/CNN_Classification.pdf`): Covers CNN architecture and image preprocessing, used in `fire_detection.py` for frame classification.
- **Lab 4: CNN Patterns** (`lab_manuals/CNN_Patterns.pdf`): Discusses image preprocessing techniques like normalization and edge detection, applied to video frames.
- **Lab 4+5: Advanced CNN** (`lab_manuals/CNN_Advanced.pdf`): Explores regularization and optimization, relevant to the pre-trained model’s design.

See the deep-learning-labs repository for lab manuals and related projects (e.g., `Pneumonia_detection_using_X-rays.ipynb`).

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/ai-fire-detection.git
   cd ai-fire-detection
   ```

2. **Install Dependencies**: Install the required Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   Key libraries include: `tensorflow`, `opencv-python`, `numpy`, `flask`, `logging`.

3. **Download the Pre-trained Model**:

   - Place the `fire_detection_model_image.h5` model in the root directory.
   - Alternatively, train your own model using a dataset like Kaggle Fire Detection Dataset.

4. **Run the Flask Application**: Start the Flask server to launch the web interface:

   ```bash
   python app.py
   ```

   Access the application at `http://localhost:5000` in your browser.

5. **Process Videos**:

   - Use the `upload.html` page to upload a video.
   - Monitor progress on the `progress.html` page (updated via `progress_data` in `app.py`).
   - View extracted fire clips on the `result.html` page, saved in the `outputs/` directory.

6. **Run Fire Detection Script Directly** (Optional): To process a video without the web interface:

   ```bash
   python fire_detection.py
   ```

   Update `video_path` and `output_dir` in `fire_detection.py` to specify input video and output directory.

## Usage

1. **Register/Login**: Use `register.html` to create an account and `login.html` to access the system.
2. **Upload Video**: Upload a video via `upload.html`. The system processes the video using `fire_detection.py`.
3. **Monitor Progress**: Check processing status on `progress.html`, which displays a percentage based on frame processing.
4. **View Results**: Access extracted fire clips on `result.html`, with MP4 files saved in `outputs/`.
5. **Logout**: Use `logout.html` to end the session.

## Configuration Parameters (fire_detection.py)

- `frame_skip`: Number of frames to skip when fire probability is low (default: 5).
- `min_clip_length`: Minimum length of extracted clips in seconds (default: 5).
- `frame_threshold`: Fire probability threshold for frame classification (default: 0.3).
- `vote_window`: Number of frames in the voting window (default: 10).
- `vote_majority`: Minimum number of fire votes in the window to confirm fire (default: 3).

## Example

To process a video named `input_video.mp4` and save fire clips to `outputs/`:

```python
from fire_detection import extract_fire_clips
extract_fire_clips(
    video_path="input_video.mp4",
    output_dir="outputs/",
    progress_id="video1",
    frame_skip=5,
    min_clip_length=5,
    frame_threshold=0.3,
    vote_window=10,
    vote_majority=3
)
```

## Future Improvements

- **Model Enhancement**: Fine-tune the CNN model with additional fire datasets to improve accuracy.
- **Real-time Processing**: Optimize `fire_detection.py` for real-time video streaming.
- **Metrics**: Include precision, recall, and F1-score for fire detection evaluation.
- **Multi-object Detection**: Extend the model to detect other objects (e.g., smoke) alongside fire.

## Notes

- **Model Path**: Ensure `fire_detection_model_image.h5` is in the correct path or update `MODEL_PATH` in `fire_detection.py`.
- **File Size**: Video outputs are saved as MP4 files in `outputs/`. Use Git LFS for large files if uploading to GitHub (`git lfs track "*.mp4"`).
- **Permissions**: The code and summarized lab manuals are shared with permission for educational purposes. Contact Mam Iqra Nasem for access to original course materials.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.