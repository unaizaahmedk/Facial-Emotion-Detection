## Facial Emotion Detection

A deep learning project that detects and classifies human facial expressions into 7 emotion categories in both static images and real-time webcam feeds, using a Convolutional Neural Network trained on the FER-2013 dataset.

---

### Features

**1. Data Preprocessing**
Images are loaded in grayscale, resized to 48x48 pixels, and normalized to a 0-1 range. Labels are extracted from folder names and encoded using LabelEncoder, then converted to one-hot vectors for multi-class classification.

**2. CNN Model**
```
Sequential Model
├── Conv2D (128 filters, 3x3, ReLU)
│   ├── MaxPooling2D (2x2)
│   └── Dropout (0.4)
├── Conv2D (256 filters, 3x3, ReLU)
│   ├── MaxPooling2D (2x2)
│   └── Dropout (0.4)
├── Conv2D (512 filters, 3x3, ReLU)
│   ├── MaxPooling2D (2x2)
│   └── Dropout (0.4)
├── Flatten
├── Dense (512, ReLU)
│   └── Dropout (0.4)
├── Dense (256, ReLU)
│   └── Dropout (0.3)
└── Output Dense (7, Softmax)
```

**3. Static Image Prediction**
A preprocessed image is passed through the trained model to predict and display the emotion label along with a visualization using Matplotlib.

**4. Real-Time Emotion Detection**
OpenCV captures a live webcam feed. A Haar Cascade classifier detects faces in each frame, which are then cropped, preprocessed, and passed through the model. The predicted emotion label is overlaid on the frame using bounding boxes and text annotations in real time.

---

### Emotions Classified

`angry` `disgust` `fear` `happy` `neutral` `sad` `surprise`

---

### Tech Stack

| | |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Preprocessing | scikit-learn, tqdm |

---

### Project Structure

```
Facial-Emotion-Detection/
├── images/
│   ├── train/                  # Training images organized by emotion label
│   └── test/                   # Test images organized by emotion label
├── notebooks/
│   └── emotion_detection.ipynb # Main notebook with full pipeline
├── model/                      # Create this folder manually before running
│   ├── emotiondetector.json    # Saved model architecture (generated after training)
│   └── emotiondetector.h5      # Saved model weights (generated after training)
└── README.md
```

> **Note:** The `model/` folder is not included in the repository. Please create it manually in the root directory before running the notebook, as the trained model architecture and weights will be saved there after training.

---

### Installation

**Prerequisites:** Python 3.8+, pip, Jupyter Notebook

```bash
# Clone the repository
git clone https://github.com/unaizaahmedk/Facial-Emotion-Detection.git
cd Facial-Emotion-Detection

# Install dependencies
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn tqdm
```

---

### Usage

**1. Train the model**

Open and run `notebooks/emotion_detection.ipynb`. The trained model will be saved to the `model/` folder.

**2. Test on a static image**

Update the `image_path` variable in the notebook to point to your image and run the prediction cell.

**3. Run real-time detection**

Run the webcam section of the notebook. Press `q` to quit the webcam feed.

---

### Model Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 128 |
| Epochs | 30 |
| Input Shape | 48x48x1 (grayscale) |
| Output Classes | 7 |

---
