# LifeVision

A machine learning pipeline that processes Instagram images to detect emotional cues and estimate suicide risk based on facial expressions.

## Table of Contents

* [Background](#background)
* [Dataset Preparation](#dataset-preparation)
* [Preprocessing & Face Extraction](#preprocessing--face-extraction)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Usage](#usage)
* [Ethical Considerations](#ethical-considerations)
* [Future Work](#future-work)
* [License](#license)

## Background

Suicide is a global public health issue, with over 50,000 deaths annually. Research shows that 6–13% of suicide cases can be linked to social media activity. This project aims to leverage facial emotion analysis on Instagram posts to identify individuals at higher risk of suicidal ideation.

## Dataset Preparation

1. **Image Lists**: Two text files (`normal.txt`, `depressed.txt`) list filenames of images downloaded from Instagram profiles.
2. **Directory Structure**:

   ```
   instagram-emotion-detection/
   ├── normal_pics/         # Raw images labeled 'normal'
   ├── depressed_pics/      # Raw images labeled 'depressed'
   ├── data_set/            # Organized images for face extraction
   │   ├── normal/          # Cropped faces from normal images
   │   └── depressed/       # Cropped faces from depressed images
   ├── dataset/             # Final face samples for training
   └── trainer/             # Serialized model files
   ```

## Preprocessing & Face Extraction

* **Copy Images**: Use OpenCV script to organize raw images into `data_set/normal` and `data_set/depressed`.
* **Face Detection**: Cascade classifier (`haarcascade_frontalface_default.xml`) detects and crops faces from each photo.
* **Labeling**: User provides an emotion ID (0 = Normal, 1 = Depressed). Cropped faces are renamed `User.<ID>.<count>.jpg` and saved in `dataset/`.

```python
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # Crop and save each face with appropriate label
```

## Model Training

* **Algorithm**: Local Binary Patterns Histograms (LBPH) for facial recognition and emotion classification.
* **Training Pipeline**:

  1. Load images and labels from `dataset/`.
  2. Convert to grayscale NumPy arrays.
  3. Train LBPH recognizer and save model to `trainer/trainer.yml`.

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/trainer.yml')
```

## Evaluation

* **Test Inference**: Load `trainer.yml` and predict on new Instagram images.
* **Metrics**: Display predicted label (`Normal` or `Depressed`) and confidence percentage.
* **Visualization**: Save annotated result image and confidence score.

```python
id, confidence = recognizer.predict(face_region)
print(f"Prediction: {names[id]} ({100 - confidence:.1f}% confidence)")
```

## Usage

1. Prepare your environment:

   ```bash
   pip install -r requirements.txt
   python -m pip install opencv-python-headless pillow
   ```
2. Populate `normal_pics/` and `depressed_pics/` and update `normal.txt`, `depressed.txt`.
3. Run preprocessing and face extraction:

   ```bash
   python preprocess_faces.py
   ```
4. Train model:

   ```bash
   python train_model.py
   ```
5. Predict on a new image:

   ```bash
   python predict.py --image path/to/instagram_image.jpg
   ```

## Ethical Considerations

* **Privacy**: All Instagram images should be obtained with explicit consent.
* **Bias**: Emotion classification may vary across demographics. Validate models on diverse populations.
* **Interpretability**: High-risk predictions should not be used in isolation; always involve mental health professionals.

## Future Work

* Integrate deep learning-based emotion recognition (e.g., CNNs, transformers).
* Expand labels to multi-class emotions (sadness, anger, fear, etc.).
* Build a secure web dashboard for clinicians to review flagged cases.

## License

*No license specified.* All rights reserved. Please contact the repository owner for licensing details.
