

## Overview

This project captures real-time video from a webcam, processes the video frames using a Vision Transformer (ViT) model for image classification, and displays the top 5 predicted classes with their probabilities on the video feed.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x installed on your local machine.
- Necessary Python packages installed (see below).
- A webcam connected to your machine.

### Required Python Packages

You can install the required packages using pip:

```sh
pip install opencv-python-headless transformers torch pillow numpy
```

## Setup

1. Clone the repository or download the script to your local machine.
2. Ensure your webcam is connected and recognized by your system.

## Usage

1. Run the script:

```sh
python script_name.py
```

2. The script will open your webcam, capture video frames, process each frame using the ViT model, and display the top 5 predicted classes with their probabilities on the video feed.

### Important Variables and Functions

#### Variables

- `feature_extractor`: An instance of `ViTFeatureExtractor` to preprocess the images.
- `model`: An instance of `ViTForImageClassification` to perform the image classification.
- `cap`: The video capture object to access the webcam.

#### Functions

- `cv2.VideoCapture(0)`: Opens the default webcam.
- `cv2.putText(frame, text, org, font, fontScale, color, thickness, lineType)`: Adds text to the video frame.
- `cv2.imshow(window_name, frame)`: Displays the video frame in a window.
- `cv2.waitKey(delay)`: Waits for a key event.

### Main Process

1. **Load the Model and Feature Extractor:**
   ```python
   feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
   model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
   ```

2. **Open the Webcam:**
   ```python
   cap = cv2.VideoCapture(0)
   if not cap.isOpened():
       print("Error: Could not open video.")
       exit()
   ```

3. **Capture and Process Video Frames:**
   ```python
   while True:
       ret, frame = cap.read()
       if not ret:
           print("Failed to grab frame")
           break

       image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       inputs = feature_extractor(images=image, return_tensors="pt")
       with torch.no_grad():
           outputs = model(**inputs)
           logits = outputs.logits
           probs = torch.nn.functional.softmax(logits, dim=-1)

       top5_prob, top5_catid = torch.topk(probs, 5)

       for i in range(top5_prob.size(1)):
           label = model.config.id2label[top5_catid[0, i].item()]
           prob = top5_prob[0, i].item() * 100
           cv2.putText(frame, f'{label}: {prob:.2f}%', (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

       cv2.imshow('Video', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   ```

4. **Release Resources:**
   ```python
   cap.release()
   cv2.destroyAllWindows()
   ```

## Contributing

If you want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
