import cv2
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image
import numpy as np

# Laad het model en de feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Open de webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the image and make a prediction
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top 5 predicted classes and their probabilities
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    # Display the top results on the frame
    for i in range(top5_prob.size(1)):
        label = model.config.id2label[top5_catid[0, i].item()]
        prob = top5_prob[0, i].item() * 100
        cv2.putText(frame, f'{label}: {prob:.2f}%', (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
