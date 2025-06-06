import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from PIL import Image

# Configuration
AGE_MODEL_PATH = "D:\\Pytorch_HumanAgeModel\\Models\\age_prediction_resnet50.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
def load_model(model_path, num_classes):
    # Load ResNet50 and modify the final layer
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # Use updated `weights` parameter
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load state_dict with strict=False to allow size mismatch resolution
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    if "fc.weight" in checkpoint and checkpoint["fc.weight"].shape[0] != num_classes:
        print("Adjusting final layer to match checkpoint's output size")
        model.fc = nn.Linear(model.fc.in_features, checkpoint["fc.weight"].shape[0])

    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# Age Model (Checkpoint has 8 classes for 8 different age groups ranging from 0 to 80)
age_model = load_model(AGE_MODEL_PATH, 8)

# Preprocessing for ResNet
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Face Detection Model (DNN)
face_net = cv2.dnn.readNetFromCaffe(
    "D:\\Pytorch_HumanAgeModel\\deploy.prototxt",
    "D:\\Pytorch_HumanAgeModel\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Set CUDA backend and target for face detection
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Age group labels (Adjust to match checkpoint if necessary)
age_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80"]

# Real-time Video Feed
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # Extract face region
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue  # Skip if face is empty

            # Preprocess face
            face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_pil)
            face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)

            # Age Prediction
            age_output = age_model(face_tensor)
            age_pred = torch.argmax(age_output, dim=1).item()
            age_label = age_labels[age_pred] if age_pred < len(age_labels) else "Unknown"
            # Display Predictions
            label = f"Age: {age_label}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Human Age Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
