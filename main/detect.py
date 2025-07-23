import sys
import time
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, 'custom_modules')
from mediapipe.python.solutions import hands
import mediapipe as mp
# use cyndilib
from cyndilib.sender import Sender
from cyndilib.video_frame import VideoSendFrame
from cyndilib.wrapper.ndi_structs import FourCC
from actions import ActionHandler
sys.path.append('../')
from neuralnet import model as nn_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'main/model/efficientnet_model.pth'
model_info = torch.load(model_path, map_location=device)
model = nn_model.EfficientNetB0(num_classes=29).to(device)
model.load_state_dict(model_info)
model.eval()

class_labels = {i: chr(65+i) if i < 26 else ['del','nothing','space'][i-26] for i in range(29)}
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

sender = Sender("ASL-Detection-NDI")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    sys.exit(1)
    
ret, test_frame = cap.read()
if not ret:
    print("Error: Cannot read from webcam")
    sys.exit(1)
    
height, width = test_frame.shape[:2]

video_frame = VideoSendFrame()
video_frame.set_resolution(width, height)
video_frame.set_frame_rate(30)
video_frame.set_fourcc(FourCC.BGRA)

sender.set_video_frame(video_frame)

try:
    with sender:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            bbox = None
            predicted_class = None
            confidence_value = 0
            action = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    points = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
                    x_min, y_min, _ = np.min(points, axis=0)
                    x_max, y_max, _ = np.max(points, axis=0)
                    padding = 0.05
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(1, x_max + padding), min(1, y_max + padding)
                    bbox = [int(x_min * frame.shape[1]), int(y_min * frame.shape[0]),
                            int(x_max * frame.shape[1]), int(y_max * frame.shape[0])]

                    hand_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if hand_img.size == 0:
                        bbox = None
                        continue

                    pil_img = Image.fromarray(hand_img)
                    pil_img = transform(pil_img).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        outputs = model(pil_img)
                        _, predicted = torch.max(outputs, 1)
                        confidence_value = F.softmax(outputs, dim=1).max().item()
                        predicted_class = predicted.item()
                        action = class_labels[predicted_class]

                        handler = ActionHandler(confidence_value, action)
                        handler.execute_action()

                    color = (0, 255, 0) if confidence_value >= 0.96 else (0, 0, 255)
                    label = f"{action}: {confidence_value * 100:.2f}%" if action else "Low Confidence"
                    (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                    cv.rectangle(frame, (bbox[0], bbox[1] - th - 20), (bbox[0] + tw + 20, bbox[1]), (255, 255, 255), -1)
                    cv.putText(frame, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)


            frame_bgra = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
            frame_buffer = frame_bgra.flatten()
            sender.write_video_async(frame_buffer)
            
            cv.imshow("ASL Detection", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

finally:
    cap.release()
    cv.destroyAllWindows()