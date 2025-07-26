import sys
import time
import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp
import NDIlib as ndi

def start_detection():
    try:
        sys.path.insert(0, 'custom_modules')
        from mediapipe.python.solutions import hands
        from actions import ActionHandler
        sys.path.append('../')
        from neuralnet import model as nn_model
    except ImportError as e:
        print(f"Import error: {e}")
        raise


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = 'main/model/efficientnet_model.pth'
    print(f"Loading model from: {model_path}")
    
    try:
        model_info = torch.load(model_path, map_location=device)
        model = nn_model.EfficientNetB0(num_classes=29).to(device)
        model.load_state_dict(model_info)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

    class_labels = {i: chr(65+i) if i < 26 else ['del','nothing','space'][i-26] for i in range(29)}
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


    print("Initializing NDI...")
    if not ndi.initialize():
        print("NDI init failed")
        return
    print("NDI initialized successfully")

    global cap, ndi_send
    
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ASL-Detection-NDI'
    ndi_send = ndi.send_create(send_settings)
    video_frame = ndi.VideoFrameV2()


    print("Attempting to open webcam...")
    cap = None
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        try:
            print(f"Trying camera index {camera_index}...")
            cap = cv.VideoCapture(camera_index)
            
            if cap.isOpened():
                # Set some properties to help with compatibility
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv.CAP_PROP_FPS, 30)
                
                # Try to read a frame to test if webcam is working
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Webcam opened successfully at index {camera_index}")
                    break
                else:
                    print(f"Camera {camera_index} opened but cannot read frames")
                    cap.release()
                    cap = None
            else:
                print(f"Camera {camera_index} failed to open")
                if cap:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"Error with camera {camera_index}: {e}")
            if cap:
                cap.release()
                cap = None
    
    if cap is None:
        print("Error: Could not open any webcam")
        return

    global detection_running
    detection_running = True
    try:
        print("[DEBUG] Detection Started!")
        while detection_running:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    continue
            except Exception as e:
                print(f"Error reading frame: {e}")
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
            video_frame.data = frame_bgra
            video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            video_frame.line_stride_in_bytes = frame_bgra.strides[0]
            video_frame.xres = frame_bgra.shape[1]
            video_frame.yres = frame_bgra.shape[0]

            ndi.send_send_video_v2(ndi_send, video_frame)
            
            # Show a simple status window (with error handling)
            try:
                status_frame = frame.copy()
                cv.putText(status_frame, f"Detection Running - {action}: {confidence_value*100:.1f}%", 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(status_frame, "Press ESC to stop", (10, 60), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv.imshow("ASL Detection Status", status_frame)
                
                if cv.waitKey(1) & 0xFF == 27:
                    break
            except Exception as e:
                print(f"Display error (continuing without window): {e}")
                # Continue without the display window
                pass

    finally:
        cap.release()
        ndi.send_destroy(ndi_send)
        ndi.destroy()
        cv.destroyAllWindows()

def stop_detection():
    global cap, ndi_send, detection_running
    detection_running = False
    if cap is not None:
        cap.release()
    if ndi_send is not None:
        ndi.send_destroy(ndi_send)
        ndi.destroy()
    cv.destroyAllWindows()

# Initialize global variables
cap = None
ndi_send = None
detection_running = False

if __name__ == "__main__":
    start_detection()
    time.sleep(5)