<div align="center">
  <a href="https://shipwrecked.hackclub.com/?t=ghrm" target="_blank">
    <img src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/739361f1d440b17fc9e2f74e49fc185d86cbec14_badge.png" 
         alt="This project is part of Shipwrecked, the world's first hackathon on an island!" 
         style="width: 35%;">
  </a>
</div>


# HandSpeak: Real-Time ASL Detection to OBS Virtual Camera via NDI

This project streams ASL hand detection output directly to OBS using NDI and MediaPipe Hands.

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/gamerwaves/HandSpeak.git
cd HandSpeak
```

---

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

> **Note:**  
> If you're not using bash, the activation command may vary.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download and Install OBS Studio

- Download the latest version of OBS Studio from [https://obsproject.com/download](https://obsproject.com/download)

---

### 5. Download the DistroAV Plugin

- Download from: [https://obsproject.com/forum/resources/distroav-network-audio-video-in-obs-studio-using-ndi®-technology.528/](https://obsproject.com/forum/resources/distroav-network-audio-video-in-obs-studio-using-ndi®-technology.528/)
- DistroAV is a lightweight NDI display/preview and routing tool.
- Use the installation instructions on the download page for your os.

> **Note:**  
> On Linux, you also need to install **ndi-sdk** using your distribution's package manager.

---

### 6. Run the Detection Script

```bash
python main/detect.py
```

This will start MediaPipe hand tracking and send frames over NDI.

---

### 7. Add NDI Source in OBS

- In OBS, click the **➕ (plus)** button under *Sources*.
- Select **NDI Source**.
- Click the small **dropdown arrow** beside the source name.
- Select the NDI stream titled `ASL-Detection-NDI`.

---

### 8. Start OBS Virtual Camera

- Click **Start Virtual Camera** in OBS.
- You can now use this virtual camera in Zoom, Google Meet, Discord, etc.

---

## ✅ Notes

- Make sure OBS and the Python script are on the same network (localhost is fine).
- DistroAV requires the latest OBS version for it to work properly.

---

## 🧠 Credits

- Built with [MediaPipe](https://mediapipe.dev), [OpenCV](https://opencv.org), and [NDI](https://www.ndi.tv/tools/).
- OBS integration via `obs-websocket` and `obs-ndi`.
- README.md by ChatGPT :)
