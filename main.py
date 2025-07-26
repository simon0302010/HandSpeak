import sys
import threading
import socket
import queue
from tkinter import Tk, Button, Label, Frame, messagebox
from obswebsocket import obsws, requests
sys.path.append('./main')
from detect import start_detection, stop_detection

# Setup
HOST = "localhost"
PORT = 4455
PASSWORD = ""

SOURCE_NAME = "MyNDISource"
NDI_SOURCE_FULL_NAME = ""

class DetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Detection Controller")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # State tracking
        self.detection_running = False
        self.detection_thread = None
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        
        # Get hostname for NDI source
        self.hostname = socket.gethostname()
        self.ndi_source_name = f"{self.hostname}.local (ASL-Detection-NDI)"
        
        self.setup_ui()
        
        # Start checking for messages from threads
        self.check_messages()
    
    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = Label(main_frame, text="ASL Detection Controller", 
                           font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions frame
        instructions_frame = Frame(main_frame)
        instructions_frame.pack(fill='x', pady=(0, 20))
        
        instructions_text = f"""After clicking "Start Detection":

1. Go to OBS
2. Double-click "MyNDISource" 
3. Click the dropdown for "Source name"
4. Select "{self.ndi_source_name}"

The detection window will open and start processing your hand gestures."""
        
        instructions_label = Label(instructions_frame, text=instructions_text, 
                                  justify='left', wraplength=450)
        instructions_label.pack()
        
        # Button frame
        button_frame = Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Detection button
        self.detection_button = Button(button_frame, text="Start Detection", 
                                      command=self.toggle_detection,
                                      font=("Arial", 12, "bold"),
                                      bg="#4CAF50", fg="white",
                                      width=15, height=2)
        self.detection_button.pack()
        
        # Status label
        self.status_label = Label(main_frame, text="Ready to start detection", 
                                 font=("Arial", 10), fg="gray")
        self.status_label.pack(pady=(10, 0))
    
    def check_messages(self):
        """Check for messages from threads and update GUI"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                if message['type'] == 'status':
                    self.status_label.config(text=message['text'], fg=message['color'])
                elif message['type'] == 'button':
                    self.detection_button.config(text=message['text'], bg=message['bg'])
                elif message['type'] == 'error':
                    messagebox.showerror("Error", message['text'])
                    self.detection_running = False
                    self.detection_button.config(text="Start Detection", bg="#4CAF50")
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_messages)
    
    def toggle_detection(self):
        print(f"Toggle detection called. Current state: {self.detection_running}")
        if not self.detection_running:
            print("Starting detection...")
            self.start_detection()
        else:
            print("Stopping detection...")
            self.stop_detection()
    
    def start_detection(self):
        try:
            # Set running state first
            self.detection_running = True
            
            # Update UI via queue
            self.message_queue.put({
                'type': 'button',
                'text': 'Stop Detection',
                'bg': '#f44336'
            })
            self.message_queue.put({
                'type': 'status',
                'text': 'Starting detection...',
                'color': 'blue'
            })
            
            # Run OBS setup in a separate thread
            obs_thread = threading.Thread(target=self.setup_obs)
            obs_thread.daemon = True
            obs_thread.start()
            
            # Start detection in a separate thread
            self.detection_thread = threading.Thread(target=self.run_detection)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Update status to running
            self.message_queue.put({
                'type': 'status',
                'text': 'Detection is running',
                'color': 'green'
            })
            
        except Exception as e:
            self.detection_running = False
            self.message_queue.put({
                'type': 'error',
                'text': f"Failed to start detection: {str(e)}"
            })
    
    def run_detection(self):
        """Wrapper to run detection and handle exceptions"""
        try:
            print("Detection thread started")
            print("About to call start_detection()...")
            
            # Test basic imports first
            print("Testing imports...")
            import sys
            sys.path.insert(0, 'custom_modules')
            print("Added custom_modules to path")
            
            from mediapipe.python.solutions import hands
            print("MediaPipe hands imported successfully")
            
            from actions import ActionHandler
            print("ActionHandler imported successfully")
            
            sys.path.append('../')
            from neuralnet import model as nn_model
            print("Neural network model imported successfully")
            
            print("All imports successful, calling start_detection()...")
            start_detection()
            print("Detection thread finished")
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            # Send message to main thread via queue
            self.message_queue.put({
                'type': 'error',
                'text': f"Detection error: {str(e)}"
            })
    
    def stop_detection(self):
        try:
            # Update UI via queue
            self.message_queue.put({
                'type': 'button',
                'text': 'Start Detection',
                'bg': '#4CAF50'
            })
            self.message_queue.put({
                'type': 'status',
                'text': 'Stopping detection...',
                'color': 'blue'
            })
            
            # Stop detection
            stop_detection()
            
            self.detection_running = False
            self.message_queue.put({
                'type': 'status',
                'text': 'Detection stopped',
                'color': 'gray'
            })
            
        except Exception as e:
            self.message_queue.put({
                'type': 'error',
                'text': f"Failed to stop detection: {str(e)}"
            })
            self.detection_running = False
    
    def setup_obs(self):
        """Setup OBS connection and NDI source"""
        try:
            ws = obsws(HOST, PORT, PASSWORD)
            ws.connect()
            
            # Get current scene
            scene_response = ws.call(requests.GetCurrentProgramScene())
            scene_name = scene_response.getSceneName()
            
            # Try removing old input
            try:
                items_resp = ws.call(requests.GetSceneItemList(sceneName=scene_name))
                for item in items_resp.getSceneItems():
                    if item['sourceName'] == SOURCE_NAME:
                        ws.call(requests.RemoveSceneItem(sceneName=scene_name, sceneItemId=item['sceneItemId']))
            except Exception:
                pass
            
            # Create new NDI input
            ws.call(requests.CreateInput(
                sceneName=scene_name,
                inputName=SOURCE_NAME,
                inputKind="ndi_source",
                sourceName={"source_name": self.ndi_source_name},
                sceneItemEnabled=True
            ))
            
            # Start virtual cam
            ws.call(requests.StartVirtualCam())
            
            ws.disconnect()
            
        except Exception as e:
            # Send message to main thread via queue
            self.message_queue.put({
                'type': 'status',
                'text': f"OBS setup failed: {str(e)}",
                'color': 'orange'
            })

def main():
    root = Tk()
    app = DetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()