from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import json
import pyautogui
import time
from datetime import datetime
import os
import subprocess
import win32gui
import win32con
import win32api
import ctypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import threading
from queue import Queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Allow detection of up to 2 hands
    min_detection_confidence=0.5,  # Lower threshold for better detection
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize gesture history
gesture_history = []
MAX_HISTORY = 10

# Track hand position for swipe detection
last_hand_position = None
swipe_threshold = 100  # pixels

# Add gesture stabilization
GESTURE_STABILITY_THRESHOLD = 5  # Number of frames to confirm a gesture
current_gesture_count = 0
last_stable_gesture = None

# Add scroll control variables
last_scroll_position = None
scroll_threshold = 50

# Add these global variables after other initializations
frame_queue = Queue(maxsize=2)
camera_lock = threading.Lock()
camera_active = False
target_fps = 30
frame_interval = 1.0 / target_fps

# Add these new variables
GESTURE_HISTORY = deque(maxlen=5)  # Store last 5 gestures
CUSTOM_GESTURES = {}  # Store custom gesture mappings
GESTURE_COMBINATIONS = {
    ('peace', 'thumbs_up'): 'take_screenshot',
    ('fist', 'open_palm'): 'toggle_presentation',
    ('three_fingers', 'pointing'): 'switch_window'
}

# Add these imports at the top
import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Add these global variables after other initializations
executor = ThreadPoolExecutor(max_workers=2)
frame_skip = 2  # Process every 2nd frame
frame_count = 0
gesture_visualization = {
    'open_palm': {'color': (0, 255, 0), 'text': 'Open Palm'},
    'pointing': {'color': (255, 0, 0), 'text': 'Pointing'},
    'thumbs_up': {'color': (0, 0, 255), 'text': 'Thumbs Up'},
    'fist': {'color': (255, 255, 0), 'text': 'Fist'},
    'peace': {'color': (255, 0, 255), 'text': 'Peace'},
    'three_fingers': {'color': (0, 255, 255), 'text': 'Three Fingers'},
    'four_fingers': {'color': (128, 0, 128), 'text': 'Four Fingers'}
}

def get_audio_interface():
    """Get system audio interface"""
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

def set_brightness(level):
    """Set screen brightness (Windows only)"""
    try:
        # Convert brightness level (0-100) to Windows brightness value (0-100)
        brightness = int(level)
        ctypes.windll.powrprof.PowerWriteACValueIndex(0, 1, 0, brightness)
        return True
    except:
        return False

def get_current_brightness():
    """Get current screen brightness"""
    try:
        return ctypes.windll.powrprof.PowerReadACValueIndex(0, 1, 0)
    except:
        return 50

def control_window(action):
    """Control active window"""
    try:
        hwnd = win32gui.GetForegroundWindow()
        if action == "minimize":
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        elif action == "maximize":
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
        elif action == "close":
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        return True
    except:
        return False

def control_media(action):
    """Control media playback"""
    if action == "play_pause":
        pyautogui.press("playpause")
    elif action == "next":
        pyautogui.press("nexttrack")
    elif action == "previous":
        pyautogui.press("prevtrack")
    elif action == "volume_up":
        pyautogui.press("volumeup")
    elif action == "volume_down":
        pyautogui.press("volumedown")
    elif action == "mute":
        pyautogui.press("volumemute")

def detect_scroll(hand_landmarks, frame_height):
    """Detect vertical scroll gesture"""
    global last_scroll_position
    
    # Get middle finger tip position
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    current_y = int(middle_tip.y * frame_height)
    
    if last_scroll_position is None:
        last_scroll_position = current_y
        return None
    
    # Calculate vertical movement
    movement = current_y - last_scroll_position
    last_scroll_position = current_y
    
    # Detect scroll
    if abs(movement) > scroll_threshold:
        if movement > 0:
            return "scroll_down"
        else:
            return "scroll_up"
    
    return None

def detect_zoom(hand_landmarks):
    """Detect zoom gesture using distance between thumb and index finger"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate distance between thumb and index finger
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
    if distance > 0.2:  # Threshold for zoom out
        return "zoom_out"
    elif distance < 0.1:  # Threshold for zoom in
        return "zoom_in"
    return None

def is_finger_extended(finger_tip, finger_base, finger_mcp, threshold=0.1):
    """Improved finger extension detection"""
    # Check vertical position
    vertical_extended = finger_tip.y < finger_base.y
    
    # Check distance from palm
    distance = np.sqrt((finger_tip.x - finger_mcp.x)**2 + (finger_tip.y - finger_mcp.y)**2)
    distance_extended = distance > threshold
    
    return vertical_extended and distance_extended

def calculate_volume(hand_landmarks):
    # Get index finger tip position
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    # Convert y position to volume (0-100)
    volume = int((1 - index_tip.y) * 100)
    return max(0, min(100, volume))

def control_mouse(hand_landmarks, frame_width, frame_height):
    # Get index finger tip position
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Convert coordinates to screen position
    screen_x = int(index_tip.x * screen_width)
    screen_y = int(index_tip.y * screen_height)
    
    # Move mouse cursor
    pyautogui.moveTo(screen_x, screen_y)

def detect_swipe(hand_landmarks, frame_width):
    global last_hand_position
    
    # Get current hand position
    current_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width)
    
    if last_hand_position is None:
        last_hand_position = current_x
        return None
    
    # Calculate movement
    movement = current_x - last_hand_position
    
    # Update last position
    last_hand_position = current_x
    
    # Detect swipe
    if abs(movement) > swipe_threshold:
        if movement > 0:
            return "swipe_right"
        else:
            return "swipe_left"
    
    return None

def take_screenshot():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    return filename

def open_application(app_name):
    try:
        subprocess.Popen(app_name)
        return True
    except:
        return False

def control_presentation(gesture):
    if gesture == "swipe_right":
        pyautogui.press("right")  # Next slide
    elif gesture == "swipe_left":
        pyautogui.press("left")   # Previous slide
    elif gesture == "open_palm":
        pyautogui.press("f5")     # Start presentation

def load_custom_gestures():
    """Load custom gesture mappings from file"""
    global CUSTOM_GESTURES
    try:
        if os.path.exists('custom_gestures.json'):
            with open('custom_gestures.json', 'r') as f:
                CUSTOM_GESTURES = json.load(f)
    except Exception as e:
        print(f"Error loading custom gestures: {e}")

def save_custom_gestures():
    """Save custom gesture mappings to file"""
    try:
        with open('custom_gestures.json', 'w') as f:
            json.dump(CUSTOM_GESTURES, f)
    except Exception as e:
        print(f"Error saving custom gestures: {e}")

def detect_gesture_combination():
    """Detect gesture combinations from history"""
    if len(GESTURE_HISTORY) >= 2:
        last_two = tuple(GESTURE_HISTORY[-2:])
        return GESTURE_COMBINATIONS.get(last_two)
    return None

def perform_advanced_action(action):
    """Perform advanced system actions"""
    try:
        if action == 'open_palm':
            take_screenshot()
            return "Screenshot taken"
        elif action == 'toggle_presentation':
            pyautogui.press('f5')
            return "Presentation toggled"
        elif action == 'switch_window':
            pyautogui.hotkey('alt', 'tab')
            return "Window switched"
        elif action == 'volume_up':
            pyautogui.press('volumeup')
            return "Volume increased"
        elif action == 'volume_down':
            pyautogui.press('volumedown')
            return "Volume decreased"
        elif action == 'mute':
            pyautogui.press('volumemute')
            return "Audio muted"
        elif action == 'brightness_up':
            current = get_current_brightness()
            set_brightness(min(current + 10, 100))
            return "Brightness increased"
        elif action == 'brightness_down':
            current = get_current_brightness()
            set_brightness(max(current - 10, 0))
            return "Brightness decreased"
        elif action == 'minimize_window':
            control_window("minimize")
            return "Window minimized"
        elif action == 'maximize_window':
            control_window("maximize")
            return "Window maximized"
        elif action == 'close_window':
            control_window("close")
            return "Window closed"
        elif action == 'play_pause':
            pyautogui.press('playpause')
            return "Media play/pause"
        elif action == 'next_track':
            pyautogui.press('nexttrack')
            return "Next track"
        elif action == 'previous_track':
            pyautogui.press('prevtrack')
            return "Previous track"
    except Exception as e:
        print(f"Error performing action: {e}")
        return None

def detect_gesture(hand_landmarks):
    """Enhanced gesture detection with advanced features"""
    # Get finger states with more lenient thresholds
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Get finger base positions
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Check if fingers are extended
    thumb_extended = is_finger_extended(thumb_tip, thumb_base, thumb_base, threshold=0.1)
    index_extended = is_finger_extended(index_tip, index_base, index_base, threshold=0.1)
    middle_extended = is_finger_extended(middle_tip, middle_base, middle_base, threshold=0.1)
    ring_extended = is_finger_extended(ring_tip, ring_base, ring_base, threshold=0.1)
    pinky_extended = is_finger_extended(pinky_tip, pinky_base, pinky_base, threshold=0.1)

    # Detect gestures
    gesture = None
    if (thumb_extended and index_extended and middle_extended and 
        ring_extended and pinky_extended):
        gesture = "open_palm"
    elif (not thumb_extended and index_extended and 
          not middle_extended and not ring_extended and not pinky_extended):
        gesture = "pointing"
    elif (thumb_extended and not index_extended and 
          not middle_extended and not ring_extended and not pinky_extended):
        gesture = "thumbs_up"
    elif (not thumb_extended and not index_extended and 
          not middle_extended and not ring_extended and not pinky_extended):
        gesture = "fist"
    elif (not thumb_extended and index_extended and 
          middle_extended and not ring_extended and not pinky_extended):
        gesture = "peace"
    elif (not thumb_extended and index_extended and 
          middle_extended and ring_extended and not pinky_extended):
        gesture = "three_fingers"
    elif (not thumb_extended and index_extended and 
          middle_extended and ring_extended and pinky_extended):
        gesture = "four_fingers"

    # Add gesture to history
    if gesture:
        GESTURE_HISTORY.append(gesture)
        
        # Check for gesture combinations
        combination_action = detect_gesture_combination()
        if combination_action:
            return f"{gesture}+{combination_action}"

    return gesture

def camera_thread():
    """Separate thread for camera capture"""
    global camera_active, frame_queue
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Log camera properties
    print("Camera properties:")
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    
    last_frame_time = time.time()
    
    while camera_active:
        current_time = time.time()
        if current_time - last_frame_time < frame_interval:
            time.sleep(0.001)  # Small sleep to prevent CPU overuse
            continue
            
        with camera_lock:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Log frame properties
            print(f"Frame shape: {frame.shape}, Mean brightness: {np.mean(frame)}")
                
            # Clear old frames from queue
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Queue.Empty:
                    break
                    
            # Add new frame to queue
            try:
                frame_queue.put(frame, block=False)
            except Queue.Full:
                pass
                
        last_frame_time = current_time
    
    cap.release()

def optimize_frame(frame):
    """Optimize frame for processing"""
    # Resize frame for faster processing
    frame = cv2.resize(frame, (320, 240))
    
    # Adjust brightness and contrast
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, rgb_frame

def draw_gesture_info(frame, gesture, action_result):
    """Draw advanced gesture visualization"""
    if gesture in gesture_visualization:
        viz = gesture_visualization[gesture]
        # Draw gesture name with background
        text = viz['text']
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 30), (20 + text_width, 60), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, viz['color'], 2)
        
        # Draw action result if available
        if action_result:
            cv2.rectangle(frame, (10, 70), (20 + text_width, 100), (0, 0, 0), -1)
            cv2.putText(frame, action_result, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw gesture history
        y_pos = 130
        for hist_gesture in GESTURE_HISTORY:
            if hist_gesture in gesture_visualization:
                hist_viz = gesture_visualization[hist_gesture]
                cv2.rectangle(frame, (10, y_pos), (20 + text_width, y_pos + 30), (0, 0, 0), -1)
                cv2.putText(frame, hist_viz['text'], (15, y_pos + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, hist_viz['color'], 2)
                y_pos += 40

def generate_frames():
    """Generate frames for video feed with optimized performance"""
    global camera_active, frame_queue, frame_count
    
    camera_active = True
    last_frame_time = time.time()
    
    # Initialize camera with optimized settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set optimized camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    try:
        while camera_active:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture frame")
                break
            
            # Skip frames for better performance
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # Optimize frame
            frame, rgb_frame = optimize_frame(frame)
            
            # Process frame in separate thread
            future = executor.submit(process_frame, rgb_frame)
            results = future.result()
            
            if results:
                # Draw hand landmarks and gesture info
                display_frame = draw_results(frame, results)
                
                # Add FPS counter
                fps = 1.0 / (time.time() - last_frame_time)
                cv2.putText(display_frame, f"FPS: {int(fps)}", 
                           (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Encode frame with optimized settings
                ret, buffer = cv2.imencode('.jpg', display_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            last_frame_time = time.time()
            
    finally:
        camera_active = False
        cap.release()

def process_frame(rgb_frame):
    """Process frame in separate thread"""
    try:
        # Process the frame and detect hands
        results = hands.process(rgb_frame)
        return results
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def draw_results(frame, results):
    """Draw results on frame"""
    display_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks with improved visibility
            mp_draw.draw_landmarks(
                display_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            
            # Perform action if gesture detected
            action_result = None
            if gesture:
                if '+' in gesture:  # Handle gesture combinations
                    gesture, action = gesture.split('+')
                    action_result = perform_advanced_action(action)
                else:
                    action_result = perform_advanced_action(gesture)
            
            # Draw advanced gesture visualization
            draw_gesture_info(display_frame, gesture, action_result)
            
            # Emit gesture and action through WebSocket
            socketio.emit('gesture_detected', {
                'gesture': gesture,
                'action': action_result,
                'history': list(GESTURE_HISTORY)
            })
    else:
        # If no hands detected, emit None
        socketio.emit('gesture_detected', {
            'gesture': None,
            'action': None,
            'history': list(GESTURE_HISTORY)
        })
    
    return display_frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/controls')
def controls():
    return render_template('controls.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_preview')
def camera_preview():
    """Camera preview route"""
    return render_template('camera_preview.html')

@app.route('/custom_gestures')
def get_custom_gestures():
    return jsonify(CUSTOM_GESTURES)

@app.route('/gesture_history')
def get_gesture_history():
    return jsonify(list(GESTURE_HISTORY))

# Add cleanup on application exit
@app.teardown_appcontext
def cleanup(exception=None):
    global camera_active
    camera_active = False

# Initialize custom gestures on startup
load_custom_gestures()

if __name__ == '__main__':
    socketio.run(app, debug=True) 