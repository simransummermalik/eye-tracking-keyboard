import cv2
import mediapipe as mp
import math
import time
import numpy as np
from collections import deque

class WorkingEyeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Set up webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get camera dimensions
        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Keyboard layout
        self.keys = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'DEL', 'CLEAR']
        ]
        
        # Keyboard properties
        self.key_width = 100
        self.key_height = 80
        self.key_margin = 8
        self.keyboard_start_y = self.cam_height - 350
        
        # Simple smoothing - just a small buffer
        self.gaze_buffer = deque(maxlen=3)  # Very small buffer
        
        # Dwell system
        self.dwell_time = 1.2
        self.current_key = None
        self.dwell_start_time = 0
        
        # Text output
        self.typed_text = ""
        
        # Simple calibration - just 5 points
        self.calibration_mode = True  # Start with calibration
        self.calibration_points = [
            (200, 200),   # Top-left
            (1080, 200),  # Top-right
            (640, 360),   # Center
            (200, 520),   # Bottom-left
            (1080, 520),  # Bottom-right
        ]
        self.current_calibration_point = 0
        self.calibration_data = []
        
        # Simple mapping
        self.gaze_scale_x = 1.0
        self.gaze_scale_y = 1.0
        self.gaze_offset_x = 0
        self.gaze_offset_y = 0
        
        # Eye landmarks - simplified
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Key positions
        self.key_positions = {}
        self.calculate_key_positions()
        
        print("🎯 Starting simple calibration...")
        print("Look at each red dot and press SPACE")
        
    def calculate_key_positions(self):
        """Calculate key positions"""
        self.key_positions = {}
        
        for row_idx, row in enumerate(self.keys):
            total_width = sum(self.get_key_width(key) + self.key_margin for key in row) - self.key_margin
            start_x = (self.cam_width - total_width) // 2
            
            row_y = self.keyboard_start_y + row_idx * (self.key_height + self.key_margin)
            
            current_x = start_x
            for key in row:
                key_width = self.get_key_width(key)
                
                self.key_positions[key] = {
                    'x1': current_x,
                    'y1': row_y,
                    'x2': current_x + key_width,
                    'y2': row_y + self.key_height,
                }
                
                current_x += key_width + self.key_margin
    
    def get_key_width(self, key):
        """Get width for specific keys"""
        if key == 'SPACE':
            return self.key_width * 3
        elif key in ['DEL', 'CLEAR']:
            return self.key_width * 2
        else:
            return self.key_width
    
    def get_simple_gaze_direction(self, landmarks):
        """Simple gaze direction - just use iris position"""
        try:
            # Get iris centers
            left_iris_points = []
            right_iris_points = []
            
            for idx in self.LEFT_IRIS:
                if idx < len(landmarks):
                    point = landmarks[idx]
                    left_iris_points.append([point.x, point.y])
            
            for idx in self.RIGHT_IRIS:
                if idx < len(landmarks):
                    point = landmarks[idx]
                    right_iris_points.append([point.x, point.y])
            
            if not left_iris_points or not right_iris_points:
                return None
            
            # Simple average
            left_iris = np.mean(left_iris_points, axis=0)
            right_iris = np.mean(right_iris_points, axis=0)
            
            # Average both eyes
            avg_iris = (left_iris + right_iris) / 2
            
            return avg_iris
            
        except Exception as e:
            print(f"Gaze error: {e}")
            return None
    
    def simple_calibration(self, gaze_raw, screen_point):
        """Simple calibration - just store points"""
        if gaze_raw is not None:
            self.calibration_data.append({
                'gaze': gaze_raw,
                'screen': screen_point
            })
            print(f"✓ Calibrated point {len(self.calibration_data)}/5")
            
            if len(self.calibration_data) >= 5:
                self.finish_simple_calibration()
    
    def finish_simple_calibration(self):
        """Finish calibration with simple linear mapping"""
        if len(self.calibration_data) >= 5:
            # Extract gaze and screen points
            gaze_points = np.array([d['gaze'] for d in self.calibration_data])
            screen_points = np.array([d['screen'] for d in self.calibration_data])
            
            # Simple linear scaling
            gaze_min_x, gaze_max_x = np.min(gaze_points[:, 0]), np.max(gaze_points[:, 0])
            gaze_min_y, gaze_max_y = np.min(gaze_points[:, 1]), np.max(gaze_points[:, 1])
            
            screen_min_x, screen_max_x = np.min(screen_points[:, 0]), np.max(screen_points[:, 0])
            screen_min_y, screen_max_y = np.min(screen_points[:, 1]), np.max(screen_points[:, 1])
            
            # Calculate scaling factors
            if gaze_max_x != gaze_min_x:
                self.gaze_scale_x = (screen_max_x - screen_min_x) / (gaze_max_x - gaze_min_x)
                self.gaze_offset_x = screen_min_x - gaze_min_x * self.gaze_scale_x
            
            if gaze_max_y != gaze_min_y:
                self.gaze_scale_y = (screen_max_y - screen_min_y) / (gaze_max_y - gaze_min_y)
                self.gaze_offset_y = screen_min_y - gaze_min_y * self.gaze_scale_y
            
            self.calibration_mode = False
            print("🎉 Simple calibration complete!")
            print("You can now use the eye tracker")
    
    def map_gaze_simple(self, gaze_raw):
        """Simple gaze mapping"""
        if gaze_raw is None:
            return None
        
        # Apply simple linear transformation
        screen_x = gaze_raw[0] * self.gaze_scale_x + self.gaze_offset_x
        screen_y = gaze_raw[1] * self.gaze_scale_y + self.gaze_offset_y
        
        # Clamp to screen
        screen_x = max(0, min(self.cam_width - 1, int(screen_x)))
        screen_y = max(0, min(self.cam_height - 1, int(screen_y)))
        
        return (screen_x, screen_y)
    
    def simple_smooth(self, gaze_point):
        """Very simple smoothing"""
        if gaze_point is None:
            return None
        
        self.gaze_buffer.append(gaze_point)
        
        if len(self.gaze_buffer) < 2:
            return gaze_point
        
        # Simple average of last few points
        avg_x = sum(p[0] for p in self.gaze_buffer) / len(self.gaze_buffer)
        avg_y = sum(p[1] for p in self.gaze_buffer) / len(self.gaze_buffer)
        
        return (int(avg_x), int(avg_y))
    
    def get_key_at_position(self, x, y):
        """Get key at position"""
        for key, pos in self.key_positions.items():
            if pos['x1'] <= x <= pos['x2'] and pos['y1'] <= y <= pos['y2']:
                return key
        return None
    
    def draw_calibration_screen(self, frame):
        """Simple calibration screen"""
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.cam_width, self.cam_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Instructions
        cv2.putText(frame, "SIMPLE CALIBRATION", (self.cam_width//2 - 150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        cv2.putText(frame, "Look at the RED DOT and press SPACE", (self.cam_width//2 - 200, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Point {self.current_calibration_point + 1} of 5", 
                   (self.cam_width//2 - 80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw current calibration point
        if self.current_calibration_point < len(self.calibration_points):
            point = self.calibration_points[self.current_calibration_point]
            cv2.circle(frame, point, 25, (0, 0, 255), -1)
            cv2.circle(frame, point, 30, (255, 255, 255), 3)
    
    def draw_keyboard(self, frame):
        """Draw keyboard"""
        overlay = frame.copy()
        
        # Keyboard background
        kb_x1 = min(pos['x1'] for pos in self.key_positions.values()) - 10
        kb_y1 = min(pos['y1'] for pos in self.key_positions.values()) - 10
        kb_x2 = max(pos['x2'] for pos in self.key_positions.values()) + 10
        kb_y2 = max(pos['y2'] for pos in self.key_positions.values()) + 10
        
        cv2.rectangle(overlay, (kb_x1, kb_y1), (kb_x2, kb_y2), (20, 20, 20), -1)
        
        # Draw keys
        for key, pos in self.key_positions.items():
            x1, y1, x2, y2 = pos['x1'], pos['y1'], pos['x2'], pos['y2']
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 60), -1)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (100, 100, 100), 2)
            
            font_scale = 0.8 if len(key) == 1 else 0.6
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            
            cv2.putText(overlay, key, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    def draw_gaze_indicator(self, frame, gaze_point):
        """Draw gaze indicator"""
        if gaze_point is None:
            return
            
        x, y = gaze_point
        
        # Gaze point
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 15, (0, 255, 255), 2)
        
        # Dwell progress
        if self.current_key:
            current_time = time.time()
            elapsed = current_time - self.dwell_start_time
            progress = min(elapsed / self.dwell_time, 1.0)
            
            radius = 30
            cv2.circle(frame, (x, y), radius, (50, 50, 50), 3)
            
            if progress > 0:
                angle = int(360 * progress)
                cv2.ellipse(frame, (x, y), (radius, radius), 0, -90, -90 + angle, (0, 255, 0), 4)
            
            cv2.putText(frame, f"{int(progress * 100)}%", (x - 15, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def highlight_key(self, frame, key):
        """Highlight key"""
        if key not in self.key_positions:
            return
            
        pos = self.key_positions[key]
        x1, y1, x2, y2 = pos['x1'], pos['y1'], pos['x2'], pos['y2']
        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 4)
    
    def draw_ui(self, frame, gaze_point):
        """Draw UI"""
        # Text area
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.cam_width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        display_text = self.typed_text
        if len(display_text) > 50:
            display_text = "..." + display_text[-47:]
        
        cv2.putText(frame, f"Text: {display_text}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status
        status_y = self.cam_height - 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, status_y - 15), (self.cam_width, self.cam_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        if gaze_point:
            cv2.putText(frame, f"Gaze: ({gaze_point[0]}, {gaze_point[1]})", (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Gaze: Not detected", (20, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if self.current_key:
            current_time = time.time()
            elapsed = current_time - self.dwell_start_time
            progress = min(elapsed / self.dwell_time, 1.0)
            cv2.putText(frame, f"Looking at: {self.current_key} ({progress:.0%})", (300, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame, "Q=Quit | R=Recalibrate", (self.cam_width - 200, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def process_key_press(self, key):
        """Process key press"""
        if key == "SPACE":
            self.typed_text += " "
        elif key == "DEL":
            self.typed_text = self.typed_text[:-1]
        elif key == "CLEAR":
            self.typed_text = ""
        else:
            self.typed_text += key
        
        print(f"✓ Key pressed: {key}")
        print(f"Text: '{self.typed_text}'")
    
    def run(self):
        """Main loop - simplified"""
        print("👁️ Working Eye Tracking Keyboard")
        print("🎯 Simple 5-point calibration")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            gaze_point = None
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get raw gaze direction
                    gaze_raw = self.get_simple_gaze_direction(face_landmarks.landmark)
                    
                    if gaze_raw is not None:
                        if not self.calibration_mode:
                            # Map to screen coordinates
                            mapped_gaze = self.map_gaze_simple(gaze_raw)
                            gaze_point = self.simple_smooth(mapped_gaze)
                        else:
                            # Show raw gaze during calibration for debugging
                            raw_screen_x = int(gaze_raw[0] * self.cam_width)
                            raw_screen_y = int(gaze_raw[1] * self.cam_height)
                            cv2.circle(frame, (raw_screen_x, raw_screen_y), 5, (255, 0, 0), -1)
            
            if self.calibration_mode:
                self.draw_calibration_screen(frame)
            else:
                self.draw_keyboard(frame)
                
                if gaze_point:
                    looked_key = self.get_key_at_position(gaze_point[0], gaze_point[1])
                    
                    if looked_key:
                        self.highlight_key(frame, looked_key)
                        
                        if looked_key == self.current_key:
                            current_time = time.time()
                            if current_time - self.dwell_start_time >= self.dwell_time:
                                self.process_key_press(looked_key)
                                self.current_key = None
                        else:
                            self.current_key = looked_key
                            self.dwell_start_time = time.time()
                    else:
                        self.current_key = None
                
                self.draw_gaze_indicator(frame, gaze_point)
                self.draw_ui(frame, gaze_point)
            
            cv2.imshow("Working Eye Tracking Keyboard", frame)
            
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break
            elif key_pressed == ord(' ') and self.calibration_mode:
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        gaze_raw = self.get_simple_gaze_direction(face_landmarks.landmark)
                        if gaze_raw is not None and self.current_calibration_point < len(self.calibration_points):
                            point = self.calibration_points[self.current_calibration_point]
                            self.simple_calibration(gaze_raw, point)
                            self.current_calibration_point += 1
                            break
            elif key_pressed == ord('r'):
                # Restart calibration
                self.calibration_mode = True
                self.current_calibration_point = 0
                self.calibration_data = []
                print("🔄 Restarting calibration...")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = WorkingEyeTracker()
    tracker.run()