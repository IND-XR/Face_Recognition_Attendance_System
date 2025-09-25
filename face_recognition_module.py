"""
Standalone Face Recognition Module for Attendance System
This module provides face recognition capabilities that can be integrated
with any database backend (Supabase, MySQL, PostgreSQL, etc.)

Dependencies:
- opencv-python (cv2)
- numpy
- face-recognition
- Pillow (optional, for image manipulation)

Author: Face Recognition Module Extractor
"""

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable
import json
import base64
from io import BytesIO
from PIL import Image

class FaceRecognitionAttendance:
    """
    A standalone face recognition system for attendance marking.
    This class can be integrated with any database backend.
    """
    
    def __init__(self, 
                 images_base_path: str = 'face_data',
                 tolerance: float = 0.5,
                 frame_scale: float = 0.5,
                 min_face_size: int = 80,
                 min_face_confidence: float = 0.6):
        """
        Initialize the Face Recognition Attendance system.
        
        Args:
            images_base_path: Root folder containing face images
            tolerance: Face matching tolerance (lower = stricter)
            frame_scale: Scale factor for video frame processing
            min_face_size: Minimum face size to detect
            min_face_confidence: Minimum confidence threshold for face matches
        """
        self.images_base_path = images_base_path
        self.tolerance = tolerance
        self.frame_scale = frame_scale
        self.min_face_size = min_face_size
        self.min_face_confidence = min_face_confidence
        
        # Storage for face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_metadata = {}  # Store additional metadata per person
        
        # Session tracking
        self.attended_session = set()
        
        # Camera settings
        self.camera_index = None
        self.cap = None
        
        # Callback functions
        self.on_attendance_marked = None
        self.on_face_detected = None
        self.on_unknown_face = None
        
    def set_callbacks(self, 
                      on_attendance_marked: Optional[Callable] = None,
                      on_face_detected: Optional[Callable] = None,
                      on_unknown_face: Optional[Callable] = None):
        """
        Set callback functions for various events.
        
        Args:
            on_attendance_marked: Called when attendance is marked
                                 Signature: (student_id, metadata, timestamp)
            on_face_detected: Called when a face is detected
                            Signature: (student_id, confidence, frame)
            on_unknown_face: Called when an unknown face is detected
                           Signature: (face_encoding, frame)
        """
        self.on_attendance_marked = on_attendance_marked
        self.on_face_detected = on_face_detected
        self.on_unknown_face = on_unknown_face
        
    def load_faces_for_group(self, group_identifier: str) -> Dict:
        """
        Load face encodings for a specific group (course/class/department).
        
        Args:
            group_identifier: Identifier for the group (e.g., course code)
            
        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'total_students': 0,
            'successful_encodings': 0,
            'failed_encodings': 0,
            'students': []
        }
        
        group_path = os.path.join(self.images_base_path, group_identifier)
        
        if not os.path.isdir(group_path):
            raise ValueError(f"Group folder '{group_identifier}' not found in {self.images_base_path}")
        
        print(f"Loading faces for group: {group_identifier}")
        
        for student_folder in os.listdir(group_path):
            student_path = os.path.join(group_path, student_folder)
            if not os.path.isdir(student_path):
                continue
                
            stats['total_students'] += 1
            
            # Load images for this student
            encodings = self._load_student_encodings(student_path, student_folder)
            
            if encodings:
                # Use average encoding for better accuracy
                avg_encoding = np.mean(encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(student_folder)
                
                # Store metadata
                self.face_metadata[student_folder] = {
                    'group': group_identifier,
                    'encoding_count': len(encodings),
                    'path': student_path
                }
                
                stats['successful_encodings'] += 1
                stats['students'].append(student_folder)
            else:
                stats['failed_encodings'] += 1
                print(f"  Warning: No valid encodings for {student_folder}")
                
        return stats
    
    def _load_student_encodings(self, student_path: str, student_id: str) -> List:
        """
        Load face encodings for a specific student.
        
        Args:
            student_path: Path to student's image folder
            student_id: Student identifier
            
        Returns:
            List of face encodings
        """
        encodings = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for image_file in os.listdir(student_path):
            if not image_file.lower().endswith(image_extensions):
                continue
                
            image_path = os.path.join(student_path, image_file)
            
            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                    
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face encodings
                face_encodings = face_recognition.face_encodings(rgb_image)
                
                if face_encodings:
                    encodings.append(face_encodings[0])
                    
            except Exception as e:
                print(f"  Error processing {image_file}: {e}")
                
        return encodings
    
    def capture_student_images(self, 
                              student_id: str,
                              group_identifier: str,
                              num_images: int = 25,
                              camera_index: int = 0) -> Tuple[bool, str, List[str]]:
        """
        Capture multiple images of a student for training.
        
        Args:
            student_id: Student identifier (registration number)
            group_identifier: Group identifier (e.g., course code)
            num_images: Number of images to capture
            camera_index: Camera device index
            
        Returns:
            Tuple of (success, message, image_paths)
        """
        try:
            # Create directory structure
            folder_path = os.path.join(self.images_base_path, group_identifier, student_id)
            os.makedirs(folder_path, exist_ok=True)
            
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return False, "Failed to access camera", []
            
            image_paths = []
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            # Capture instructions
            directions = [
                "Look Straight",
                "Turn Face LEFT",
                "Turn Face RIGHT",
                "Turn Face UP",
                "Turn Face DOWN"
            ]
            images_per_direction = num_images // len(directions)
            
            window_name = f"Capture Images - {student_id}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "Press ENTER to start capturing...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) == 13:  # Enter key
                    break
                    
            # Capture images for each direction
            for direction in directions:
                # Show direction instruction
                for i in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame = cv2.flip(frame, 1)
                    
                    # Detect and draw face
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                    cv2.putText(frame, direction, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Starting in {i}...", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                    cv2.waitKey(1000)
                    
                # Capture images
                for img_num in range(images_per_direction):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame = cv2.flip(frame, 1)
                    display_frame = frame.copy()
                    
                    # Draw face rectangle for display
                    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        
                    cv2.putText(display_frame, direction, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Image {img_num + 1}/{images_per_direction}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(window_name, display_frame)
                    
                    # Save image
                    image_path = os.path.join(folder_path, 
                                             f"{student_id}_{len(image_paths) + 1}.jpg")
                    cv2.imwrite(image_path, frame)
                    image_paths.append(image_path)
                    
                    cv2.waitKey(500)  # Short delay between captures
                    
            cap.release()
            cv2.destroyWindow(window_name)
            
            return True, f"Captured {len(image_paths)} images successfully", image_paths
            
        except Exception as e:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            return False, f"Error: {str(e)}", []
    
    def start_recognition(self, 
                         camera_index: int = 0,
                         window_name: str = "Face Recognition Attendance",
                         show_window: bool = True) -> None:
        """
        Start the face recognition process.
        
        Args:
            camera_index: Camera device index
            window_name: Name for the display window
            show_window: Whether to show the video window
        """
        if not self.known_face_encodings:
            raise ValueError("No face encodings loaded. Load faces first.")
            
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
        print(f"Starting face recognition with {len(self.known_face_encodings)} known faces")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                    
                # Process frame
                processed_frame, detections = self.process_frame(frame)
                
                # Handle detections
                for detection in detections:
                    self._handle_detection(detection)
                    
                if show_window:
                    cv2.imshow(window_name, processed_frame)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            self.stop_recognition()
            if show_window:
                cv2.destroyWindow(window_name)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame for face recognition.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (annotated frame, list of detections)
        """
        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), 
                                 fx=self.frame_scale, 
                                 fy=self.frame_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detections = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Filter small faces
            face_height = bottom - top
            face_width = right - left
            if (face_height < self.min_face_size * self.frame_scale or 
                face_width < self.min_face_size * self.frame_scale):
                continue
                
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            # Find best match
            name = "Unknown"
            confidence = 0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > self.min_face_confidence:
                    # Additional validation
                    if len(face_distances) > 1:
                        sorted_distances = np.sort(face_distances)
                        if sorted_distances[1] - sorted_distances[0] >= 0.1:
                            name = self.known_face_names[best_match_index]
                    else:
                        name = self.known_face_names[best_match_index]
                        
            # Scale coordinates back
            scale = int(1 / self.frame_scale)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            # Draw on frame
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Add label
            if name != "Unknown":
                label = f"{name} ({confidence:.2f})"
                if confidence > 0.8:
                    label += " ✓"
            else:
                label = "Unknown Face"
                
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add to detections
            detections.append({
                'name': name,
                'confidence': confidence,
                'location': (top, right, bottom, left),
                'encoding': face_encoding,
                'timestamp': datetime.now()
            })
            
        return frame, detections
    
    def _handle_detection(self, detection: Dict) -> None:
        """
        Handle a face detection event.
        
        Args:
            detection: Detection dictionary
        """
        name = detection['name']
        confidence = detection['confidence']
        
        if name != "Unknown":
            # Check if not already marked in this session
            if name not in self.attended_session:
                self.attended_session.add(name)
                
                # Get metadata
                metadata = self.face_metadata.get(name, {})
                
                # Call attendance callback
                if self.on_attendance_marked:
                    self.on_attendance_marked(name, metadata, detection['timestamp'])
                    
                print(f"✓ Attendance marked: {name} at {detection['timestamp']}")
                
            # Call face detected callback
            if self.on_face_detected:
                self.on_face_detected(name, confidence, None)
        else:
            # Call unknown face callback
            if self.on_unknown_face:
                self.on_unknown_face(detection['encoding'], None)
    
    def stop_recognition(self) -> None:
        """Stop the face recognition process."""
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def clear_session(self) -> None:
        """Clear the current session data."""
        self.attended_session.clear()
        
    def get_session_attendance(self) -> List[str]:
        """Get list of students marked present in current session."""
        return list(self.attended_session)
    
    def export_encodings(self, filepath: str) -> None:
        """
        Export face encodings to a file for backup or transfer.
        
        Args:
            filepath: Path to save the encodings
        """
        data = {
            'encodings': [enc.tolist() for enc in self.known_face_encodings],
            'names': self.known_face_names,
            'metadata': self.face_metadata,
            'config': {
                'tolerance': self.tolerance,
                'frame_scale': self.frame_scale,
                'min_face_size': self.min_face_size,
                'min_face_confidence': self.min_face_confidence
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_encodings(self, filepath: str) -> None:
        """
        Import face encodings from a file.
        
        Args:
            filepath: Path to the encodings file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.known_face_encodings = [np.array(enc) for enc in data['encodings']]
        self.known_face_names = data['names']
        self.face_metadata = data.get('metadata', {})
        
        # Update config if present
        if 'config' in data:
            config = data['config']
            self.tolerance = config.get('tolerance', self.tolerance)
            self.frame_scale = config.get('frame_scale', self.frame_scale)
            self.min_face_size = config.get('min_face_size', self.min_face_size)
            self.min_face_confidence = config.get('min_face_confidence', self.min_face_confidence)


# Example usage with database integration
class DatabaseIntegration:
    """
    Example of how to integrate the face recognition module with a database.
    This can be adapted for Supabase, MySQL, PostgreSQL, etc.
    """
    
    def __init__(self, face_recognition_system: FaceRecognitionAttendance):
        self.fr_system = face_recognition_system
        
        # Set up callbacks
        self.fr_system.set_callbacks(
            on_attendance_marked=self.mark_attendance_in_db,
            on_face_detected=self.log_face_detection,
            on_unknown_face=self.handle_unknown_face
        )
        
    def mark_attendance_in_db(self, student_id: str, metadata: Dict, timestamp: datetime):
        """
        Mark attendance in your database.
        Replace this with your actual database logic (Supabase, MySQL, etc.)
        """
        # Example for Supabase (pseudocode):
        # supabase.table('attendance').insert({
        #     'student_id': student_id,
        #     'course_id': metadata.get('group'),
        #     'timestamp': timestamp.isoformat(),
        #     'status': 'present'
        # }).execute()
        
        print(f"DB: Marking attendance for {student_id} at {timestamp}")
        
    def log_face_detection(self, student_id: str, confidence: float, frame):
        """Log face detection event."""
        print(f"Face detected: {student_id} (confidence: {confidence:.2f})")
        
    def handle_unknown_face(self, face_encoding, frame):
        """Handle unknown face detection."""
        print("Unknown face detected")
        # Could save unknown face for later review
        

def main():
    """Example of using the face recognition module."""
    
    # Initialize the face recognition system
    fr_system = FaceRecognitionAttendance(
        images_base_path='static/face_data',
        tolerance=0.5,
        frame_scale=0.5,
        min_face_size=80,
        min_face_confidence=0.6
    )
    
    # Load faces for a specific group (course)
    try:
        stats = fr_system.load_faces_for_group('CS101')  # Replace with your course code
        print(f"Loaded {stats['successful_encodings']} faces successfully")
    except ValueError as e:
        print(f"Error loading faces: {e}")
        return
    
    # Set up database integration
    db_integration = DatabaseIntegration(fr_system)
    
    # Start recognition
    try:
        fr_system.start_recognition(camera_index=0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Get attendance list
        attendance = fr_system.get_session_attendance()
        print(f"\nSession attendance: {attendance}")


if __name__ == "__main__":
    main()