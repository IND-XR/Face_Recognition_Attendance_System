print("DEBUG: Attendance.py script started")
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import ctypes
import time
import platform
import sys
import csv
# Assuming you have a db_connection.py similar to app.py or define the connection here
# For now, I'll add placeholders for database interaction. You'll need to integrate
# your actual database connection logic here.
import MySQLdb # Import MySQLdb
from db_connection import get_db_connection # Import the actual connection function

# --- Placeholder for Database Connection (REPLACE WITH YOUR ACTUAL DB CONNECTION) ---
# REMOVED PLACEHOLDER
# --- End Placeholder ---


# Configuration
TOLERANCE = 0.5  # Lowered from 0.6 to be more strict (0.5 is a good balance between accuracy and flexibility)
FRAME_SCALE = 0.5  # Increased from 0.25 to get better quality for detection
MIN_FACE_SIZE = 80  # Lowered from 100 to detect faces at slightly further distances
MIN_FACE_CONFIDENCE = 0.6  # Minimum confidence threshold for face matches
ATTENDANCE_DIR = 'Attendance'  # Root directory for attendance CSV files
IMAGES_PATH = 'static/face_data'  # Root folder containing subfolders for each person

# Get course, unit, and venue from command line arguments
if len(sys.argv) < 4:
    print("Error: Please provide course code, unit code, and venue ID as arguments")
    print("Usage: python Attendance.py <course_code> <unit_code> <venue_id>")
    sys.exit(1)

selected_course = sys.argv[1].upper()  # Convert to uppercase to match folder names
selected_unit = sys.argv[2] # Get unit code
selected_venue_id = int(sys.argv[3]) # Get venue ID

print(f"\nStarting face recognition for course: {selected_course}, Unit: {selected_unit}, Venue ID: {selected_venue_id}")

# Initialize lists
known_face_encodings = []
known_face_names = []
processed_students = set()
attended_students_session = set() # Keep track of students marked present in this session

# Check if the course folder exists
course_path = os.path.join(IMAGES_PATH, selected_course)
if not os.path.isdir(course_path):
    print(f"Error: Course folder '{selected_course}' not found in {IMAGES_PATH}")
    print(f"Available course folders: {[d for d in os.listdir(IMAGES_PATH) if os.path.isdir(os.path.join(IMAGES_PATH, d))]}")
    sys.exit(1)

print(f"\nProcessing students in course folder: {course_path}")

print("Please wait, loading and encoding student images. This may take a moment...")

# Process only the selected course folder
for reg_folder in os.listdir(course_path):
    print(f"Encoding images for {reg_folder}...", flush=True)
    reg_path = os.path.join(course_path, reg_folder)
    if not os.path.isdir(reg_path):
        continue

    print(f"\nProcessing student: {reg_folder}")
    student_images = [f for f in os.listdir(reg_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not student_images:
        print(f"  Warning: No images found for {reg_folder}")
        continue

    print(f"  Found {len(student_images)} images")
    student_encodings = []

    for image_file in student_images:
        image_path = os.path.join(reg_path, image_file)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Warning: Could not load {image_file}")
            continue

        # Convert to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Remove alpha channel if present
        if rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[:, :, :3]

        # Ensure dtype is uint8
        rgb_image = rgb_image.astype('uint8')

        # Double-check shape and dtype
        if rgb_image.dtype != np.uint8 or len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            print(f"  Error: {image_file} is not a valid 8-bit RGB image. Skipping.")
            continue

        # Find face encodings
        face_encodings = face_recognition.face_encodings(rgb_image)

        if len(face_encodings) > 0:
            student_encodings.append(face_encodings[0])
            print(f"  Added encoding from {image_file}", flush=True)
        else:
            print(f"  Warning: No face found in {image_file}")

    if student_encodings:
        # Use the average encoding for this student
        avg_encoding = np.mean(student_encodings, axis=0)
        known_face_encodings.append(avg_encoding)
        known_face_names.append(reg_folder)
        processed_students.add(reg_folder)
        print(f"  Successfully processed {reg_folder} with {len(student_encodings)} valid encodings")
    print(f"Finished encoding for {reg_folder}", flush=True)

if not known_face_encodings:
    print(f"\nError: No valid face encodings found for course {selected_course}")
    sys.exit(1)

print("Encoding complete:", flush=True)
print(f"- Course: {selected_course}", flush=True)
print(f"- Total students processed: {len(processed_students)}", flush=True)
print(f"- Total encodings loaded: {len(known_face_encodings)}", flush=True)
print(f"- Students with valid encodings: {', '.join(processed_students)}", flush=True)

print("ENCODING_COMPLETE", flush=True)

def mark_attendance(reg_number, unit_code, venue_id):
    """Mark attendance in the database if not already recorded for this session"""
    if reg_number in attended_students_session:
        # print(f"Attendance already marked for {reg_number} in this session.")
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
             print("Database connection failed in mark_attendance.")
             return
        
        cursor = conn.cursor()

        # Check if attendance is already marked for this student, unit, venue, and today
        # Note: You might want a more sophisticated check depending on your requirements
        # This example checks if the student is marked present for this unit/venue today
        query = """
        SELECT COUNT(*) 
        FROM tblattendance 
        WHERE studentRegistrationNumber = %s 
          AND unit = %s 
          AND venueId = %s 
          AND DATE(dateMarked) = CURDATE()
        """
        cursor.execute(query, (reg_number, unit_code, venue_id))
        result = cursor.fetchone()
        
        if result and result[0] > 0:
            # print(f"Attendance already exists for {reg_number} for {unit_code} at {venue_id} today.")
            attended_students_session.add(reg_number) # Add to session set even if already in DB
            return

        # If not already marked, insert new attendance record
        insert_query = """
        INSERT INTO tblattendance (studentRegistrationNumber, course, attendanceStatus, dateMarked, unit, venueId)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        # Assuming 'Present' status. Adjust if your schema uses different values.
        cursor.execute(insert_query, (reg_number, selected_course, 'Present', datetime.now(), unit_code, venue_id)) 
        conn.commit()

        attended_students_session.add(reg_number)
        print(f"✅ Attendance marked in DB for {reg_number} for {unit_code} at {venue_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"Error marking attendance in DB for {reg_number}: {e}")
        if conn and conn.is_connected():
             conn.rollback() # Roll back the transaction on error
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
             # Note: Keep the connection open if Attendance.py runs for a long time,
             # or close it and reopen for each mark if performance is not critical.
             # Closing it here for simplicity in this example.
             conn.close()

def bring_window_to_front(window_name):
    """Bring OpenCV window to front - works on Windows"""
    if platform.system() == "Windows":
        try:
            # More comprehensive approach to bring window to front
            hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
            if hwnd != 0:
                # Get current foreground window
                current_hwnd = ctypes.windll.user32.GetForegroundWindow()
                current_thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
                target_thread_id = ctypes.windll.user32.GetWindowThreadProcessId(current_hwnd, None)

                # Attach to the target thread's input
                ctypes.windll.user32.AttachThreadInput(current_thread_id, target_thread_id, True)

                # Bring window to front
                ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_TOPMOST
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                ctypes.windll.user32.BringWindowToTop(hwnd)
                ctypes.windll.user32.SetFocus(hwnd)

                # Remove topmost flag (optional - keeps it always on top if you comment this out)
                ctypes.windll.user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0001 | 0x0002)  # HWND_NOTOPMOST

                # Detach from target thread
                ctypes.windll.user32.AttachThreadInput(current_thread_id, target_thread_id, False)

                return True
        except Exception as e:
            print(f"Warning: Could not bring window to front: {e}")
            return False
    return False


# Initialize video capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Create named window with specific properties
window_name = 'Face Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# Flag to track if window has been brought to front
window_focused = False

while True:
    # Grab single frame
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Mirror the frame for more natural interaction
    frame = cv2.flip(frame, 1)

    # Resize and convert to RGB for processing
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Filter out small faces
        face_height = bottom - top
        face_width = right - left
        if face_height < MIN_FACE_SIZE * FRAME_SCALE or face_width < MIN_FACE_SIZE * FRAME_SCALE:
            continue

        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find best match
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        confidence = 1 - face_distances[best_match_index]

        # More strict matching criteria
        if matches[best_match_index] and confidence > MIN_FACE_CONFIDENCE:
            # Additional validation: check if this is significantly better than second best match
            if len(face_distances) > 1:
                sorted_distances = np.sort(face_distances)
                if sorted_distances[1] - sorted_distances[0] < 0.1:  # If second best match is too close
                    name = "Unknown"
                else:
                    name = known_face_names[best_match_index]
                    # Mark attendance in the database
                    mark_attendance(name, selected_unit, selected_venue_id)
            else:
                name = known_face_names[best_match_index]
                # Mark attendance in the database
                mark_attendance(name, selected_unit, selected_venue_id)

        # Scale coordinates back to original frame size
        top *= int(1 / FRAME_SCALE)
        right *= int(1 / FRAME_SCALE)
        bottom *= int(1 / FRAME_SCALE)
        left *= int(1 / FRAME_SCALE)

        # Draw rectangle and label with more detailed information
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # More detailed label showing confidence and match quality
        if name != "Unknown":
            label = f"{name} ({confidence:.2f})"
            if confidence > 0.8:
                label += " ✓"  # Add checkmark for high confidence matches
        else:
            label = "Unknown Face"
        cv2.putText(frame, label, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # Display result
    cv2.imshow(window_name, frame)

    # Bring window to front (only once at startup)
    if not window_focused:
        time.sleep(0.5)  # Give time for window to be created
        if bring_window_to_front(window_name):
            window_focused = True
           # print("Window brought to front successfully")
        else:
            # Fallback method for other systems or if the first method fails
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            time.sleep(0.1)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)  # Remove always-on-top
            window_focused = True

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()