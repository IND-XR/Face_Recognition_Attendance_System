import cv2
import os
import time

def list_image_files(directory, extensions=(".jpg", ".jpeg", ".png")):
    """
    List all image files in a directory with given extensions.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(extensions)]

def capture_student_image(course, reg_number):
    """
    Automatically capture 15 student images with 0.5 second intervals.
    
    Args:
        course (str): Course name
        reg_number (str): Student registration number
    
    Returns:
        tuple: (success, message, image_paths)
    """
    try:
        # Create directory path under static folder
        base_path = os.path.join("static", "face_data")
        folder_path = os.path.join(base_path, course, reg_number)
        
        # Create folders if they don't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Try different camera indices
        cap = None
        for camera_index in [1]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Using camera {camera_index}")
                break
            cap.release()
            
        if cap is None or not cap.isOpened():
            return False, "Failed to access camera.", None
        
        directions = [
            "Look Straight",
            "Turn Face LEFT",
            "Turn Face RIGHT",
            "Turn Face UP",
            "Turn Face DOWN"
        ]
        images_per_direction = 5
        total_images = len(directions) * images_per_direction

        image_paths = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Show camera window and wait for Enter key
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "Press Enter to start capturing images...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Capture Images", frame)
            cv2.setWindowProperty("Capture Images", cv2.WND_PROP_TOPMOST, 1)
            key = cv2.waitKey(1)
            # 13 is Enter key for Windows
            if key == 13:
                break
        # Clear the window before starting capture
        cv2.destroyWindow("Capture Images")

        for direction in directions:
            # Countdown before capturing for this direction
            countdown_seconds = 3
            for i in range(countdown_seconds, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                # Draw face rectangle for display only
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    break
                cv2.putText(display_frame, direction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Starting in {i}...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture Images", display_frame)
                cv2.setWindowProperty("Capture Images", cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1000)  # Wait 1 second

            # Capture images for this direction
            for img_num in range(images_per_direction):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                # Draw face rectangle for display only
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    break
                cv2.putText(display_frame, direction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Capturing image {img_num + 1}/{images_per_direction}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture Images", display_frame)
                cv2.setWindowProperty("Capture Images", cv2.WND_PROP_TOPMOST, 1)
                # Save image (without overlays)
                image_path = os.path.join(folder_path, f"{reg_number}_{len(image_paths) + 1}.jpg")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(image_path, rgb_frame)
                image_paths.append(image_path)
                cv2.waitKey(1000)  # Wait 1 second between captures
        
        # Release camera and close window
        cap.release()
        cv2.destroyAllWindows()
        
        if len(image_paths) == total_images:
            return True, f"{total_images} images saved successfully to {folder_path}", image_paths
        else:
            return False, f"Capture incomplete. Only {len(image_paths)} images captured.", image_paths
        
    except Exception as e:
        # Ensure cleanup in case of error
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        return False, f"Error: {str(e)}", None

def validate_saved_images(image_paths_or_dir):
    """
    Validate that saved images can be loaded correctly. Accepts a list of image paths or a directory.
    """
    # If a directory is passed, list image files
    if isinstance(image_paths_or_dir, str) and os.path.isdir(image_paths_or_dir):
        image_paths = list_image_files(image_paths_or_dir)
    else:
        image_paths = image_paths_or_dir
    
    if not image_paths:
        return False, "No images to validate"
    
    valid_count = 0
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None and len(img.shape) == 3:
                valid_count += 1
                print(f"✓ {img_path} - Valid (Shape: {img.shape})")
            else:
                print(f"✗ {img_path} - Invalid or corrupted")
        except Exception as e:
            print(f"✗ {img_path} - Error: {str(e)}")
    
    return valid_count == len(image_paths), f"{valid_count}/{len(image_paths)} images are valid"

# Example usage (for testing):
if __name__ == "__main__":
    course = "cs"  # Match your folder structure
    reg_number = "CSS001"  # Match your existing folder
    
    print(f"Starting image capture for {course}/{reg_number}")
    success, message, image_paths = capture_student_image(course, reg_number)
    print(f"Capture result: {message}")
    
    # Validate using directory or image_paths
    print("\nValidating saved images...")
    valid, validation_msg = validate_saved_images(os.path.join('static', 'face_data', course, reg_number))
    print(f"Validation result: {validation_msg}")
    
    if not valid:
        print("Some images may be corrupted. Consider recapturing them.")