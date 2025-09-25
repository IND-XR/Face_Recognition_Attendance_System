"""
Supabase Integration for Face Recognition Attendance System
This example shows how to integrate the face recognition module with your Supabase schema.

Requirements:
- supabase-py: pip install supabase
- python-dotenv: pip install python-dotenv (for environment variables)
- numpy: pip install numpy
- face_recognition_module (the standalone module we created)

Your Supabase schema includes:
- students (with enrollments for course association)
- teachers
- courses (identified by code)
- course_sessions (individual class sessions)
- attendance (marking presence)
- face_embeddings (pgvector for face recognition)
"""

import os
from datetime import datetime, date, time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client
from face_recognition_module import FaceRecognitionAttendance
import json
import numpy as np

# Load environment variables
load_dotenv()

class SupabaseAttendanceSystem:
    """
    Integration of Face Recognition with your Supabase backend schema.
    """
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize the Supabase attendance system.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase project anon/public key
        """
        # Initialize Supabase client
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Key must be provided")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize face recognition system
        self.fr_system = FaceRecognitionAttendance(
            images_base_path='face_data',  # Local storage for face images
            tolerance=0.5,
            frame_scale=0.5,
            min_face_size=80,
            min_face_confidence=0.6
        )
        
        # Set up callbacks
        self.fr_system.set_callbacks(
            on_attendance_marked=self.mark_attendance,
            on_face_detected=self.log_detection,
            on_unknown_face=self.handle_unknown_face
        )
        
        # Current session information
        self.current_session = {
            'session_id': None,
            'course_id': None,
            'teacher_id': None,
            'session_date': None,
            'location': None
        }
        
        # Cache for student IDs (registration_number -> student_id mapping)
        self.student_id_cache = {}
    
    def get_course_by_code(self, course_code: str) -> Optional[Dict]:
        """
        Get course details by course code.
        
        Args:
            course_code: The course code
            
        Returns:
            Course record or None
        """
        try:
            response = self.supabase.table('courses').select("*").eq(
                'code', course_code
            ).execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            print(f"Error fetching course: {e}")
            return None
    
    def get_students_by_course(self, course_code: str) -> List[Dict]:
        """
        Fetch all students enrolled in a specific course.
        
        Args:
            course_code: The course code
            
        Returns:
            List of student records
        """
        try:
            # First get the course
            course = self.get_course_by_code(course_code)
            if not course:
                print(f"Course {course_code} not found")
                return []
            
            # Get students through enrollments table
            enrollments_response = self.supabase.table('enrollments').select(
                "*, students(*)"
            ).eq('course_id', course['id']).eq('is_active', True).execute()
            
            # Extract student records
            students = []
            for enrollment in enrollments_response.data:
                if 'students' in enrollment:
                    student = enrollment['students']
                    # Cache the student ID for quick lookup
                    # Using email as registration number for this example
                    # You might have a separate registration_number field
                    if student.get('email'):
                        self.student_id_cache[student['email']] = student['id']
                    students.append(student)
            
            return students
            
        except Exception as e:
            print(f"Error fetching students: {e}")
            return []
    
    def load_students_for_course(self, course_code: str) -> bool:
        """
        Load face encodings for all students in a course from local storage.
        
        Args:
            course_code: The course code
            
        Returns:
            Success status
        """
        try:
            # Load faces from local storage (face_data/course_code/*)
            stats = self.fr_system.load_faces_for_group(course_code)
            print(f"Loaded {stats['successful_encodings']} faces from local storage")
            
            # Get students from database to build the cache
            students = self.get_students_by_course(course_code)
            print(f"Found {len(students)} students enrolled in {course_code}")
            
            # Optionally load face embeddings from database
            # This would require converting between your 128-d pgvector format
            # and the face_recognition library format
            
            return True
            
        except Exception as e:
            print(f"Error loading students: {e}")
            return False
    
    def start_attendance_session(self, 
                                course_code: str,
                                teacher_id: str,
                                session_date: date = None,
                                start_time: time = None,
                                end_time: time = None,
                                location: str = None) -> Optional[str]:
        """
        Start a new attendance session for a course.
        
        Args:
            course_code: Course code
            teacher_id: Teacher ID who is conducting the session
            session_date: Date of the session (defaults to today)
            start_time: Start time (defaults to now)
            end_time: End time (optional)
            location: Location/room (optional)
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            # Get course
            course = self.get_course_by_code(course_code)
            if not course:
                print(f"Course {course_code} not found")
                return None
            
            # Create a new course session
            session_data = {
                'course_id': course['id'],
                'teacher_id': teacher_id,
                'session_date': (session_date or datetime.now().date()).isoformat(),
                'start_time': (start_time or datetime.now().time()).isoformat() if start_time else datetime.now().time().isoformat(),
                'end_time': end_time.isoformat() if end_time else None,
                'location': location,
                'status': 'active'
            }
            
            response = self.supabase.table('course_sessions').insert(
                session_data
            ).execute()
            
            if response.data:
                session_id = response.data[0]['id']
                
                # Update current session info
                self.current_session = {
                    'session_id': session_id,
                    'course_id': course['id'],
                    'teacher_id': teacher_id,
                    'session_date': session_data['session_date'],
                    'location': location
                }
                
                # Load students for this course
                self.load_students_for_course(course_code)
                
                print(f"Started attendance session: {session_id} for course {course_code}")
                return session_id
                
        except Exception as e:
            print(f"Error starting session: {e}")
            
        return None
    
    def end_attendance_session(self) -> bool:
        """
        End the current attendance session.
        
        Returns:
            Success status
        """
        if not self.current_session['session_id']:
            print("No active session to end")
            return False
            
        try:
            # Update session status
            response = self.supabase.table('course_sessions').update({
                'status': 'completed',
                'end_time': datetime.now().time().isoformat()
            }).eq('id', self.current_session['session_id']).execute()
            
            if response.data:
                print(f"Ended session: {self.current_session['session_id']}")
                
                # Clear current session
                self.current_session = {
                    'session_id': None,
                    'course_id': None,
                    'teacher_id': None,
                    'session_date': None,
                    'location': None
                }
                
                # Clear face recognition session
                self.fr_system.clear_session()
                
                return True
                
        except Exception as e:
            print(f"Error ending session: {e}")
            
        return False
    
    def mark_attendance(self, student_identifier: str, metadata: Dict, timestamp: datetime):
        """
        Callback function to mark attendance in Supabase.
        
        Args:
            student_identifier: Student identifier (registration number or email)
            metadata: Additional metadata about the student
            timestamp: Time of detection
        """
        if not self.current_session['session_id']:
            print("No active session for marking attendance")
            return
            
        try:
            # Get student ID
            student_id = None
            
            # First check cache
            if student_identifier in self.student_id_cache:
                student_id = self.student_id_cache[student_identifier]
            else:
                # Query database - try email first, then other fields
                student_response = self.supabase.table('students').select("id").eq(
                    'email', student_identifier
                ).execute()
                
                if student_response.data:
                    student_id = student_response.data[0]['id']
                    self.student_id_cache[student_identifier] = student_id
            
            if not student_id:
                print(f"Student {student_identifier} not found in database")
                return
            
            # Check if attendance already marked
            existing = self.supabase.table('attendance').select("id").eq(
                'session_id', self.current_session['session_id']
            ).eq('student_id', student_id).execute()
            
            if existing.data:
                print(f"Attendance already marked for {student_identifier}")
                return
            
            # Optionally store face embedding (convert to 128-d for pgvector)
            face_embedding_id = None
            if 'encoding' in metadata and metadata['encoding'] is not None:
                # face_recognition typically uses 128-d encodings
                # Store it in face_embeddings table
                try:
                    embedding_data = {
                        'student_id': student_id,
                        'embedding': metadata['encoding'].tolist() if hasattr(metadata['encoding'], 'tolist') else list(metadata['encoding']),
                        'source': 'attendance_capture',
                        'session_id': self.current_session['session_id']
                    }
                    
                    embedding_response = self.supabase.table('face_embeddings').insert(
                        embedding_data
                    ).execute()
                    
                    if embedding_response.data:
                        face_embedding_id = embedding_response.data[0]['id']
                except Exception as e:
                    print(f"Error storing face embedding: {e}")
            
            # Mark attendance
            attendance_data = {
                'session_id': self.current_session['session_id'],
                'student_id': student_id,
                'is_present': True,
                'marked_at': timestamp.isoformat(),
                'marked_by': self.current_session['teacher_id'],
                'method': 'face_recognition',
                'confidence': metadata.get('confidence', 0),
                'face_embedding_id': face_embedding_id
            }
            
            response = self.supabase.table('attendance').insert(
                attendance_data
            ).execute()
            
            if response.data:
                print(f"✓ Attendance marked for {student_identifier} at {timestamp}")
                
        except Exception as e:
            print(f"Error marking attendance: {e}")
    
    def log_detection(self, student_identifier: str, confidence: float, frame):
        """
        Log face detection event (optional).
        
        Args:
            student_identifier: Student identifier
            confidence: Detection confidence score
            frame: Video frame (not used in this example)
        """
        # This is called frequently, so we'll just print for now
        # You could implement more sophisticated logging if needed
        if confidence > 0.7:  # Only log high-confidence detections
            print(f"Face detected: {student_identifier} (confidence: {confidence:.2f})")
    
    def handle_unknown_face(self, face_encoding, frame):
        """
        Handle unknown face detection.
        
        Args:
            face_encoding: Face encoding of unknown person
            frame: Video frame (not used in this example)
        """
        print("Unknown face detected")
        # You could save unknown faces for later review or training
    
    def get_attendance_report(self, session_id: str = None, course_code: str = None, 
                             date_from: date = None, date_to: date = None) -> List[Dict]:
        """
        Get attendance report based on filters.
        
        Args:
            session_id: Specific session ID
            course_code: Filter by course code
            date_from: Start date
            date_to: End date
            
        Returns:
            List of attendance records
        """
        try:
            query = self.supabase.table('attendance').select(
                "*, students(full_name, email), course_sessions(session_date, courses(code, title))"
            )
            
            if session_id:
                query = query.eq('session_id', session_id)
            
            if course_code:
                # Get course first
                course = self.get_course_by_code(course_code)
                if course:
                    # Filter through course_sessions
                    sessions_response = self.supabase.table('course_sessions').select("id").eq(
                        'course_id', course['id']
                    ).execute()
                    
                    session_ids = [s['id'] for s in sessions_response.data]
                    if session_ids:
                        query = query.in_('session_id', session_ids)
            
            response = query.execute()
            return response.data
            
        except Exception as e:
            print(f"Error getting attendance report: {e}")
            return []
    
    def add_student(self, full_name: str, email: str, 
                   department: str = None, phone: str = None) -> Optional[str]:
        """
        Add a new student to the system.
        
        Args:
            full_name: Student full name
            email: Student email
            department: Department (optional)
            phone: Phone number (optional)
            
        Returns:
            Student ID if successful
        """
        try:
            # Add student
            student_data = {
                'full_name': full_name,
                'email': email,
                'department': department,
                'phone': phone,
                'is_active': True
            }
            
            response = self.supabase.table('students').insert(student_data).execute()
            
            if response.data:
                student_id = response.data[0]['id']
                print(f"Student {full_name} added successfully with ID: {student_id}")
                return student_id
                
        except Exception as e:
            print(f"Error adding student: {e}")
            
        return None
    
    def enroll_student_in_course(self, student_id: str, course_code: str) -> bool:
        """
        Enroll a student in a course.
        
        Args:
            student_id: Student ID
            course_code: Course code
            
        Returns:
            Success status
        """
        try:
            # Get course
            course = self.get_course_by_code(course_code)
            if not course:
                print(f"Course {course_code} not found")
                return False
            
            # Create enrollment
            enrollment_data = {
                'student_id': student_id,
                'course_id': course['id'],
                'is_active': True
            }
            
            response = self.supabase.table('enrollments').insert(enrollment_data).execute()
            
            if response.data:
                print(f"Student enrolled in {course_code}")
                return True
                
        except Exception as e:
            print(f"Error enrolling student: {e}")
            
        return False
    
    def capture_and_save_student_faces(self, student_email: str, course_code: str):
        """
        Capture student face images and save encodings.
        
        Args:
            student_email: Student email (used as identifier)
            course_code: Course code for organizing images
        """
        # Capture images
        success, message, image_paths = self.fr_system.capture_student_images(
            student_id=student_email,  # Using email as identifier
            group_identifier=course_code,
            num_images=25,
            camera_index=0
        )
        
        if success:
            print(message)
            
            # Generate and store initial embeddings
            encodings = self.fr_system._load_student_encodings(
                os.path.join(self.fr_system.images_base_path, course_code, student_email),
                student_email
            )
            
            if encodings:
                # Store average encoding in Supabase
                try:
                    # Get student ID
                    student_response = self.supabase.table('students').select("id").eq(
                        'email', student_email
                    ).execute()
                    
                    if student_response.data:
                        student_id = student_response.data[0]['id']
                        
                        # Store as registration embedding
                        avg_encoding = np.mean(encodings, axis=0)
                        embedding_data = {
                            'student_id': student_id,
                            'embedding': avg_encoding.tolist(),
                            'source': 'registration'
                        }
                        
                        self.supabase.table('face_embeddings').insert(embedding_data).execute()
                        print("Face embeddings saved to Supabase")
                        
                except Exception as e:
                    print(f"Error saving embeddings to Supabase: {e}")
        else:
            print(f"Failed to capture images: {message}")
    
    def get_session_attendance_stats(self, session_id: str) -> Dict:
        """
        Get attendance statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with attendance statistics
        """
        try:
            # Get all enrollments for the course
            session_response = self.supabase.table('course_sessions').select(
                "*, courses(*)"
            ).eq('id', session_id).execute()
            
            if not session_response.data:
                return {}
            
            course_id = session_response.data[0]['course_id']
            
            # Get total enrolled students
            enrollments = self.supabase.table('enrollments').select(
                "count", count='exact'
            ).eq('course_id', course_id).eq('is_active', True).execute()
            
            total_enrolled = enrollments.count if enrollments else 0
            
            # Get present students
            attendance = self.supabase.table('attendance').select(
                "count", count='exact'
            ).eq('session_id', session_id).eq('is_present', True).execute()
            
            present_count = attendance.count if attendance else 0
            
            return {
                'total_enrolled': total_enrolled,
                'present': present_count,
                'absent': total_enrolled - present_count,
                'attendance_rate': (present_count / total_enrolled * 100) if total_enrolled > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting session stats: {e}")
            return {}


def main():
    """
    Example of using the Supabase attendance system with your schema.
    """
    
    # Initialize the system
    # Set your Supabase credentials in environment variables or pass them here
    attendance_system = SupabaseAttendanceSystem(
        supabase_url="YOUR_SUPABASE_URL",  # or set SUPABASE_URL env variable
        supabase_key="YOUR_SUPABASE_ANON_KEY"  # or set SUPABASE_KEY env variable
    )
    
    # Example workflow:
    
    # 1. Add a student (if not already added)
    student_id = attendance_system.add_student(
        full_name="John Doe",
        email="john.doe@university.edu",
        department="Computer Science",
        phone="+1234567890"
    )
    
    if student_id:
        # 2. Enroll student in a course
        attendance_system.enroll_student_in_course(student_id, "CS101")
        
        # 3. Capture student faces for recognition
        attendance_system.capture_and_save_student_faces("john.doe@university.edu", "CS101")
    
    # 4. Start an attendance session
    # You need to have a teacher_id from your teachers table
    teacher_id = "YOUR_TEACHER_UUID"  # Get this from your teachers table
    
    session_id = attendance_system.start_attendance_session(
        course_code="CS101",
        teacher_id=teacher_id,
        location="Room 101"
    )
    
    if session_id:
        try:
            # 5. Start face recognition
            print("\nStarting face recognition. Press 'q' to quit.")
            attendance_system.fr_system.start_recognition(camera_index=0)
            
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
            
        finally:
            # 6. End the session
            attendance_system.end_attendance_session()
            
            # 7. Get attendance statistics
            stats = attendance_system.get_session_attendance_stats(session_id)
            print(f"\nAttendance Statistics:")
            print(f"  Total Enrolled: {stats.get('total_enrolled', 0)}")
            print(f"  Present: {stats.get('present', 0)}")
            print(f"  Absent: {stats.get('absent', 0)}")
            print(f"  Attendance Rate: {stats.get('attendance_rate', 0):.1f}%")
            
            # 8. Get detailed attendance report
            report = attendance_system.get_attendance_report(session_id=session_id)
            print(f"\nDetailed Report: {len(report)} attendance records")


if __name__ == "__main__":
    main()