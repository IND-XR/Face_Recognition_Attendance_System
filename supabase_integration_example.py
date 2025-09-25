"""
Supabase Integration Example for Face Recognition Attendance System
This example shows how to integrate the face recognition module with Supabase.

Requirements:
- supabase-py: pip install supabase
- python-dotenv: pip install python-dotenv (for environment variables)
- face_recognition_module (the standalone module we created)

Author: Supabase Integration Example
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from face_recognition_module import FaceRecognitionAttendance
import asyncio
import json

# Load environment variables
load_dotenv()

class SupabaseAttendanceSystem:
    """
    Integration of Face Recognition with Supabase backend.
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
            images_base_path='face_data',  # You can customize this path
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
        
        # Session information
        self.current_session = {
            'course_id': None,
            'unit_id': None,
            'venue_id': None,
            'session_id': None,
            'lecturer_id': None
        }
        
    def create_tables_if_not_exists(self):
        """
        SQL to create necessary tables in Supabase.
        Run these in your Supabase SQL editor.
        """
        sql_commands = """
        -- Students table
        CREATE TABLE IF NOT EXISTS students (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            registration_number VARCHAR(50) UNIQUE NOT NULL,
            first_name VARCHAR(100) NOT NULL,
            last_name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            course_id UUID REFERENCES courses(id),
            face_encodings JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Courses table
        CREATE TABLE IF NOT EXISTS courses (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            course_code VARCHAR(50) UNIQUE NOT NULL,
            course_name VARCHAR(255) NOT NULL,
            department VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Units/Subjects table
        CREATE TABLE IF NOT EXISTS units (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            unit_code VARCHAR(50) UNIQUE NOT NULL,
            unit_name VARCHAR(255) NOT NULL,
            course_id UUID REFERENCES courses(id),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Venues table
        CREATE TABLE IF NOT EXISTS venues (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            venue_code VARCHAR(50) UNIQUE NOT NULL,
            venue_name VARCHAR(255) NOT NULL,
            capacity INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Attendance Sessions table
        CREATE TABLE IF NOT EXISTS attendance_sessions (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            course_id UUID REFERENCES courses(id),
            unit_id UUID REFERENCES units(id),
            venue_id UUID REFERENCES venues(id),
            lecturer_id UUID,
            session_date DATE NOT NULL,
            start_time TIMESTAMP WITH TIME ZONE NOT NULL,
            end_time TIMESTAMP WITH TIME ZONE,
            status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Attendance Records table
        CREATE TABLE IF NOT EXISTS attendance_records (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id UUID REFERENCES attendance_sessions(id),
            student_id UUID REFERENCES students(id),
            registration_number VARCHAR(50),
            marked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            status VARCHAR(50) DEFAULT 'present',
            confidence_score FLOAT,
            UNIQUE(session_id, student_id)
        );

        -- Detection Logs table (optional, for analytics)
        CREATE TABLE IF NOT EXISTS detection_logs (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id UUID REFERENCES attendance_sessions(id),
            student_id UUID REFERENCES students(id),
            detection_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            confidence_score FLOAT,
            is_unknown BOOLEAN DEFAULT FALSE
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_attendance_session_student ON attendance_records(session_id, student_id);
        CREATE INDEX IF NOT EXISTS idx_student_registration ON students(registration_number);
        CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_sessions(session_date);
        """
        
        print("SQL commands to create tables:")
        print(sql_commands)
        return sql_commands
    
    async def get_students_by_course(self, course_code: str) -> List[Dict]:
        """
        Fetch all students for a specific course from Supabase.
        
        Args:
            course_code: The course code
            
        Returns:
            List of student records
        """
        try:
            # First get the course ID
            course_response = self.supabase.table('courses').select("*").eq(
                'course_code', course_code
            ).execute()
            
            if not course_response.data:
                print(f"Course {course_code} not found")
                return []
            
            course_id = course_response.data[0]['id']
            
            # Get students for this course
            students_response = self.supabase.table('students').select("*").eq(
                'course_id', course_id
            ).execute()
            
            return students_response.data
            
        except Exception as e:
            print(f"Error fetching students: {e}")
            return []
    
    async def load_students_for_course(self, course_code: str) -> bool:
        """
        Load face encodings for all students in a course.
        
        Args:
            course_code: The course code
            
        Returns:
            Success status
        """
        try:
            # Load faces from local storage
            stats = self.fr_system.load_faces_for_group(course_code)
            print(f"Loaded {stats['successful_encodings']} faces from local storage")
            
            # Optionally, sync with Supabase to get any additional encodings
            students = await self.get_students_by_course(course_code)
            
            for student in students:
                if student.get('face_encodings'):
                    # Load additional encodings from database if available
                    # This is useful if encodings are stored in the cloud
                    pass
                    
            return True
            
        except Exception as e:
            print(f"Error loading students: {e}")
            return False
    
    def start_attendance_session(self, 
                                course_id: str,
                                unit_id: str,
                                venue_id: str,
                                lecturer_id: Optional[str] = None) -> Optional[str]:
        """
        Start a new attendance session.
        
        Args:
            course_id: Course ID
            unit_id: Unit/Subject ID
            venue_id: Venue ID
            lecturer_id: Lecturer ID (optional)
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            # Create a new attendance session
            session_data = {
                'course_id': course_id,
                'unit_id': unit_id,
                'venue_id': venue_id,
                'lecturer_id': lecturer_id,
                'session_date': datetime.now().date().isoformat(),
                'start_time': datetime.now().isoformat(),
                'status': 'active'
            }
            
            response = self.supabase.table('attendance_sessions').insert(
                session_data
            ).execute()
            
            if response.data:
                session_id = response.data[0]['id']
                
                # Update current session info
                self.current_session = {
                    'course_id': course_id,
                    'unit_id': unit_id,
                    'venue_id': venue_id,
                    'session_id': session_id,
                    'lecturer_id': lecturer_id
                }
                
                print(f"Started attendance session: {session_id}")
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
            # Update session status and end time
            response = self.supabase.table('attendance_sessions').update({
                'status': 'completed',
                'end_time': datetime.now().isoformat()
            }).eq('id', self.current_session['session_id']).execute()
            
            if response.data:
                print(f"Ended session: {self.current_session['session_id']}")
                
                # Clear current session
                self.current_session = {
                    'course_id': None,
                    'unit_id': None,
                    'venue_id': None,
                    'session_id': None,
                    'lecturer_id': None
                }
                
                return True
                
        except Exception as e:
            print(f"Error ending session: {e}")
            
        return False
    
    def mark_attendance(self, student_id: str, metadata: Dict, timestamp: datetime):
        """
        Callback function to mark attendance in Supabase.
        
        Args:
            student_id: Student registration number
            metadata: Additional metadata about the student
            timestamp: Time of detection
        """
        if not self.current_session['session_id']:
            print("No active session for marking attendance")
            return
            
        try:
            # Get student UUID from registration number
            student_response = self.supabase.table('students').select("id").eq(
                'registration_number', student_id
            ).execute()
            
            if not student_response.data:
                print(f"Student {student_id} not found in database")
                return
                
            student_uuid = student_response.data[0]['id']
            
            # Check if attendance already marked
            existing = self.supabase.table('attendance_records').select("id").eq(
                'session_id', self.current_session['session_id']
            ).eq('student_id', student_uuid).execute()
            
            if existing.data:
                print(f"Attendance already marked for {student_id}")
                return
                
            # Mark attendance
            attendance_data = {
                'session_id': self.current_session['session_id'],
                'student_id': student_uuid,
                'registration_number': student_id,
                'marked_at': timestamp.isoformat(),
                'status': 'present',
                'confidence_score': metadata.get('confidence', 0)
            }
            
            response = self.supabase.table('attendance_records').insert(
                attendance_data
            ).execute()
            
            if response.data:
                print(f"✓ Attendance marked in Supabase for {student_id} at {timestamp}")
                
        except Exception as e:
            print(f"Error marking attendance: {e}")
    
    def log_detection(self, student_id: str, confidence: float, frame):
        """
        Log face detection event (optional, for analytics).
        
        Args:
            student_id: Student registration number
            confidence: Detection confidence score
            frame: Video frame (not used in this example)
        """
        if not self.current_session['session_id']:
            return
            
        try:
            # Get student UUID
            student_response = self.supabase.table('students').select("id").eq(
                'registration_number', student_id
            ).execute()
            
            if student_response.data:
                student_uuid = student_response.data[0]['id']
                
                # Log the detection
                log_data = {
                    'session_id': self.current_session['session_id'],
                    'student_id': student_uuid,
                    'detection_time': datetime.now().isoformat(),
                    'confidence_score': confidence,
                    'is_unknown': False
                }
                
                self.supabase.table('detection_logs').insert(log_data).execute()
                
        except Exception as e:
            print(f"Error logging detection: {e}")
    
    def handle_unknown_face(self, face_encoding, frame):
        """
        Handle unknown face detection.
        
        Args:
            face_encoding: Face encoding of unknown person
            frame: Video frame (not used in this example)
        """
        if not self.current_session['session_id']:
            return
            
        try:
            # Log unknown face detection
            log_data = {
                'session_id': self.current_session['session_id'],
                'detection_time': datetime.now().isoformat(),
                'is_unknown': True
            }
            
            self.supabase.table('detection_logs').insert(log_data).execute()
            print("Unknown face detected and logged")
            
        except Exception as e:
            print(f"Error logging unknown face: {e}")
    
    def get_attendance_report(self, session_id: str = None, course_id: str = None, 
                             date_from: str = None, date_to: str = None) -> List[Dict]:
        """
        Get attendance report based on filters.
        
        Args:
            session_id: Specific session ID
            course_id: Filter by course
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            
        Returns:
            List of attendance records
        """
        try:
            query = self.supabase.table('attendance_records').select(
                "*, students(registration_number, first_name, last_name), attendance_sessions(session_date, courses(course_name), units(unit_name))"
            )
            
            if session_id:
                query = query.eq('session_id', session_id)
                
            if course_id:
                # This would need a join through attendance_sessions
                pass
                
            response = query.execute()
            return response.data
            
        except Exception as e:
            print(f"Error getting attendance report: {e}")
            return []
    
    def add_student(self, registration_number: str, first_name: str, 
                   last_name: str, email: str, course_code: str) -> bool:
        """
        Add a new student to the system.
        
        Args:
            registration_number: Student registration number
            first_name: Student first name
            last_name: Student last name
            email: Student email
            course_code: Course code
            
        Returns:
            Success status
        """
        try:
            # Get course ID
            course_response = self.supabase.table('courses').select("id").eq(
                'course_code', course_code
            ).execute()
            
            if not course_response.data:
                print(f"Course {course_code} not found")
                return False
                
            course_id = course_response.data[0]['id']
            
            # Add student
            student_data = {
                'registration_number': registration_number,
                'first_name': first_name,
                'last_name': last_name,
                'email': email,
                'course_id': course_id
            }
            
            response = self.supabase.table('students').insert(student_data).execute()
            
            if response.data:
                print(f"Student {registration_number} added successfully")
                return True
                
        except Exception as e:
            print(f"Error adding student: {e}")
            
        return False
    
    def capture_and_save_student_faces(self, registration_number: str, course_code: str):
        """
        Capture student face images and save encodings.
        
        Args:
            registration_number: Student registration number
            course_code: Course code
        """
        # Capture images
        success, message, image_paths = self.fr_system.capture_student_images(
            student_id=registration_number,
            group_identifier=course_code,
            num_images=25,
            camera_index=0
        )
        
        if success:
            print(message)
            
            # Optionally, generate and store encodings in Supabase
            # This allows for cloud-based face recognition
            encodings = self.fr_system._load_student_encodings(
                os.path.join(self.fr_system.images_base_path, course_code, registration_number),
                registration_number
            )
            
            if encodings:
                # Store encodings in Supabase (as JSON)
                try:
                    encoding_data = {
                        'face_encodings': json.dumps([enc.tolist() for enc in encodings])
                    }
                    
                    self.supabase.table('students').update(encoding_data).eq(
                        'registration_number', registration_number
                    ).execute()
                    
                    print("Face encodings saved to Supabase")
                    
                except Exception as e:
                    print(f"Error saving encodings to Supabase: {e}")
        else:
            print(f"Failed to capture images: {message}")


# Example usage
async def main():
    """
    Example of using the Supabase attendance system.
    """
    
    # Initialize the system
    # You need to set your Supabase credentials in environment variables or pass them here
    attendance_system = SupabaseAttendanceSystem(
        supabase_url="YOUR_SUPABASE_URL",
        supabase_key="YOUR_SUPABASE_ANON_KEY"
    )
    
    # Example workflow:
    
    # 1. Add a student (if not already added)
    attendance_system.add_student(
        registration_number="STU001",
        first_name="John",
        last_name="Doe",
        email="john.doe@example.com",
        course_code="CS101"
    )
    
    # 2. Capture student faces
    attendance_system.capture_and_save_student_faces("STU001", "CS101")
    
    # 3. Load students for a course
    await attendance_system.load_students_for_course("CS101")
    
    # 4. Start an attendance session
    session_id = attendance_system.start_attendance_session(
        course_id="course-uuid-here",
        unit_id="unit-uuid-here",
        venue_id="venue-uuid-here",
        lecturer_id="lecturer-uuid-here"
    )
    
    if session_id:
        try:
            # 5. Start face recognition
            attendance_system.fr_system.start_recognition(camera_index=0)
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        finally:
            # 6. End the session
            attendance_system.end_attendance_session()
            
            # 7. Get attendance report
            report = attendance_system.get_attendance_report(session_id=session_id)
            print(f"\nAttendance Report: {len(report)} students marked present")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())