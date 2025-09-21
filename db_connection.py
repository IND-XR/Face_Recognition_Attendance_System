import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='anmol150',
            database='new11',
            port=3306  # explicitly specify port
        )
        if connection.is_connected():
            print("Successfully connected to MySQL database")
            return connection
        else:
            print("Connection failed")
            return None
    except Error as e:
        if e.errno == 2003:
            print("MySQL server is not running. Please start the MySQL service.")
        elif e.errno == 1045:
            print("Invalid username or password")
        elif e.errno == 1049:
            print("Database 'new11' does not exist")
        else:
            print(f"Error connecting to MySQL: {e}")
        return None

def check_user_login(email, password, role):
    try:
        conn = get_db_connection()
        if conn is None:
            print("Could not establish database connection")
            return None
            
        cursor = conn.cursor(dictionary=True)
        
        if role == 'Admin':
            query = "SELECT * FROM tbladmin WHERE emailAddress = %s AND password = %s"
        elif role == 'Lecturer':
            query = "SELECT * FROM tbllecturer WHERE emailAddress = %s AND password = %s"
        else:
            return None
            
        cursor.execute(query, (email, password))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return user
    except Error as e:
        print(f"Error in check_user_login: {e}")
        return None

def get_total_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tblstudents")
    total_students = cursor.fetchone()[0]
    conn.close()
    return total_students

def get_total_units():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "SELECT COUNT(*) FROM tblunit"
        
        cursor.execute(query)
        total = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return total
    except Exception as e:
        print(f"Error fetching total units: {e}")
        return 0

def get_total_lecturers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tbllecturer")
    total_lecturers = cursor.fetchone()[0]
    conn.close()
    return total_lecturers

def get_lecturers():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tbllecturer")
    lecturers = cursor.fetchall()
    conn.close()
    return lecturers

def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                s.id,
                s.registrationNumber,
                s.firstName,
                s.lastName,
                s.faculty,
                s.courseCode,
                s.email,
                DATE_FORMAT(s.dateRegistered, '%Y-%m-%d') as dateRegistered
            FROM tblstudents s
            ORDER BY s.registrationNumber ASC
        """)
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        return students
    except Exception as e:
        print(f"Error fetching students: {e}")
        return []

def get_lecture_rooms():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tblvenue")
    lecture_rooms = cursor.fetchall()
    conn.close()
    return lecture_rooms

def get_courses():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT 
            c.id,
            c.name,
            c.courseCode,
            c.facultyID,  # Get the faculty ID
            COALESCE(
                (SELECT COUNT(*) FROM tblunit WHERE courseID = c.courseCode), 
                0
            ) as total_units,
            COALESCE(
                (SELECT COUNT(*) FROM tblstudents WHERE courseCode = c.courseCode), 
                0
            ) as total_students,
            c.dateCreated
        FROM tblcourse c
        ORDER BY c.name
        """
        
        print("Executing course query...")  # Debug print
        cursor.execute(query)
        courses = cursor.fetchall()
        print(f"Found {len(courses)} courses:", courses)  # Debug print
        
        cursor.close()
        connection.close()
        
        return courses
    except Exception as e:
        print(f"Error fetching courses: {e}")
        return []

def get_units():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT 
            id,
            name,
            unitCode as code,
            courseID as course,
            (SELECT COUNT(*) FROM tblattendance WHERE unit = u.unitCode) as total_students,
            dateCreated as date_created
        FROM tblunit u
        """
        
        cursor.execute(query)
        units = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return units
    except Exception as e:
        print(f"Error fetching units: {e}")
        return []

def get_total_faculties():
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = "SELECT COUNT(*) FROM tblfaculty"
        
        cursor.execute(query)
        total = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        return total
    except Exception as e:
        print(f"Error fetching total faculties: {e}")
        return 0

def get_faculties():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get all faculties from tblfaculty
        query = """
        SELECT 
            id,
            facultyName,
            facultyCode
        FROM tblfaculty 
        ORDER BY facultyName
        """
        
        cursor.execute(query)
        faculties = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return faculties
    except Exception as e:
        print(f"Error fetching faculties: {e}")
        return []

def add_faculty_to_db(faculty_code, faculty_name):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = """
        INSERT INTO tblfaculty (facultyName, facultyCode, dateRegistered) 
        VALUES (%s, %s, NOW())
        """
        cursor.execute(query, (faculty_name, faculty_code))
        connection.commit()
        
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        print(f"Error adding faculty: {str(e)}")
        return False

def add_course_to_db(name, code, faculty_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        print(f"Adding course - Name: {name}, Code: {code}, Faculty: {faculty_id}")  # Debug print
        
        # First verify if the faculty exists
        cursor.execute("SELECT facultyCode FROM tblfaculty WHERE facultyCode = %s", (faculty_id,))
        faculty = cursor.fetchone()
        
        if not faculty:
            print(f"Faculty with code {faculty_id} not found")
            return False
            
        # Add the course with the faculty code
        cursor.execute("""
            INSERT INTO tblcourse (name, courseCode, facultyID, dateCreated)
            VALUES (%s, %s, %s, NOW())
        """, (name, code, faculty_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"Course added successfully with faculty code: {faculty_id}")
        return True
        
    except Exception as e:
        print(f"Error adding course: {e}")
        return False

def add_unit_to_db(unit_code, unit_name, course_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # First verify if the course exists
        cursor.execute("SELECT courseCode FROM tblcourse WHERE courseCode = %s", (course_id,))
        course = cursor.fetchone()
        
        if not course:
            print(f"Course with code {course_id} not found")  # Debug print
            return False
            
        # If course exists, add the unit
        cursor.execute("""
            INSERT INTO tblunit (unitCode, name, courseID, dateCreated)
            VALUES (%s, %s, %s, NOW())
        """, (unit_code, unit_name, course_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        print(f"Error adding unit: {e}")  # Debug print
        return False

def get_lecturer_by_email(email):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get lecturer details
        query = """
        SELECT 
            l.*,
            (SELECT COUNT(*) FROM tblunit u 
             WHERE u.lecturerID = l.id) as total_units
        FROM tbllecturer l
        WHERE l.emailAddress = %s
        """
        
        cursor.execute(query, (email,))
        lecturer = cursor.fetchone()
        
        if lecturer:
            # Get lecturer's courses (only for their faculty)
            courses_query = """
            SELECT DISTINCT c.* 
            FROM tblcourse c
            WHERE c.facultyID = %s
            """
            cursor.execute(courses_query, (lecturer['facultyCode'],))
            lecturer['courses'] = cursor.fetchall()
            
            # Get lecturer's units
            units_query = """
            SELECT u.* FROM tblunit u
            WHERE u.lecturerID = %s
            """
            cursor.execute(units_query, (lecturer['id'],))
            lecturer['units'] = cursor.fetchall()
            
            # Get venues (only for their faculty)
            venues_query = "SELECT * FROM tblvenue WHERE currentStatus = 'Available' AND facultyCode = %s"
            cursor.execute(venues_query, (lecturer['facultyCode'],))
            lecturer['venues'] = cursor.fetchall()
            
        cursor.close()
        connection.close()
        return lecturer
        
    except Exception as e:
        print(f"Error in get_lecturer_by_email: {str(e)}")
        return None

def get_units_by_course(course_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get all units for this course
        query = """
        SELECT 
            u.id,
            u.name,
            u.unitCode,
            u.courseID
        FROM tblunit u
        JOIN tblcourse c ON u.courseID = c.courseCode
        WHERE c.id = %s
        ORDER BY u.name
        """
        
        cursor.execute(query, (course_id,))
        units = cursor.fetchall()
        
        # Debug print
        print(f"Found units for course {course_id}:", units)
        
        cursor.close()
        connection.close()
        
        return units
    except Exception as e:
        print(f"Error fetching units by course: {e}")
        return []

def get_students_by_course_unit(course_id, unit_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # First get the course code
        cursor.execute("SELECT courseCode FROM tblcourse WHERE id = %s", (course_id,))
        course_result = cursor.fetchone()
        
        if not course_result:
            print("Course not found")
            return []
            
        course_code = course_result['courseCode']
        
        # Get all students for this course
        query = """
        SELECT 
            s.id,
            s.firstName,
            s.lastName,
            s.registrationNumber as regNo,
            s.email,
            s.courseCode,
            c.name as courseName,
            u.name as unitName
        FROM tblstudents s
        JOIN tblcourse c ON s.courseCode = c.courseCode
        JOIN tblunit u ON u.courseID = c.courseCode
        WHERE s.courseCode = %s AND u.id = %s
        ORDER BY s.firstName
        """
        
        cursor.execute(query, (course_code, unit_id))
        students = cursor.fetchall()
        
        # Debug print
        print(f"Found {len(students)} students for course {course_code} and unit {unit_id}")
        
        cursor.close()
        connection.close()
        
        return students
    except Exception as e:
        print(f"Error fetching students by course and unit: {e}")
        return []

def get_lecturer_courses(lecturer_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Get courses where lecturer teaches at least one unit
        query = """
        SELECT DISTINCT 
            c.id, 
            c.name, 
            c.courseCode,
            c.facultyID
        FROM tblcourse c
        INNER JOIN tblunit u ON u.courseID = c.courseCode
        WHERE u.lecturerID = %s
        ORDER BY c.name
        """
        
        cursor.execute(query, (lecturer_id,))
        courses = cursor.fetchall()
        
        # Debug print to check the results
        print(f"Found courses for lecturer {lecturer_id}:", courses)
        
        cursor.close()
        connection.close()
        
        return courses
    except Exception as e:
        print(f"Error fetching lecturer courses: {e}")
        return []

def get_students_by_course(course_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # First get the course code
        cursor.execute("SELECT courseCode FROM tblcourse WHERE id = %s", (course_id,))
        course_result = cursor.fetchone()
        
        if not course_result:
            print("Course not found")
            return []
            
        course_code = course_result['courseCode']
        
        # Get all students for this course
        query = """
        SELECT 
            s.id,
            s.firstName,
            s.lastName,
            s.registrationNumber as regNo,
            s.email,
            s.courseCode,
            c.name as courseName
        FROM tblstudents s
        JOIN tblcourse c ON s.courseCode = c.courseCode
        WHERE s.courseCode = %s
        ORDER BY s.firstName
        """
        
        cursor.execute(query, (course_code,))
        students = cursor.fetchall()
        
        # Debug print
        print(f"Found {len(students)} students for course {course_code}")
        print("Students:", students)
        
        cursor.close()
        connection.close()
        
        return students
    except Exception as e:
        print(f"Error fetching students by course: {e}")
        return []

def add_venue(className, currentStatus, capacity, classType, faculty_code):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Verify the faculty code exists
        cursor.execute("SELECT facultyCode FROM tblfaculty WHERE facultyCode = %s", (faculty_code,))
        if not cursor.fetchone():
            print(f"Faculty code {faculty_code} not found")
            return False
            
        query = """
        INSERT INTO tblvenue 
        (className, currentStatus, capacity, classification, facultyCode) 
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            className,
            currentStatus,
            capacity,
            classType,
            faculty_code
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
        
    except Exception as e:
        print(f"Error adding venue: {e}")
        return False

def add_lecturer_to_db(first_name, last_name, email, phone, password, faculty_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Get faculty code from faculty ID
        cursor.execute("SELECT facultyCode FROM tblfaculty WHERE id = %s", (faculty_id,))
        faculty_result = cursor.fetchone()
        
        if not faculty_result:
            print("Faculty not found")
            return False
            
        faculty_code = faculty_result[0]
        
        query = """
        INSERT INTO tbllecturer 
        (firstName, lastName, emailAddress, phoneNo, password, facultyCode, dateCreated) 
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """
        
        cursor.execute(query, (
            first_name,
            last_name,
            email,
            phone,
            password,
            faculty_code
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return True
    except Exception as e:
        print(f"Error adding lecturer: {e}")
        return False

def get_courses_by_faculty(faculty_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT id, name, courseCode 
        FROM tblcourse 
        WHERE facultyID = %s
        """
        
        cursor.execute(query, (faculty_id,))
        courses = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        print(f"Found courses for faculty {faculty_id}:", courses)  # Debug print
        return courses
        
    except Exception as e:
        print(f"Error getting courses by faculty: {e}")
        return []

def add_student_to_db(first_name, last_name, email, reg_no, course_id, faculty):
    try:
        # Get a new database connection using the existing function
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Insert student with faculty information
        cursor.execute("""
            INSERT INTO tblstudents (
                firstName, 
                lastName, 
                email, 
                registrationNumber, 
                courseCode,
                faculty,
                dateRegistered
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            first_name,
            last_name,
            email,
            reg_no,
            course_id,
            faculty
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"Successfully added student: {first_name} {last_name}")  # Debug print
        return True
        
    except Exception as e:
        print(f"Error in add_student_to_db: {e}")  # Debug print
        if 'connection' in locals() and connection.is_connected():
            connection.close()
        return False

def update_course_totals(course_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Update total units
        units_query = """
        UPDATE tblcourse c 
        SET total_units = (
            SELECT COUNT(*) 
            FROM tblunit u 
            WHERE u.courseID = c.courseCode
        )
        WHERE id = %s
        """
        
        # Update total students
        students_query = """
        UPDATE tblcourse c 
        SET total_students = (
            SELECT COUNT(*) 
            FROM tblstudents s 
            WHERE s.courseCode = c.courseCode
        )
        WHERE id = %s
        """
        
        cursor.execute(units_query, (course_id,))
        cursor.execute(students_query, (course_id,))
        connection.commit()
        
    except Exception as e:
        print(f"Error updating course totals: {e}")
    finally:
        cursor.close()
        connection.close()

def update_faculty_totals(faculty_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Update faculty totals
        query = """
        UPDATE tblfaculty f 
        SET 
            total_courses = (
                SELECT COUNT(*) 
                FROM tblcourse c 
                WHERE c.facultyID = f.id
            ),
            total_students = (
                SELECT COUNT(*) 
                FROM tblstudents s 
                WHERE s.faculty = f.facultyCode
            ),
            total_lectures = (
                SELECT COUNT(*) 
                FROM tbllecturer l 
                WHERE l.facultyCode = f.facultyCode
            )
        WHERE id = %s
        """
        
        cursor.execute(query, (faculty_id,))
        connection.commit()
        
    except Exception as e:
        print(f"Error updating faculty totals: {e}")
    finally:
        cursor.close()
        connection.close()

def get_venues_by_course(course_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Get the faculty code for the course
        cursor.execute("SELECT facultyID FROM tblcourse WHERE id = %s", (course_id,))
        course = cursor.fetchone()
        if not course or not course['facultyID']:
            return []
        faculty_code = course['facultyID']

        # Get venues for this faculty
        cursor.execute("SELECT * FROM tblvenue WHERE facultyCode = %s", (faculty_code,))
        venues = cursor.fetchall()
        cursor.close()
        connection.close()
        return venues
    except Exception as e:
        print(f"Error fetching venues by course: {e}")
        return []

def get_available_venues_by_course(faculty_code):
    """Get available venues for a faculty"""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                id,
                className,
                currentStatus,
                capacity,
                classType
            FROM tblvenue
            WHERE facultyCode = %s AND currentStatus = 'Available'
            ORDER BY className
        """, (faculty_code,))
        
        venues = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return venues
    except Exception as e:
        print(f"Error getting available venues: {e}")
        return []

def get_students_for_course_unit(course_id, unit_id):
    # Example using MySQLdb
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT s.registrationNumber, CONCAT(s.firstName, ' ', s.lastName) as name
        FROM tblstudents s
        JOIN tblcourse c ON s.courseCode = c.courseCode
        JOIN tblunit u ON u.courseID = c.courseCode
        WHERE c.id = %s AND u.id = %s
    """, (course_id, unit_id))
    students = cursor.fetchall()
    cursor.close()
    connection.close()
    return students

def get_attendance_for_unit(unit_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("""
        SELECT 
            studentRegistrationNumber as registrationNumber,
            attendanceStatus,
            dateMarked
        FROM tblattendance
        WHERE unit = (
            SELECT unitCode FROM tblunit WHERE id = %s
        )
    """, (unit_id,))
    records = cursor.fetchall()
    cursor.close()
    connection.close()
    return records
