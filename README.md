📑 Table of Contents

📖 About the Project

🚀 Getting Started

✅ Prerequisites

⚙️ Installation

🛠 Database Setup

▶️ Run the Project

📸 Screenshots

🛠 Built With

🤝 Contributing

📜 License

🙌 Acknowledgements

📖 About the Project

The Face Recognition Attendance System is a Flask-based web application that integrates OpenCV for face detection and MySQL for secure data storage.
It helps automate the process of attendance management in schools, colleges, and organizations.

✨ Features:

🔐 Role-based Login (Teacher / Parent / Admin)

🧑‍🏫 Teacher Portal (Attendance, Timetable, Reports)

🗂 MySQL Database Integration

🎥 Live Camera Feed for Face Recognition

📊 Attendance Report Generation

🚀 Getting Started

Follow these instructions to set up the project locally.

✅ Prerequisites

Python 3.8+

MySQL Community Server + MySQL Workbench

Git

⚙️ Installation
# Clone the repository
git clone https://github.com/your-username/attendance-system.git
cd attendance-system

# Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🛠 Database Setup

Open MySQL Workbench.

Create a database:

CREATE DATABASE new11;


Import the schema:

Go to File → Open SQL Script → select attendance.sql.

Execute All ⚡.

Update db_connection.py with your MySQL credentials:

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="new11",
    port=3306
)

▶️ Run the Project
# Run Flask server
python app.py


Then open 👉 http://127.0.0.1:5000
 in your browser.

📸 Screenshots

(Add screenshots inside a screenshots/ folder and link them here)

![Login Page](screenshots/login.png)
![Attendance Dashboard](screenshots/dashboard.png)

🛠 Built With

Python

Flask

MySQL

OpenCV

🤝 Contributing

Contributions are welcome! 🎉

Fork the project

Create a new branch (feature/YourFeature)

Commit your changes

Push to your branch

Open a Pull Request

📜 License

Distributed under the MIT License. See LICENSE
 for details.

🙌 Acknowledgements

Flask Community

OpenCV Developers

MySQL Community

All contributors who help improve this project
