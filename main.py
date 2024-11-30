import cv2
import numpy as np
import face_recognition
import psycopg2
import pyaudio
import threading
import queue
from datetime import datetime
import requests
from urllib.parse import urljoin
import time
import logging
import os
from pathlib import Path


class ComprehensiveExamProctor:
    def __init__(self):
        # Setup directories and logging
        self.setup_directories()
        self.setup_logging()

        # Database connection - using your original connection
        self.db_conn = psycopg2.connect(
            dbname="ProcPlus",
            user="postgres.psmllggonmzfpfyhgqqb",
            password="I3v3Ywrx2sCg5zUl",
            host="aws-0-ap-south-1.pooler.supabase.com",
            port=6543
        )
        self.cursor = self.db_conn.cursor()

        # Your original media server URL
        self.media_base_url = "http://localhost:8000"

        # Initialize video capture and recording
        self.setup_video()
        self.setup_audio()

        # Student data storage
        self.known_faces = {}
        self.load_student_faces()

        # State variables
        self.previous_frame = None
        self.prev_frame_time = 0
        self.current_violations = {}
        self.violation_queue = queue.Queue()

        # Configuration
        self.config = {
            'fps_limit': 10,
            'motion_threshold': 50,
            'gaze_threshold': 30,
            'audio_threshold': 0.1,
            'face_tolerance': 0.6
        }

        self.violation_logs = []

    def setup_directories(self):
        """Create necessary directories"""
        base_dir = Path("exam_surveillance_output")
        self.dirs = {
            'base': base_dir,
            'logs': base_dir / "logs",
            'violations': base_dir / "violations",
            'recordings': base_dir / "recordings"
        }
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_file = self.dirs['logs'] / f'exam_proctor_{timestamp}.log'

        self.logger = logging.getLogger('ExamProctor')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_video(self):
        """Initialize video capture and recording"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open video capture")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        video_file = str(self.dirs['recordings'] / f'exam_recording_{timestamp}.avi')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            video_file, fourcc, 20.0, (width, height)
        )

    def setup_audio(self):
        """Initialize audio monitoring"""
        self.audio_queue = queue.Queue()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100

        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def load_student_faces(self):
        """Load student faces from database"""
        try:
            self.cursor.execute("""
                SELECT id, email, first_name, last_name, profile_image 
                FROM authentication_user 
                WHERE profile_image IS NOT NULL
            """)
            students = self.cursor.fetchall()

            self.logger.info(f"Found {len(students)} students with profile images")

            for student_id, email, first_name, last_name, profile_image in students:
                if profile_image:
                    face_encoding = self.download_and_encode_face(profile_image)
                    if face_encoding is not None:
                        self.known_faces[student_id] = {
                            'name': f"{first_name} {last_name}",
                            'email': email,
                            'encoding': face_encoding
                        }
                        self.logger.info(f"Loaded face encoding for {email}")

        except Exception as e:
            self.logger.error(f"Error loading student faces: {e}")

    def download_and_encode_face(self, profile_image_url):
        """Download and encode profile images"""
        try:
            if not profile_image_url.startswith('/'):
                profile_image_url = '/' + profile_image_url
            if not profile_image_url.startswith('/media/'):
                profile_image_url = '/media' + profile_image_url

            full_url = urljoin(self.media_base_url, profile_image_url)
            self.logger.info(f"Downloading image from: {full_url}")

            response = requests.get(full_url)
            if response.status_code != 200:
                self.logger.error(f"Failed to download image: {response.status_code}")
                return None

            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                return None

            return face_recognition.face_encodings(rgb_image, face_locations)[0]

        except Exception as e:
            self.logger.error(f"Error processing profile image: {e}")
            return None

    def monitor_audio(self):
        """Audio monitoring thread function"""
        while True:
            try:
                data = np.frombuffer(self.audio_stream.read(self.CHUNK), dtype=np.float32)
                if np.max(np.abs(data)) > self.config['audio_threshold']:
                    self.violation_queue.put(("Audio", "Talking detected"))
            except Exception as e:
                self.logger.error(f"Audio monitoring error: {e}")

    def log_violation(self, violation_type, user_info, description):
        """Enhanced violation logging with user details"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        violation_data = {
            'timestamp': timestamp,
            'type': violation_type,
            'user': user_info,  # Can be email or "Unknown Person"
            'description': description
        }

        # Add to in-memory log
        self.violation_logs.append(violation_data)

        # Log to file
        self.logger.warning(
            f"Violation: {violation_type} | User: {user_info} | Description: {description}"
        )

        return violation_data

    def detect_motion(self, frame):
        """Enhanced motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        motion_score = np.sum(thresh) / 255
        self.previous_frame = gray

        return motion_score > self.config['motion_threshold']

    def detect_gaze(self, frame, face_location):
        """Detect if student is looking away"""
        try:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_landmarks = face_recognition.face_landmarks(face_image)

            if face_landmarks:
                left_eye = np.mean(face_landmarks[0]['left_eye'], axis=0)
                right_eye = np.mean(face_landmarks[0]['right_eye'], axis=0)

                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))

                return abs(eye_angle) > self.config['gaze_threshold']
        except Exception as e:
            self.logger.error(f"Gaze detection error: {e}")
        return False

    def process_frame(self, frame):
        """Process frame with all detections"""
        current_time = time.time()
        time_elapsed = current_time - self.prev_frame_time

        # FPS control
        if time_elapsed > 1. / self.config['fps_limit']:
            self.prev_frame_time = current_time
            current_violations = []

            # Convert to RGB for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Track detected users in this frame
            detected_users = []

            # Process each face
            for face_encoding, face_loc in zip(face_encodings, face_locations):
                name = "Unknown Person"
                color = (0, 0, 255)  # Red for unknown faces

                # Check against known faces
                for student_id, data in self.known_faces.items():
                    if face_recognition.compare_faces(
                            [data['encoding']],
                            face_encoding,
                            tolerance=self.config['face_tolerance']
                    )[0]:
                        name = data['email']
                        detected_users.append(name)
                        color = (0, 255, 0)  # Green for recognized faces

                        # Check gaze
                        if self.detect_gaze(frame, face_loc):
                            violation = self.log_violation(
                                "Gaze",
                                name,
                                "Looking away from screen"
                            )
                            current_violations.append((violation['type'], f"{name}: {violation['description']}"))
                        break

                # Draw face box and name
                top, right, bottom, left = face_loc
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            # Check motion with user context
            if self.detect_motion(frame):
                users_str = ', '.join(detected_users) if detected_users else "Unknown users"
                violation = self.log_violation(
                    "Motion",
                    users_str,
                    "Suspicious movement detected"
                )
                current_violations.append((violation['type'], f"{users_str}: {violation['description']}"))

            # Check audio violations
            while not self.violation_queue.empty():
                audio_type, audio_msg = self.violation_queue.get()
                users_str = ', '.join(detected_users) if detected_users else "Unknown users"
                violation = self.log_violation(
                    audio_type,
                    users_str,
                    audio_msg
                )
                current_violations.append((violation['type'], f"{users_str}: {violation['description']}"))

            # Draw violations on frame
            y_offset = 30
            for violation_type, message in current_violations:
                cv2.putText(
                    frame,
                    f"{violation_type}: {message}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )
                y_offset += 25

            # Add FPS counter
            fps = 1 / time_elapsed
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Record if violations found
            if current_violations:
                self.video_writer.write(frame)

        return frame

    def save_violation_logs(self):
        """Save violation logs to file"""
        if self.violation_logs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            log_file = self.dirs['logs'] / f'violations_{timestamp}.txt'

            with open(log_file, 'w') as f:
                for violation in self.violation_logs:
                    f.write(
                        f"[{violation['timestamp']}] "
                        f"{violation['type']} | "
                        f"User: {violation['user']} | "
                        f"{violation['description']}\n"
                    )

    def run(self):
        """Main monitoring loop"""
        # Start audio monitoring thread
        audio_thread = threading.Thread(target=self.monitor_audio, daemon=True)
        audio_thread.start()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    break

                # Process frame with all detections
                processed_frame = self.process_frame(frame)

                # Display the frame
                cv2.imshow('Exam Surveillance', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        self.save_violation_logs()  # Save violation logs before cleanup

        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()

        self.cursor.close()
        self.db_conn.close()


if __name__ == "__main__":
    proctor = ComprehensiveExamProctor()
    proctor.run()