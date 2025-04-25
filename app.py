from src.lib.detection import *
from flask import Flask, render_template , Response, abort
from flask import Flask, redirect, url_for, render_template, request, session, flash
from datetime import timedelta # To store input data for permanent or longer time (session)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import shutil
import os, cv2
import sys
from ultralytics import YOLO
import torch
import numpy as np
# from .models.behaviour_net import BehaviourClassifier


app = Flask(__name__)
app.secret_key = "pakr25"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.sqlite3'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.permanent_session_lifetime = timedelta(days=10)

db = SQLAlchemy(app)

class Accounts(db.Model): # accounts=table name
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=True)
    password = db.Column(db.String(100), nullable=False)
    
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if username or email already exists
        existing_user = Accounts.query.filter((Accounts.name == username) | (Accounts.email == email)).first()
        if existing_user:
            flash("Username or email already taken. Please try again.")
            return redirect(url_for("register"))

        # Hash password and save user
        hashed_password = generate_password_hash(password)
        new_user = Accounts(name=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        session["user"] = username
        session["email"] = email
        flash("Registration successful!")
        return redirect(url_for("dashboard"))

    return render_template("register.html")
  

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = Accounts.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user"] = user.name
            session["email"] = user.email
            flash("Login successful!")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.")
            return redirect(url_for("login"))

    return render_template("login.html")



@app.route('/logout')
def logout():
      session.pop("user", None)
      session.pop("email", None)
      flash("You have been logged out.")
      return redirect(url_for("login"))
  
  
def detect_behaviour(video_file_path, model_file_path_1, model_file_path_2):
    # Load YOLOv11 model
    yolo_model = YOLO(model_file_path_1)

    # Load secondary behavior classification model (PyTorch)
    # behavior_model = torch.load(model_file_path_2, map_location=torch.device('cpu'))
    # behavior_model.eval()
    # Code replaced by ...
    

    # Open video file
    cap = cv2.VideoCapture(video_file_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Resize for performance (optional)
        resized_frame = cv2.resize(frame, (640, 360))

        # Run YOLOv11 inference
        results = yolo_model(resized_frame)[0]

        # Draw detections on frame
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(resized_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # (Optional) Add custom behavior classification here:
        # You can crop person ROIs, generate pose keypoints, etc., and run through behavior_model

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        if not ret:
            continue

        # Yield the frame for streaming
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
  
  
def generate_thumbnails(video_folder='static/videos', thumbnail_folder='static/thumbnails'):
    os.makedirs(thumbnail_folder, exist_ok=True)

    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, filename)
            thumbnail_path = os.path.join(thumbnail_folder, filename.replace('.mp4', '.jpg'))

            # Skip if thumbnail already exists
            if os.path.exists(thumbnail_path):
                continue

            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(thumbnail_path, frame)
            cap.release()


@app.route('/Fotp')
def Fotp():
    return render_template('Fotp.html')  


@app.route('/projectinfo')
def projectinfo():
    return render_template('projectinfo.html')  


@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")
    # if "user" in session:
    #     return render_template("dashboard.html", user=session["user"])
    # else:
    #     flash("Please log in first.")
    #     return redirect(url_for("login"))

@app.route('/notifications')
def notifications():
    return render_template('notifications.html')

@app.route('/footages')
def footages():
    video_dir = os.path.join("static", "videos")
    videos = [v for v in os.listdir(video_dir) if v.endswith(".mp4")]
    return render_template("footages.html", videos=videos)


# Route to serve video stream


@app.route('/video_feed/<video_name>')
def video_feed(video_name):
    try:
        # Check for special video separately
        if video_name == "videoplayback_1.mp4":
            video_path = os.path.join('static', 'featured', video_name)
        else:
            video_path = os.path.join('static', 'videos', video_name)

        print("Looking for:", video_path)

        if not os.path.exists(video_path):
            return f"Video file {video_name} not found at {video_path}", 404

        model_file_path_1 = os.path.join(os.getcwd(), 'yolo11x-pose.pt')
        model_file_path_2 = os.path.join(os.getcwd(), 'uit_model_final.pth')

        return Response(
            detect_behaviour(
                video_file_path=video_path,
                model_file_path_1=model_file_path_1,
                model_file_path_2=model_file_path_2
            ),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    except Exception as e:
        print("Error in video_feed:", e)
        return f"Internal Server Error: {str(e)}", 500




# Try this ...
# @app.route('/video_feed/<video_name>')
# def video_feed(video_name):
#     return f"You requested: {video_name}"



# @app.route('/video_feed/<video_name>')
# def video_feed(video_name):
#     try:
#         video_path = os.path.join('static', 'videos', video_name)

#         if not os.path.exists(video_path):
#             return f"Video file {video_name} not found.", 404

#         model_file_path_1 = os.path.join(os.getcwd(), 'yolo11x-pose.pt')
#         model_file_path_2 = os.path.join(os.getcwd(), 'uit_model_final.pth')

#         return Response(
#             detect_behaviour(
#                 video_file_path=video_path,
#                 model_file_path_1=model_file_path_1,
#                 model_file_path_2=model_file_path_2
#             ),
#             mimetype='multipart/x-mixed-replace; boundary=frame'
#         )

#     except Exception as e:
#         print(f"[ERROR] {e}")
#         return f"Internal Server Error: {str(e)}", 500
    
    
    

# @app.route('/video_feed')
# def video_feed():
#     try:
#         output_dir = os.path.join(os.getcwd(), 'output')
#         if os.path.exists(output_dir):
#             shutil.rmtree(output_dir)
        
#         upload_dir = os.path.join(os.getcwd(), 'upload')
#         if os.path.exists(upload_dir):
#             shutil.rmtree(upload_dir)

        
#         return Response(detect_behaviour(
#                 video_file_path=r"C:\Users\Piyali\Behaviour-Analysis-Project\static\videos\videoplayback (1).mp4", 
#                 model_file_path_1=r"C:\Users\Piyali\Behaviour-Analysis-Project\yolo11x-pose.pt",
#                 model_file_path_2=r"C:\Users\Piyali\Behaviour-Analysis-Project\uit_model_final.pth"
#             ),
#                 mimetype='multipart/x-mixed-replace; boundary=frame') 

#     except Exception as e:
#         print(f"Error occurred: {e}")
#         sys.exit(1)

if __name__=="__main__":
    with app.app_context():
     db.create_all()

    generate_thumbnails()
    app.run(debug=True)
