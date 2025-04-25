# from flask import Flask, Blueprint, request, render_template, jsonify, redirect, url_for
# from flask_bcrypt import Bcrypt
# from flask_pymongo import PyMongo

# app = Flask(__name__)

# # Initialize Flask-Bcrypt
# bcrypt = Bcrypt(app)

# # Set up MongoDB client and database
# app.config["MONGO_URI"] = "mongodb+srv://piyalimondal2003pm:piyali2003@piyali.b8jte.mongodb.net/?retryWrites=true&w=majority&appName=Piyali"
# mongo = PyMongo(app)
# db = mongo.db  # This is the database from the PyMongo instance
# users_collection = db.users  # The collection where user data will be stored

# # Create a Blueprint for authentication routes
# auth_bp = Blueprint('auth', __name__)

# # Route for registration page (GET method renders the HTML)
# @auth_bp.route('/sign_up', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         # Extract username, email, and password from the form data
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']

#         # Check if the user already exists (based on the username or email)
#         existing_user = users_collection.find_one({'username': username})
#         if existing_user:
#             return render_template('register.html', message="User already exists!")

#         # Hash the password before saving it to the database
#         hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

#         # Insert the new user with the hashed password into MongoDB
#         users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})

#         # Optionally, you can redirect the user to the login page after successful registration
#         return redirect(url_for('auth.login'))  # Redirect to login page after registration

#     return render_template('register.html')  # GET request renders the registration page

# # Route for login page (GET method renders the HTML)
# @auth_bp.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         # Extract username and password from the form data
#         username = request.form['username']
#         password = request.form['password']

#         # Retrieve the user from the database
#         user = users_collection.find_one({'username': username})
#         if not user:
#             return render_template('login.html', message="Invalid credentials")

#         # Check if the provided password matches the stored hashed password
#         if bcrypt.check_password_hash(user['password'], password):
#             return redirect(url_for('dashboard'))  # Redirect to the dashboard on successful login
#         else:
#             return render_template('login.html', message="Invalid credentials")

#     return render_template('login.html')  # GET request renders the login page

# # Route to test the connection to MongoDB
# @auth_bp.route('/test_db', methods=['GET'])
# def test_db():
#     try:
#         # Check if the MongoDB connection is working by doing a simple command
#         db.command('ping')
#         return jsonify({"message": "MongoDB connection is successful!"}), 200
#     except Exception as e:
#         return jsonify({"message": f"MongoDB connection failed: {str(e)}"}), 500

# # Register Blueprint with the main Flask app
# app.register_blueprint(auth_bp, url_prefix='/auth')

# @app.route('/dashboard')
# def dashboard():
#     return "Welcome to the dashboard!"  # Replace with a real dashboard template or content

# if __name__ == '__main__':
#     app.run(debug=True)
