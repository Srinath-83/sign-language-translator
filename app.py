from flask import Flask, render_template, request, redirect, url_for, Response, session
import pyttsx3
import time
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import random
import string

app = Flask(__name__)
app.secret_key = "your_secret_key"  # change to a secure secret key

# Load models and label encoder
print("[INFO] Loading letter model...")
letter_model = load_model("model/asl_landmark_model.h5")
print("[INFO] Letter model loaded.")

print("[INFO] Loading label encoder...")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
print("[INFO] Label encoder loaded.")

print("[INFO] Loading phrase model...")
phrase_model = load_model("model/phrase_model.h5")
print("[INFO] Phrase model loaded.")

current_model_mode = 'letter'  # global model mode variable

phrase_labels = [
    "again", "bad", "bathroom", "book", "busy", "do not want", "eat", "father",
    "fine", "finish", "forget", "go", "good", "happy", "hello", "help", "how", "i",
    "learn", "like", "meet", "milk", "more", "mother", "my", "name", "need", "nice",
    "no", "please", "question", "right", "sad", "same", "see you letter", "thank you",
    "want", "what", "when", "where", "which", "who", "why", "wrong", "yes", "you", "your"
]

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

sentence = ""
last_capture_time = time.time()
capture_interval = 2
current_letter = random.choice(string.ascii_uppercase)
feedback_msg = ""

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == "Harisiddarth" and password == "Siddarth":
        session['user'] = username
        session['mode'] = current_model_mode
        return redirect('/home')
    else:
        return render_template('login.html', error="Invalid credentials")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/chart')
def chart():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('chart.html')

@app.route('/realtime')
def realtime():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/switch_mode/<mode>')
def switch_mode(mode):
    global current_model_mode
    if mode in ['letter', 'phrase']:
        current_model_mode = mode
        session['mode'] = mode
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    mode = session.get('mode', 'letter')
    return Response(generate_frames(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/speak')
def speak_sentence():
    global sentence
    if sentence.strip():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1)
            engine.say(sentence)
            engine.runAndWait()
        except Exception as e:
            print("[ERROR] TTS Error:", e)
    sentence = ""
    return redirect('/realtime')

@app.route('/clear')
def clear_sentence():
    global sentence
    sentence = ""
    return redirect('/realtime')

@app.route('/practice')
def practice():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('practice.html', letter=current_letter, feedback=feedback_msg)

@app.route('/next_letter')
def next_letter():
    global current_letter, feedback_msg
    current_letter = random.choice(string.ascii_uppercase)
    feedback_msg = ""
    return redirect(url_for('practice'))

@app.route('/practice_feed')
def practice_feed():
    return Response(generate_practice_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames(mode):
    global sentence, last_capture_time
    cap = cv2.VideoCapture(0)
    phrase_sequence = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        data_aux, x_, y_ = [], [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

        if time.time() - last_capture_time >= capture_interval:
            last_capture_time = time.time()
            if hand_detected:
                data_aux = np.pad(data_aux, (0, 42 - len(data_aux)), mode='constant')[:42]
                if mode == 'phrase':
                    phrase_sequence.append(data_aux)
                    if len(phrase_sequence) == 30:
                        input_data = np.array([phrase_sequence])
                        prediction = phrase_model.predict(input_data)
                        predicted_class = np.argmax(prediction)
                        predicted_phrase = phrase_labels[predicted_class] if predicted_class < len(phrase_labels) else "Unknown"
                        sentence += " " + predicted_phrase
                        phrase_sequence = []
                else:
                    prediction = letter_model.predict(np.array([data_aux]))
                    predicted_class = np.argmax(prediction)
                    predicted_character = label_encoder.inverse_transform([predicted_class])[0]
                    sentence += predicted_character
            else:
                if mode == 'letter':
                    sentence += " "

        cv2.putText(frame, f"Mode: {mode}", ((frame.shape[1] - 400) // 2, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (56, 124, 68), 3)
        cv2.putText(frame, f"Sentence: {sentence}", ((frame.shape[1] - 600) // 2, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (56, 124, 68), 3)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def generate_practice_frames():
    global last_capture_time, current_letter, feedback_msg
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        data_aux, x_, y_ = [], [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            data_aux = np.pad(data_aux, (0, 42 - len(data_aux)), mode='constant')[:42]

            if time.time() - last_capture_time >= capture_interval:
                last_capture_time = time.time()
                prediction = letter_model.predict(np.array([data_aux]))
                predicted_class = np.argmax(prediction)
                predicted_character = label_encoder.inverse_transform([predicted_class])[0]

                if predicted_character.upper() == current_letter.upper():
                    feedback_msg = f"✅ Great! You signed '{current_letter}' correctly."
                else:
                    feedback_msg = f"❌ Try Again! You signed '{predicted_character}', but target is '{current_letter}'."

        cv2.putText(frame, f"Sign: {current_letter}", ((frame.shape[1] - 400) // 2, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(frame, f"Feedback: {feedback_msg}", ((frame.shape[1] - 600) // 2, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 215, 0), 3)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(debug=True)