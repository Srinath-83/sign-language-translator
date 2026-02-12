# Sign Language Translator (ML + Flask Web App)

A real-time Sign Language Translator built using Python, TensorFlow, OpenCV, and MediaPipe.  
The system detects hand gestures through a webcam and translates them into letters or predefined phrases using trained deep learning models.

The application runs through a Flask-based web interface with authentication, real-time video streaming, speech output, and practice mode.

---

## Features

-  Letter Recognition Model (ASL alphabet detection)
-  Phrase Recognition Model (predefined sentence detection)
-  Real-time webcam gesture detection
-  TensorFlow/Keras trained models
-  MediaPipe hand landmark detection
-  Text-to-Speech using pyttsx3
-  Live sentence formation
-  Practice Mode with feedback system
-  Login authentication system
-  Flask-based web interface (HTML + CSS)

---

## Technologies Used

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Pickle
- pyttsx3 (Text-to-Speech)
- HTML
- CSS

---

## Project Structure

```
sign-language-translator/
│
├── model/
│   ├── asl_landmark_model.h5
│   ├── phrase_model.h5
│   └── label_encoder.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── login.html
│   ├── home.html
│   ├── index.html
│   ├── chart.html
│   └── practice.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  How It Works

### Letter Mode
- Captures hand landmarks using MediaPipe
- Processes 42 landmark features
- Predicts ASL letter using trained Keras model
- Appends predicted letter to sentence

### Phrase Mode
- Collects 30 consecutive landmark frames
- Predicts phrase using phrase recognition model
- Appends predicted phrase to sentence

### Practice Mode
- Random target letter is generated
- User signs the letter
- System gives real-time feedback (Correct / Try Again)

### Text-to-Speech
- Converts generated sentence into speech using pyttsx3

---

## Model Details

- Letter Model: `asl_landmark_model.h5`
- Phrase Model: `phrase_model.h5`
- Label Encoder: `label_encoder.pkl`
- Input: 42 hand landmark features
- Detection Confidence: 0.7

---

## Installation & Setup

### Clone the Repository

```
git clone https://github.com/your-username/sign-language-translator.git
cd sign-language-translator
```

### Create Virtual Environment (Recommended)

```
python -m venv venv
venv\Scripts\activate   (Windows)
source venv/bin/activate (Mac/Linux)
```

### Install Dependencies

```
pip install -r requirements.txt
```

If needed, manually install:

```
pip install flask opencv-python mediapipe numpy tensorflow pyttsx3
```

### Run the Application

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## Future Enhancements

- Sentence auto-correction
- Continuous gesture detection
- Database for storing history
- Deployment on cloud (Heroku / Render / AWS)
- Multi-language sign support
- Improved model accuracy

---

## Author

Developed as a Machine Learning and Computer Vision project focused on accessibility and inclusive communication using deep learning and real-time gesture recognition.
