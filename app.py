from flask import Flask, Response, render_template
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp

# Inisialisasi Flask
app = Flask(__name__)

# Load model yang sudah dilatih
MODEL_PATH = 'rock_paper_scissors_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Label kelas
CLASS_NAMES = ['Kertas', 'Batu', 'Gunting']

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk prediksi dengan bounding box
def predict_with_bounding_box(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Tambahkan margin pada bounding box
            margin = 20  # Ukuran margin dalam piksel
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            # Potong dan proses gambar
            cropped_img = frame[y_min:y_max, x_min:x_max]
            if cropped_img.size > 0:
                resized_img = cv2.resize(cropped_img, (224, 224))
                img_array = resized_img / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi kelas
                predictions = model.predict(img_array)
                predicted_class = CLASS_NAMES[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

                # Gambarkan bounding box dan hasil prediksi pada frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{predicted_class} ({confidence:.2f}%)",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
                break  # Hanya satu tangan diproses
    return frame

# Fungsi untuk streaming video real-time
def generate_frames():
    cap = cv2.VideoCapture(0)  # Buka kamera (0 untuk default webcam)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Proses frame untuk prediksi
        frame = predict_with_bounding_box(frame)

        # Encode frame sebagai JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Kirimkan frame dalam format HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index2.html')

# Route untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Menjalankan server Flask
if __name__ == '__main__':
    app.run(debug=True)
