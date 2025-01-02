import warnings
warnings.filterwarnings("ignore")

import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection, VitsModel, AutoTokenizer
import torch
import gradio as gr
import threading
import time
import sounddevice as sd
import numpy as np

# Load object detection model and processor from local directory
save_directory = "./local_model"
processor = AutoImageProcessor.from_pretrained(save_directory)
model = AutoModelForObjectDetection.from_pretrained(save_directory)

# Load the TTS model and tokenizer from the local directory
tts_model_path = "./mms-tts-eng/model"
tts_model = VitsModel.from_pretrained(tts_model_path)
tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_path)

stop_event = threading.Event()
frame = None

def live_camera_feed(camera_index):
    global frame
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 640)
    cap.set(4, 480)

    while not stop_event.is_set():
        success, img = cap.read()
        if success:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    cap.release()

def detect_objects():
    global frame
    while not stop_event.is_set():
        if frame is not None:
            img = frame.copy()
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([img.shape[:2]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

            detected_objects = []

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.5 and len(detected_objects) < 2:  # Only consider detections with score > 0.5 and limit to 2 objects
                    box = [int(i) for i in box.tolist()]
                    label_name = model.config.id2label[label.item()]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(img, f"{label_name} {int(score * 100)}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_objects.append(f"{label_name} {int(score * 100)}%")

            if detected_objects:
                # Convert detected objects to text and generate speech
                for obj in detected_objects:
                    tts_output = text_to_speech(obj)
                    play_audio(tts_output)
                    time.sleep(1)  # Delay of 1 second between TTS outputs

            yield img, detected_objects

            # Clear cache and memory
            del inputs, outputs, results, img
            torch.cuda.empty_cache()
            time.sleep(2.5)  # Delay of 2.5 seconds

def text_to_speech(text):
    # Tokenize the input text
    inputs = tts_tokenizer(text, return_tensors="pt")

    # Generate the speech waveform
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    return output

def play_audio(output):
    # Ensure the audio data has the correct shape and number of channels
    audio_data = output.squeeze().numpy()
    if audio_data.ndim == 1:
        audio_data = np.expand_dims(audio_data, axis=1)
    sd.play(audio_data, samplerate=22050)
    sd.wait()  # Wait until the audio is finished playing

def start_detection(camera_option):
    stop_event.clear()
    camera_index = int(camera_option.split()[-1])
    threading.Thread(target=live_camera_feed, args=(camera_index,)).start()
    return detect_objects()

def stop_detection():
    stop_event.set()

camera_options = ["Camera 0", "Camera 1", "Camera 2"]

with gr.Blocks() as iface:
    gr.Markdown("# Object Detection with Webcam and Text-to-Speech")
    camera_option = gr.Dropdown(choices=camera_options, label="Select Camera", value="Camera 0")
    start_button = gr.Button("Start Detection")
    stop_button = gr.Button("Stop Detection")
    live_output = gr.Image(type="numpy", label="Live Webcam")
    detection_output = gr.Textbox(label="Detection Results")

    def update_output(camera_option):
        for frame, detected_objects in start_detection(camera_option):
            yield frame, "\n".join(detected_objects)

    start_button.click(update_output, inputs=camera_option, outputs=[live_output, detection_output])
    stop_button.click(stop_detection)

iface.launch(share=True)
