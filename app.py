import warnings
warnings.filterwarnings("ignore")

import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection, VitsModel, AutoTokenizer
import torch
import gradio as gr
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

def live_camera_feed(camera_index):
    cap = cv2.VideoCapture(camera_index)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if success:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            yield frame
        else:
            break

    cap.release()

def detect_objects(frame):
    inputs = processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

def generate_audio(text):
    inputs = tts_tokenizer(text, return_tensors="pt")
    audio = tts_model.generate(**inputs)
    return audio

def process_frame(frame):
    objects = detect_objects(frame)
    # Process detected objects and generate description text
    description = "Detected objects: " + ", ".join([obj['label'] for obj in objects])
    audio = generate_audio(description)
    return audio

def main():
    camera_index = 0
    for frame in live_camera_feed(camera_index):
        audio = process_frame(frame)
        # Play audio
        sd.play(audio, samplerate=22050)
        sd.wait()

if __name__ == "__main__":
    main()