from transformers import VitsModel, AutoTokenizer
import torch
import torchaudio
import sounddevice as sd

# Load the model and tokenizer from the local directory
model_path = "./model"
model = VitsModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the text to be converted to speech
text = "person"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate the speech waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Play the audio immediately
sd.play(output.numpy(), samplerate=22050)
sd.wait()  # Wait until the audio is finished playing