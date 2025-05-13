import os
import torch
import scipy
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Assign device to CUDA, MPS, or CPU
def device_setup():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


# Use a NLP model to analyze emotion
def emotion_analysis(sentence:str):
    classifier = pipeline(
        task="text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        framework='pt')
    analysis = classifier(sentence)
    return([analysis[0][0]['label'], analysis[0][1]['label']])


# Setup MusicGen for genrating audio
def music_gen(device:torch.device, emotion:str):
    print(f"\nGenerating music that expresses {emotion[0]} and {emotion[1]}...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", attn_implementation="eager")
    model.to(device)
    prompt = f"a melody to express {emotion[0]} and {emotion[1]} with a piano, a guitar, a bass, and a drum"

    inputs = processor(
            text=prompt,
            padding=True,
            return_tensors="pt").to(device)
    # Generate a 15 sec. melody
    audio_values = model.generate(**inputs, max_new_tokens=768)
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("emotion_music.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
    print("Music generation complete!")


if __name__ == "__main__":
    try:
        sentence = input("Express your mood: ")
    except:
        raise("Invalid Input")
    emotion = emotion_analysis(sentence)
    music_gen(device_setup(), emotion)