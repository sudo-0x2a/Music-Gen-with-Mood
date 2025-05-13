---
author: sudo-0x2a
completion_data: 04/14/2025
---

# Generate AI Music Based on Your Mood
A refined version of one of my school projects. Used a sentiment analysis NLP model to extract the top 2 emotions from the user input. Then, pass the emotion tags to the prompt for music generation model to compose a 15-second melody.

## Example Output
```
Express your mood: I'm so happy that you guys came to my birthday party!

Generating music that expresses joy and excitement...
Music generation complete!
```
[Listen to the example audio](example_output.wav)

## Explaination
Models used in this project:
- NLP: [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- text-audio: [facebook/musicgen-small](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)

## Tested Setup
- Hardware: Macbook Pro 14'' M4 pro - 12C16G with 24GB RAM
- python version: 3.11.8 
*Highly recommend to run it with a Nvidia GPU or Apple Scilicon*
