from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="https://brr.sikkerai.dk/v1", api_key="sk-proj-1234567890")

with Path("audio.wav").open("rb") as audio_file:
    transcription = client.audio.transcriptions.create(model="syvai/faster-hviske-v3-conversation", file=audio_file, language="da")

print(transcription)