""" transcriber """

import os
import asyncio

import torch
import numpy as np

import whisper
from whisper import Whisper


models = {}


def get_model(file_name="tiny.en.pt") -> Whisper:
    """load models from disk"""
    if file_name not in models:
        path = os.path.join(os.path.dirname(__file__), f"./models/{file_name}")
        models[file_name] = whisper.load_model("base.en").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return models[file_name]


def transcribe_pcm_chunks(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks list"""
    print(f"transcriber sync {len(chunks)} chunks")
    try:
        # Debug the chunks content
        print(f"Chunks type: {type(chunks)}")
        print(f"First chunk type: {type(chunks[0]) if chunks else 'empty'}")
        print(f"Total bytes: {sum(len(chunk) for chunk in chunks)}")
        
        # Ensure the total length is even (required for int16)
        combined_chunks = b"".join(chunks)
        if len(combined_chunks) % 2 != 0:
            combined_chunks = combined_chunks[:-1]  # Remove last byte if odd
            
        arr = np.frombuffer(combined_chunks, np.int16).flatten().astype(np.float32) / 32768.0
        print(f"calling model with {len(arr)} samples")
        
        transcript = model.transcribe(
            arr,
            fp16=False,
            language=lang,
            logprob_threshold=log_prob,
            temperature=temperature,
        )
        print(f"Transcribed: {transcript}")
        return transcript
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise


async def transcribe_pcm_chunks_async(
    model: Whisper, chunks: list, lang="en", temperature=0.1, log_prob=-0.5
) -> dict:
    """transcribes pcm chunks async"""
    print(f"Transcribing {len(chunks)} chunks")
    return await asyncio.get_running_loop().run_in_executor(
        None, transcribe_pcm_chunks, model, chunks, lang, temperature, log_prob
    )
