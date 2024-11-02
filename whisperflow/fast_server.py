""" fast api declaration """

import tempfile
import os
from collections import deque
import io
import numpy as np
import time

import logging
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import socketio
import json
from pydub import AudioSegment

from whisperflow import __version__
import whisperflow.transcriber as ts
from .diart.sources import AudioSource


def convert_webm_bytes_to_wav_bytes(webm_bytes, sample_rate=16000):
    # Create a temporary file to store the input webm data
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
        temp_webm.write(webm_bytes)
        temp_webm_path = temp_webm.name

    try:
        # Load using pydub
        audio = AudioSegment.from_file(temp_webm_path, format="webm")
        
        # Convert to desired format
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(1)  # Convert to mono
        
        # Export to WAV bytes
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()

    finally:
        # Clean up temporary file
        os.unlink(temp_webm_path)

# Initialize Socket.IO with the correct configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    engineio_logger=True,
    logger=True
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wrap with ASGI application
application = socketio.ASGIApp(sio, app)

sessions = {}


@app.get("/health", response_model=str)
def health():
    """health function on API"""
    return f"Whisper Flow V{__version__}"


@app.post("/transcribe_pcm_chunk", response_model=dict)
def transcribe_pcm_chunk(
    files: List[UploadFile] = File(...)
):
    """transcribe chunk"""
    model = ts.get_model()
    content = files[0].file.read()
    return ts.transcribe_pcm_chunks(model, [content])


class AudioSession(AudioSource):
    def __init__(self, transcribe_callback, send_callback, connection_id):
        super().__init__("websocket", 16000)
        self.audio_chunks = deque()
        self.transcribe_callback = transcribe_callback
        self.send_callback = send_callback
        self.connection_id = connection_id
        
        
    async def add_chunk(self, chunk):
        self.audio_chunks.append(chunk)
        # byte_samples = base64.decodebytes(data.encode("utf-8"))
        if self.connection_id is not None:
            samples = np.frombuffer(chunk, dtype=np.float32)
            samples.reshape(1, -1)
            self.stream.on_next(samples)

    #TODO: this is supposed to start the server - this needs to block
    def read(self):
        while self.connection_id is not None:
            time.sleep(0.1)
    
    def close(self):
        """Close the websocket server"""
        if self.server is not None:
            self.stream.on_completed()
            self.connection_id = None

    async def process_audio(self):
        if not self.audio_chunks:
            return
            
        # Concatenate all chunks
        combined = b''.join(self.audio_chunks)
        
        # Create a temporary file for the input data
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
            temp_webm.write(combined)
            temp_webm_path = temp_webm.name

        try:
            from subprocess import run, CalledProcessError
            
            # Convert directly to PCM using ffmpeg
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads", "0",
                "-i", temp_webm_path,
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-"
            ]
            
            try:
                pcm_data = run(cmd, capture_output=True, check=True).stdout
            except CalledProcessError as e:
                logging.error(f"Failed to convert audio: {e.stderr.decode()}")
                return
                
            # Transcribe the PCM data directly
            #FIXME: instead of transcribe callback we will put this on the stream
            result = await self.transcribe_callback([pcm_data])
            await self.send_callback(result)
            
        finally:
            # Cleanup
            os.unlink(temp_webm_path)
        
        # Clear the queue after processing
        self.audio_chunks.clear()

@sio.event
async def connect(sid, environ):
    """Handle new Socket.IO connection"""
    logging.info(f"Client connected: {sid}")
    sessions[sid] = AudioSession(transcribe_async, lambda x: send_back_async(sid, x), connection_id=sid)

@sio.event
async def disconnect(sid):
    """Handle Socket.IO disconnection"""
    logging.info(f"Client disconnected: {sid}")
    if sid in sessions:
        await sessions[sid].process_audio()  # Process any remaining audio
        del sessions[sid]

@sio.event
async def audio_data(sid, data: bytes):
    """Handle binary audio data"""
    print(f"audio_data {sid}: {type(data)} {len(data)}")
    if sid in sessions:
        await sessions[sid].add_chunk(data)

@sio.event
async def stop_recording(sid, data):
    """Handle stop recording command"""
    print(f"stop_recording {sid}: {data}")
    if sid in sessions:
        await sessions[sid].process_audio()
        sessions[sid].close()

@sio.event
async def command(sid, data):
    """Handle text/JSON commands"""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        logging.info(f"Received command: {data}")
        # TODO: Implement command handling
        # Example: configuration updates, control commands, etc.
    except json.JSONDecodeError:
        logging.error("Invalid JSON received")

async def transcribe_async(chunks: list):
    """Async transcription function"""
    model = ts.get_model()
    return await ts.transcribe_pcm_chunks_async(model, chunks)

async def send_back_async(sid: str, data: dict):
    """Send results back to client"""
    await sio.emit('transcription_result', data, room=sid)

def start_server(host="localhost", port=8181, reload=False):
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run("whisperflow.fast_server:application", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WhisperFlow FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8181, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    start_server(args.host, args.port, args.reload)
