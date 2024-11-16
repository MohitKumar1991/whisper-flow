""" fast api declaration """

import tempfile
import os
import io
import time

import logging
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import socketio
import asyncio
import json
from pydub import AudioSegment

from whisperflow import __version__
import whisperflow.transcriber as ts
from whisperflow.transcriber import transcribe_pcm_chunks_async
from .diart.sources import AudioSource
from reactivex import Observer
from reactivex.scheduler.eventloop import AsyncIOThreadSafeScheduler
import reactivex as rx
from queue import Queue
from reactivex import operators as ops
from .ffmpeg import FFmpegAudioStream


class PrintObserver(Observer):
    def on_next(self, value):
        print("PObserver on_next")
        print(len(value))
    
    def on_error(self, error):
        print(f"Error: {error}")
        
    def on_completed(self):
        print("Completed")

# Initialize Socket.IO with the correct configuration
sio = socketio.AsyncServer(
    cors_allowed_origins='*',
    async_mode='asgi',
    engineio_logger=True,
    logger=True
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create ASGI application
socket_app = socketio.ASGIApp(sio)
app.mount("/", socket_app)

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
        self.audio_chunks = Queue()
        self.transcribe_callback = transcribe_callback
        self.send_callback = send_callback
        self.connection_id = connection_id
        self.scheduler = AsyncIOThreadSafeScheduler(asyncio.get_event_loop())
        self.ffmpeg_stream = FFmpegAudioStream()

    def add_chunk(self, chunk):
        #this is thread safe - so you can put chunks on any thread
        if chunk is None:
            print("add_chunk None")
            return
        # self.audio_chunks.put(chunk)
        self.stream.on_next(chunk)

    async def process_audio(self, chunks):
        print("process_audio running")
        if len(chunks) == 0:
            print("no chunks to process")
            return
        
        # Process chunks through FFmpeg stream
        pcm_chunks = []
        for chunk in chunks:
            pcm_data = await self.ffmpeg_stream.process_chunk(chunk)
            pcm_chunks.append(pcm_data)
        
        result = await self.transcribe_callback(pcm_chunks)
        await self.send_callback(self.connection_id, result)
    
    #run in main thread
    def attach_observer(self, observer):
        self.stream.subscribe(observer, scheduler=self.scheduler)
        print("attach_observer", self.stream.observers)

    #this is a blocking function which must be run in a separate thread with run_in_executor - it does nothing but block
    def read(self):
        while self.connection_id is not None:
            time.sleep(0.1)
    
    # async def read_async(self):
    #     while self.connection_id is not None:
    #         try:
    #             # Sleep briefly to avoid busy waiting
    #             await asyncio.sleep(0.3)
    #             print("read running after sleep")
                
    #             # Process any available chunks
    #             if not self.audio_chunks.empty():
    #                 chunks = []
    #                 while not self.audio_chunks.empty():
    #                     chunks.append(self.audio_chunks.get())
                    
    #                 combined = b''.join(chunks)
    #                 self.stream.on_next([combined])
    #         except Exception as e:
    #             logging.error(f"Error processing audio chunks: {e}")

    def close(self):
        """Close the websocket server"""
        if self.connection_id is not None:
            self.stream.on_completed()
            asyncio.create_task(self.ffmpeg_stream.close())
            self.connection_id = None

    

@sio.event
def connect(sid, environ):
    """Handle new Socket.IO connection"""
    logging.info(f"Client connected: {sid}")
    audio_session_source = AudioSession(transcribe_async, send_back_async, connection_id=sid)
    sessions[sid] = audio_session_source
    #this is supposed to do diarization
    audio_session_source.attach_observer(PrintObserver())
    
    #this is supposed to do transcription - now with buffering
    audio_session_source.stream.pipe(
        ops.buffer_with_count(count=3)  # Buffer 3 chunks before emitting
    ).subscribe(lambda chunks: asyncio.create_task(audio_session_source.process_audio(chunks)))

    print("observers", audio_session_source.stream.observers)

    # asyncio.create_task(audio_session_source.read_async())

@sio.event
def disconnect(sid):
    """Handle Socket.IO disconnection"""
    logging.info(f"Client disconnected: {sid}")
    if sid in sessions:
        # asyncio.create_task(sessions[sid].process_audio())
        del sessions[sid]

@sio.event
def audio_data(sid, data: bytes):
    """Handle binary audio data"""
    print(f"audio_data {sid}: {type(data)} {len(data)}")
    if sid in sessions:
        sessions[sid].add_chunk(data)

@sio.event
def stop_recording(sid, data):
    """Handle stop recording command"""
    print(f"stop_recording {sid}: {data}")
    if sid in sessions:
        sessions[sid].close()

@sio.event
def command(sid, data):
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
    """Sync transcription function"""
    model = ts.get_model()
    return await transcribe_pcm_chunks_async(model, chunks)

async def send_back_async(sid: str, data: dict):
    """Send results back to client asynchronously"""
    await sio.emit('transcription_result', data, to=sid)

def start_server(host="localhost", port=8181, reload=False):
    """Start the server using uvicorn"""
    import uvicorn
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run WhisperFlow FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8181, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    start_server(args.host, args.port, args.reload)
