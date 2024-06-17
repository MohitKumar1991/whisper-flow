""" fast api declaration """

import logging
import asyncio

from queue import Queue
from typing import List
from fastapi import FastAPI, WebSocket, Form, File, UploadFile

import whisperflow.streaming as st
import whisperflow.transcriber as ts

VERSION = "0.0.1"
app = FastAPI()


@app.get("/health", response_model=str)
def health():
    """health function on API"""
    return f"Whisper Flow V{VERSION}"


@app.post("/transcribe_pcm_chunk", response_model=dict)
def transcribe_pcm_chunk(
    model_name: str = Form(...), files: List[UploadFile] = File(...)
):
    """transcribe chunk"""
    model = ts.get_model(model_name)
    content = files[0].file.read()
    return ts.transcribe_pcm_chunks(model, [content])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """webscoket implementation"""
    model = ts.get_model()

    async def transcribe_chunks(chunks: list) -> dict:
        task = asyncio.create_task(ts.transcribe_pcm_chunks(model, chunks))
        return await task

    def segment_closed(websocket: WebSocket):
        async def send_back(data: dict):
            await websocket.send_json(data)

        return send_back

    task = None

    try:
        await websocket.accept()
        queue, should_stop = Queue(), [False]

        segment_closed_callback = segment_closed(websocket)
        task = asyncio.create_task(
            st.transcribe(
                should_stop, queue, transcribe_chunks, segment_closed_callback
            )
        )

        while True:
            data = await websocket.receive_bytes()
            queue.put(data)
    except Exception as exception:  # pylint: disable=broad-except
        logging.error(exception)
        should_stop = [True]
        if task:
            await task
        await websocket.close()
