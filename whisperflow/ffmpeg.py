import ffmpeg
import numpy as np
import asyncio

"""
  "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", temp_webm_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-"

"""

class FFmpegAudioStream:
    def __init__(self, sample_rate=16000, channels=1):
        # Initialize the sample rate and channels for the PCM output
        self.sample_rate = sample_rate
        self.channels = channels
        self.process = None

    async def start_ffmpeg_process(self):
        # Start an ffmpeg process for continuous streaming
        self.process = (
            ffmpeg
            .input('pipe:0', format='webm')
            .output(
                'pipe:1',
                format='s16le',
                acodec='pcm_s16le', 
                ac=1,
                ar=16000
            )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
        )

    
    async def read_stdout(self):
        while True:
            chunk = self.process.stdout.read()
            print("process_chunk read1", len(chunk))
            if not chunk:
                break
            yield chunk

    async def process_chunk(self, chunk: bytes) -> bytes:
        input_length = len(chunk)
        if self.process is None:
            await self.start_ffmpeg_process()
        
        # Write the chunk to ffmpeg's stdin
        print("process_chunk writing stdin")
        self.process.stdin.write(chunk)
        await asyncio.sleep(0)  # Yield control to allow other tasks to run

        # Read all available bytes from stdout
        print("process_chunk reading stdout")
        pcm_chunks = []
        while True:
            chunk = self.process.stdout.read()
            print("process_chunk read1", len(chunk))
            if not chunk:
                break
            pcm_chunks.append(chunk)
        pcm_data = b''.join(pcm_chunks)

        # Convert bytes to numpy array (PCM format)
        # pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        print("process_chunk processed finished", input_length, len(pcm_data))
        return pcm_data

    async def close(self):
        if self.process:
            # Close ffmpeg process properly
            self.process.stdin.close()
            await asyncio.sleep(0)  # Yield to ensure closure
            self.process.wait()
            self.process = None