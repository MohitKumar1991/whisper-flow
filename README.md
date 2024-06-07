<div align="center">
<h1 align="center"> Whisper Flow </h1> 
<h3>Real-Time Transcription Using OpenAI Whisper</br></h3>
<img src="https://img.shields.io/badge/Progress-1%25-red"> <img src="https://img.shields.io/badge/Feedback-Welcome-green">
</br>
</br>
<kbd>
<img src="/docs/imgs/whisper-flow.png" width="256px"> 
</kbd>
</div>


## About The Project

### OpenAI Whisper 
OpenAI [Whisper](https://github.com/openai/whisper) is a versatile speech recognition model designed for general use. Trained on a vast and varied audio dataset, Whisper can handle tasks such as multilingual speech recognition, speech translation, and language identification. It is commonly used for batch transcription, where you provide the entire audio or video file to Whisper, which then converts the speech into text. This process is not done in real-time; instead, Whisper processes the files and returns the text afterward, similar to handing over a recording and receiving the transcript later.

### Whisper Flow 
Using Whisper Flow, you can generate real-time transcriptions for your media content. Unlike batch transcriptions, where media files are uploaded and processed, streaming media is delivered to Whisper Flow in real time, and the service returns a transcript immediately.

### What is Streaming
Streaming content is sent as a series of sequential data packets, or 'chunks,' which Whisper Flow transcribes on the spot. The benefits of using streaming over batch processing include the ability to incorporate real-time speech-to-text functionality into your applications and achieving faster transcription times. However, this speed may come at the expense of accuracy in some cases.

### Stream Windowing
In scenarios involving time-streaming, it's typical to perform operations on data within specific time frames known as temporal windows. One common approach is using the [tumbling window](https://learn.microsoft.com/en-us/azure/stream-analytics/stream-analytics-window-functions#tumbling-window) technique, which involves gathering events into segments until a certain condition is met.

<div align="center">
<img src="/docs/imgs/streaming.png"> 
<div>Tumbling Window</div>
</div><br/>

### Streaming Results
Whisper Flow splits the audio stream into segments based on natural speech patterns, like speaker changes or pauses. The transcription is sent back as a series of events, with each response containing more transcribed speech until the entire segment is complete.

| Transcript                                    | EndTime  | IsPartial |
| :-------------------------------------------- | :------: | --------: |
| Reality                                       |   0.55   | True      |
| Reality is                                    |   1.15   | True      |
| Reality is created                            |   1.50   | True      |
| Reality is created by the                     |   2.10   | True      |
| Reality is created by the mind                |   2.65   | True      |
| Reality is created by the mind                |   3.15   | False     |
| we can                                        |   3.55   | True      |
| we can change                                 |   3.85   | True      |
| we can change reality                         |   4.40   | True      |
| we can change reality by changing             |   5.05   | True      |
| we can change reality by changing our mind    |   5.55   | True      |
| we can change reality by changing our mind    |   6.15   | False     |

### Benchmarking
The evaluation metrics used for comparing the performance of Whisper Flow are Word Error Rate (WER) and latency. Latency is measured as the time between two subsequent partial results, with the goal of achieving sub-second latency. We are not starting from scratch; several quality benchmarks have already been performed for different ASR engines. I will rely on the research article ["Benchmarking Open Source and Paid Services for Speech to Text"](https://www.frontiersin.org/articles/10.3389/fdata.2023.1210559/full) for guidance.

## How To Use it


