from abc import ABC, abstractmethod
from pathlib import Path
from typing import Text, Optional, AnyStr, Dict, Any, Union

from reactivex.subject import Subject
from typing import NamedTuple

from . import utils

#Replace with socketio server
WebsocketServer = NamedTuple("WebsocketServer", [("host", Text), ("port", int)])


class AudioSource(ABC):
    """Represents a source of audio that can start streaming via the `stream` property.

    Parameters
    ----------
    uri: Text
        Unique identifier of the audio source.
    sample_rate: int
        Sample rate of the audio source.
    """

    def __init__(self, uri: str, sample_rate: int):
        self.uri = uri
        self.sample_rate = sample_rate
        self.stream = Subject()

    @property
    def duration(self) -> Optional[float]:
        """The duration of the stream if known. Defaults to None (unknown duration)."""
        return None

    @abstractmethod
    def read(self):
        """Start reading the source and yielding samples through the stream."""
        pass

    @abstractmethod
    def close(self):
        """Stop reading the source and close all open streams."""
        pass


class WebSocketAudioSource(AudioSource):
    """Represents a source of audio coming from the network using the WebSocket protocol.

    Parameters
    ----------
    sample_rate: int
        Sample rate of the chunks emitted.
    host: Text
        The host to run the websocket server.
        Defaults to 127.0.0.1.
    port: int
        The port to run the websocket server.
        Defaults to 7007.
    key: Text | Path | None
        Path to a key if using SSL.
        Defaults to no key.
    certificate: Text | Path | None
        Path to a certificate if using SSL.
        Defaults to no certificate.
    """

    def __init__(
        self,
        sample_rate: int,
        host: str = "127.0.0.1",
        port: int = 7007,
        key: Optional[Union[Text, Path]] = None,
        certificate: Optional[Union[Text, Path]] = None,
    ):
        # FIXME sample_rate is not being used, this can be confusing and lead to incompatibilities.
        #  I would prefer the client to send a JSON with data and sample rate, then resample if needed
        super().__init__(f"{host}:{port}", sample_rate)
        self.client: Optional[Dict[Text, Any]] = None
        self.server = WebsocketServer(host, port, key=key, cert=certificate)
        self.server.set_fn_message_received(self._on_message_received)

    def _on_message_received(
        self,
        client: Dict[Text, Any],
        server: WebsocketServer,
        message: AnyStr,
    ):
        # Only one client at a time is allowed
        if self.client is None or self.client["id"] != client["id"]:
            self.client = client
        # Send decoded audio to pipeline
        self.stream.on_next(utils.decode_audio(message))

    def read(self):
        """Starts running the websocket server and listening for audio chunks"""
        self.server.run_forever()

    def close(self):
        """Close the websocket server"""
        if self.server is not None:
            self.stream.on_completed()
            self.server.shutdown_gracefully()

    def send(self, message: AnyStr):
        """Send a message through the current websocket.

        Parameters
        ----------
        message: AnyStr
            Bytes or string to send.
        """
        if len(message) > 0:
            self.server.send_message(self.client, message)
