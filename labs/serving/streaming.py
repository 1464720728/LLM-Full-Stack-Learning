"""Incremental UTF-8 + SSE framing, independent of TCP chunk boundaries."""
import codecs
import json


class SSEDecoder:
    def __init__(self, max_buffer_bytes=65536):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.max_buffer_bytes = max_buffer_bytes
    def feed(self, chunk: bytes, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        # A CR at the end of a TCP chunk must remain pending until its LF arrives.
        self.buffer = self.buffer.replace("\r\n", "\n")
        result = []
        while "\n\n" in self.buffer:
            block, self.buffer = self.buffer.split("\n\n", 1)
            values = [line[5:].removeprefix(" ") for line in block.split("\n") if line.startswith("data:")]
            if values:
                result.append("\n".join(values))
        if len(self.buffer.encode("utf-8")) > self.max_buffer_bytes:
            raise ValueError("SSE event exceeds buffer limit")
        if final and self.buffer.strip():
            raise ValueError("truncated SSE event")
        return result
