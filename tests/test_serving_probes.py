"""Offline probe contracts: local loopback fixture, no model/API/GPU."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import queue
import tempfile
import threading
import unittest

from labs.serving.concurrent_probe import run_bounded
from labs.serving.http_benchmark import load_prompt, measure


class ProbeTests(unittest.TestCase):
    def test_prompt_fixture_is_bounded_and_never_silently_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"prompt.txt"
            path.write_text("公开输入\n",encoding="utf-8")
            self.assertEqual(load_prompt(path),"公开输入\n")
            for raw in (b"",b"a"*65537,b"\xff"):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_prompt(path)
            path.write_bytes(b"a"*65536)
            self.assertEqual(len(load_prompt(path)),65536)

    def test_concurrency_errors_keep_identity_and_release_workers(self):
        barrier = threading.Barrier(2,timeout=5)
        lock = threading.Lock()
        state = {"active":0,"peak":0}
        def call(index):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"],state["active"])
            try:
                barrier.wait()
                if index == 2:
                    raise ValueError("injected")
                return {"value":index}
            finally:
                with lock:
                    state["active"] -= 1
        report = run_bounded(call,count=4,concurrency=2)
        self.assertEqual(state,{"active":0,"peak":2})
        self.assertEqual(report["successful"],3)
        self.assertEqual([row["index"] for row in report["rows"]],[0,1,2,3])
        self.assertEqual(report["rows"][2]["error_type"],"ValueError")
        for count,concurrency in ((0,1),(9,1),(True,1),(1,5)):
            with self.assertRaises(ValueError):
                run_bounded(lambda index:self.fail("invalid plan executed"),count,concurrency)

    def test_local_sse_probe_sends_exact_prompt_and_preserves_usage(self):
        received = queue.Queue()
        redirect_hits = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.put((self.path,payload))
                if payload["prompt"] == "redirect":
                    self.send_response(302)
                    self.send_header("Location",f"http://127.0.0.1:{self.server.server_port}/redirect-target")
                    self.send_header("Content-Length","0")
                    self.end_headers()
                    return
                events = [
                    {"choices":[{"text":"你"}]},
                    {"choices":[{"text":"好"}]},
                    {"choices":[],"usage":{"completion_tokens":2}},
                ]
                body = "".join("data: "+json.dumps(event,ensure_ascii=False)+"\n\n"
                               for event in events)+"data: [DONE]\n\n"
                if payload["prompt"] == "missing-terminal":
                    body = body.removesuffix("data: [DONE]\n\n")
                elif payload["prompt"] == "error-event":
                    body = 'data: {"error": "injected"}\n\ndata: [DONE]\n\n'
                raw = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type","text/event-stream")
                self.send_header("Content-Length",str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                redirect_hits.append(self.path)
                self.send_response(500)
                self.send_header("Content-Length","0")
                self.end_headers()
            def log_message(self,*args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1",0),Handler)
        worker = threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True)
        worker.start()
        try:
            result = measure(f"http://127.0.0.1:{server.server_port}/v1","fixture",
                             requests=1,max_tokens=3,prompt="exact public fixture")[0]
            path,payload = received.get(timeout=2)
            self.assertEqual(path,"/v1/completions")
            self.assertEqual(payload["prompt"],"exact public fixture")
            self.assertEqual(payload["max_tokens"],3)
            self.assertEqual(result["nonempty_chunks"],2)
            self.assertEqual(result["output_characters"],2)
            self.assertEqual(result["output_tokens_from_usage"],2)
            self.assertIsNotNone(result["ttft_text_ms"])
            for prompt in ("missing-terminal","error-event","redirect"):
                with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                    measure(f"http://127.0.0.1:{server.server_port}/v1","fixture",
                            requests=1,max_tokens=3,prompt=prompt)
            self.assertEqual(redirect_hits,[])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
