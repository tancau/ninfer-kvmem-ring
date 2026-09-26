"""WebUI shim: serves llama.cpp's static chat UI and proxies the API to NInfer.

Why this exists: NInfer has no built-in web UI, but the KVMem package (a llama.cpp
fork) ships llama.cpp's UI assets on disk. Those assets talk to llama.cpp endpoints
that NInfer does not serve. This shim closes the gap:

    GET  /               -> static files from the UI directory
    GET  /props          -> synthesized llama.cpp-shaped model properties
    GET  /health         -> {"status":"ok"}
    GET  /v1/models      -> proxied to NInfer
    POST /v1/*           -> proxied to NInfer, streaming (SSE) passed through
    POST /tokenize       -> minimal stub (the UI only uses it for a token count)

/props advertises endpoint_slots / endpoint_metrics / cors_proxy_enabled as false so
the UI does not call the llama.cpp endpoints that are not proxied.

Run:  python webui.py [--port 8080] [--upstream 127.0.0.1:8081] [--ui DIR]
"""
import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI_DIR = r"C:\Bonsai-KVMem\share\kvmem\ui"
MODEL_PATH = r"C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer"
CHAT_TEMPLATE = r"C:\Bonsai-App\chat-template-bonsai2.jinja"
MODEL_ID = "bonsai2-27b"

ARGS = None


def build_props():
    n_ctx = 163840
    try:
        with urllib.request.urlopen(
            "http://%s/v1/models" % ARGS.upstream, timeout=10
        ) as r:
            data = json.load(r)
        for m in data.get("data", []):
            if m.get("id") == MODEL_ID and m.get("max_model_len"):
                n_ctx = int(m["max_model_len"])
                break
    except Exception:
        pass

    template = ""
    try:
        with open(CHAT_TEMPLATE, encoding="utf-8") as fh:
            template = fh.read()
    except OSError:
        pass

    return {
        "default_generation_settings": {
            "params": {
                "n_predict": 32768,
                "max_tokens": 32768,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "seed": 0,
                "stop": [],
            },
            "n_ctx": n_ctx,
        },
        "total_slots": 1,
        "model_alias": MODEL_ID,
        "model_ftype": "",
        "model_path": MODEL_PATH,
        "modalities": {"vision": True, "audio": False, "video": False},
        "media_marker": "<__media__>",
        # These three tell the UI which llama.cpp endpoints to skip calling.
        "endpoint_slots": False,
        "endpoint_props": False,
        "endpoint_metrics": False,
        "ui": True,
        "ui_settings": {},
        "chat_template": template,
        "chat_template_caps": {},
        "bos_token": "",
        "eos_token": "",
        "build_info": "ninfer (franken/v0.11) + webui shim",
        "is_sleeping": False,
        "cors_proxy_enabled": False,
        "role": "model",
        "model_name": MODEL_ID,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ninfer-webui-shim"

    def log_message(self, fmt, *a):  # one concise line per request, for debugging the UI
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))
        sys.stderr.flush()

    # -- helpers ---------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else None
        content_type = self.headers.get("Content-Type", "application/json")

        # NInfer REQUIRES a "model" field; llama.cpp (and therefore its web UI) treats it as
        # optional for a single-model server. Fill it in so the UI works unmodified.
        # NInfer also only reads "max_tokens" (falling back to --default-max-tokens 8192
        # when absent) while the llama.cpp UI sends "n_predict". Without this translation
        # EVERY UI generation was silently capped at 8192 total tokens: long thinking hit
        # the cap mid-reasoning and the UI showed "Reasoning Cancelled" with a rushed
        # answer. Map n_predict -> max_tokens (positive values respected; missing or
        # negative means the UI default, for which we use the advertised 32768).
        if payload and method == "POST" and content_type.startswith("application/json"):
            try:
                body = json.loads(payload)
                if isinstance(body, dict):
                    if "model" not in body:
                        body["model"] = MODEL_ID
                    if "max_tokens" not in body and "max_completion_tokens" not in body:
                        n_predict = body.get("n_predict")
                        if isinstance(n_predict, bool):
                            n_predict = None
                        if isinstance(n_predict, (int, float)) and n_predict > 0:
                            body["max_tokens"] = int(n_predict)
                        else:
                            body["max_tokens"] = 32768
                    payload = json.dumps(body).encode()
            except (ValueError, UnicodeDecodeError):
                pass

        url = "http://%s%s" % (ARGS.upstream, self.path)
        req = urllib.request.Request(url, data=payload, method=method)
        req.add_header("Content-Type", content_type)
        try:
            upstream = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as e:  # upstream down
            self._send_json({"error": {"message": "upstream unavailable: %s" % e}}, 502)
            return

        streaming = "text/event-stream" in (upstream.headers.get("Content-Type") or "")
        self.send_response(upstream.status)
        self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
        if streaming:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = upstream.read(1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            body = upstream.read()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        target = os.path.normpath(os.path.join(ARGS.ui, rel))
        if not target.startswith(os.path.normpath(ARGS.ui)) or not os.path.isfile(target):
            self._send_json({"error": "not found", "path": path}, 404)
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/props":
            self._send_json(build_props())
        elif path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/tools":
            # The UI's optional built-in tool panel. NInfer exposes no tool endpoint;
            # answer with an empty catalogue so the panel renders empty instead of erroring.
            self._send_json({"tools": []})
        elif path == "/v1/streams/lookup":
            # The UI's optional conversation store. Answer empty; must be checked BEFORE the
            # generic /v1 proxy or the upstream 404s it.
            self._send_json({"streams": []})
        elif path.startswith("/v1/"):
            self._proxy("GET")
        else:
            self._static(path)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/streams/lookup":
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self._send_json({"streams": []})
        elif path.startswith("/v1/"):
            self._proxy("POST")
        elif path == "/tools":
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self._send_json({"error": "tools are not available through this shim"}, 200)
        elif path == "/tokenize":
            # Minimal stub: the UI only uses this for a context-usage count.
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self._send_json({"tokens": []})
        else:
            self._send_json({"error": "not found", "path": path}, 404)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--upstream", default="127.0.0.1:8081")
    ap.add_argument("--ui", default=UI_DIR)
    ARGS = ap.parse_args()

    if not os.path.isdir(ARGS.ui):
        sys.exit("UI directory not found: %s" % ARGS.ui)
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print("WebUI shim:  http://127.0.0.1:%d/   (UI from %s)" % (ARGS.port, ARGS.ui))
    print("API proxy :  http://127.0.0.1:%d/v1 -> http://%s/v1" % (ARGS.port, ARGS.upstream))
    srv.serve_forever()


if __name__ == "__main__":
    main()
