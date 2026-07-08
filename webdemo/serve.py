#!/usr/bin/env python3
"""Dev server for the PocketTTS webdemo.

- COOP/COEP headers (SharedArrayBuffer / WASM threads)
- HTTPS with an auto-generated self-signed cert, so other devices on the LAN
  get a secure context (SharedArrayBuffer requires HTTPS or localhost).
  Accept the certificate warning once per device.
- Serves pre-compressed .br files with Content-Encoding: br when present
  (run compress.py after changing models).

    python3 webdemo/serve.py [port]   # http on <port>, https on <port+1>

Use http://localhost:<port> on this machine (localhost is a secure context,
threads work fine over plain http). Use https://<lan-ip>:<port+1> from other
devices — non-localhost origins need TLS for SharedArrayBuffer.
"""
import http.server
import os
import socket
import ssl
import subprocess
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PORT = int(args[0]) if args else 8093
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def ensure_cert():
    cert_dir = os.path.join(HERE, ".certs")
    crt, key = os.path.join(cert_dir, "dev.crt"), os.path.join(cert_dir, "dev.key")
    if not (os.path.exists(crt) and os.path.exists(key)):
        os.makedirs(cert_dir, exist_ok=True)
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", key, "-out", crt, "-days", "3650",
            "-subj", "/CN=pockettts-dev",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1," + ",".join(
                f"IP:{ip}" for ip in local_ips()),
        ], check=True, capture_output=True)
        print(f"generated self-signed cert in {cert_dir}")
    return crt, key


def local_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return ips or ["127.0.0.1"]


def git_stamp():
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--date=format:%H:%M"],
                             capture_output=True, text=True, cwd=HERE)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


STAMP = git_stamp()


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript",
        ".js": "text/javascript",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".emb": "application/octet-stream",
    }

    def send_head(self):
        if self.path.split("?")[0] == "/version":
            body = STAMP.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        # Serve pre-compressed .br transparently when the client accepts it.
        path = self.translate_path(self.path)
        accepts_br = "br" in self.headers.get("Accept-Encoding", "")
        if accepts_br and not path.endswith(".br") and os.path.isfile(path + ".br"):
            self._br_original = path
            self.path += ".br"
        else:
            self._br_original = None
        return super().send_head()

    def guess_type(self, path):
        if path.endswith(".br"):
            return super().guess_type(path[:-3])
        return super().guess_type(path)

    def end_headers(self):
        if getattr(self, "_br_original", None):
            self.send_header("Content-Encoding", "br")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *fmt_args):
        msg = fmt_args[0] if fmt_args else ""
        if not any(x in str(msg) for x in (".onnx", ".wasm", ".br")):
            super().log_message(fmt, *fmt_args)


import threading


class HttpHandler(Handler):
    # Plain-http from another machine can't use SharedArrayBuffer (not a
    # secure context) and mistyped scheme against the TLS port just resets.
    # Redirect every non-localhost http request to the https listener.
    def send_head(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("localhost", "127.0.0.1", "[::1]"):
            self.send_response(307)
            self.send_header("Location", f"https://{host}:{PORT + 1}{self.path}")
            self.end_headers()
            return None
        return super().send_head()


http_srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), HttpHandler)

https_srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT + 1), Handler)
crt, key = ensure_cert()
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(crt, key)
https_srv.socket = ctx.wrap_socket(https_srv.socket, server_side=True)

print("PocketTTS webdemo:")
print(f"  this machine : http://localhost:{PORT}/")
for ip in local_ips():
    print(f"  on your LAN  : https://{ip}:{PORT + 1}/   (accept the cert warning once)")
print(f"  (http from other devices will load the page but WASM threads need")
print(f"   a secure context, so use the https URL off this machine)")
print("Ctrl+C to stop")

threading.Thread(target=http_srv.serve_forever, daemon=True).start()
https_srv.serve_forever()
