"""Serve this folder over HTTPS on https://localhost:8443 (Safari needs a secure context for the camera)."""
import http.server
import ssl
from pathlib import Path

root = Path(__file__).parent
handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8443), handler)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(root / "certs/cert.pem", root / "certs/key.pem")
server.socket = ctx.wrap_socket(server.socket, server_side=True)
print("Serving https://localhost:8443/mirror.html")
server.serve_forever()
