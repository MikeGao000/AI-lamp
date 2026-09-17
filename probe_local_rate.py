"""Stream frame rate measured on the Pi itself, with no SSH tunnel in the path.

The capture loop's own stages total 26-65 ms per frame, yet the stream delivered
1.72 fps through the tunnel. That points at the tunnel, so this measures the same
endpoint locally and separates "the service is slow" from "the path is slow".
"""

import http.client
import time


def rate(host, port, path, seconds=6.0):
    connection = http.client.HTTPConnection(host, port, timeout=seconds + 4)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        started = time.time()
        frames = 0
        total = 0
        previous = None
        while time.time() - started < seconds:
            chunk = response.read(65536)
            if not chunk:
                break
            total += len(chunk)
            frames += chunk.count(b"\xff\xd8")
        elapsed = time.time() - started
        print(f"{path} on {host}:{port}: {frames} frames in {elapsed:.1f}s = "
              f"{frames / max(elapsed, 0.01):.2f} fps, "
              f"{total / max(elapsed, 0.01) / 1024:.0f} KB/s")
    finally:
        connection.close()


rate("127.0.0.1", 8000, "/stream.mjpg")
