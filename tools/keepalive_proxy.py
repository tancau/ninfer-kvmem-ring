"""SSE keepalive proxy.

Problem it solves: the engine emits nothing for the whole prefill, which for a cold
135K-token conversation is several minutes. Clients enforce an idle timeout on the
stream (the one in use defaults to 300 s) and kill the request, which the user sees as
"pi-ai stream idle timeout after 300000ms" followed by a retry.

Fix: sit between the client and the engine and, whenever the engine has been silent for
KEEPALIVE seconds, write an SSE comment line (": keepalive") to the client. SSE comments
are part of the spec and must be ignored by conforming parsers, so no client change is
needed and the model output is untouched.

Framing: the engine replies with `Transfer-Encoding: chunked`. Injecting raw bytes into a
chunked body would corrupt it, so this proxy de-chunks the response and delimits it with
`Connection: close` instead, which lets us interleave comments freely.

  py keepalive_proxy.py [listen_port] [upstream_port] [keepalive_seconds]
"""
import socket
import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

LISTEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
UPSTREAM = int(sys.argv[2]) if len(sys.argv) > 2 else 8082
KEEPALIVE = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
UP_HOST = '127.0.0.1'


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def read_headers(sock):
    """Read until the end of the header block; return (header_bytes, leftover)."""
    buf = b''
    while b'\r\n\r\n' not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return buf, b''
        buf += chunk
        if len(buf) > 1024 * 1024:
            raise RuntimeError('header block too large')
    head, _, rest = buf.partition(b'\r\n\r\n')
    return head + b'\r\n\r\n', rest


def dechunk(sock, initial, on_data, on_idle):
    """Decode a chunked body, calling on_data(bytes) for payload and on_idle() when the
    upstream has been quiet for KEEPALIVE seconds."""
    buf = initial
    done = False
    sock.settimeout(KEEPALIVE)
    while not done:
        # need a full size line
        while b'\r\n' not in buf:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                on_idle()
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
        line, _, rest = buf.partition(b'\r\n')
        buf = rest
        try:
            size = int(line.split(b';')[0].strip(), 16)
        except ValueError:
            return
        if size == 0:
            return
        while len(buf) < size + 2:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                # partial data held back: counting it as idle is the safe direction
                on_idle()
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
        on_data(buf[:size])
        buf = buf[size + 2:]


def handle(client):
    try:
        head, rest = read_headers(client)
        if not head:
            return
        lines = head.split(b'\r\n')
        request_line = lines[0].decode('latin-1')
        try:
            method, path, _ = request_line.split(' ', 2)
        except ValueError:
            return
        headers = {}
        for ln in lines[1:]:
            if b':' in ln:
                k, v = ln.split(b':', 1)
                headers[k.strip().lower()] = v.strip()

        length = int(headers.get(b'content-length', b'0') or 0)
        body = rest
        while len(body) < length:
            chunk = client.recv(65536)
            if not chunk:
                break
            body += chunk

        up = socket.create_connection((UP_HOST, UPSTREAM), timeout=300)
        fwd = [b'%s %s HTTP/1.1' % (method.encode(), path.encode())]
        fwd.append(b'Host: %s:%d' % (UP_HOST.encode(), UPSTREAM))
        for k, v in headers.items():
            if k in (b'content-length', b'host', b'connection', b'accept-encoding'):
                continue
            fwd.append(b'%s: %s' % (k, v))
        fwd.append(b'Content-Length: %d' % len(body))
        fwd.append(b'Connection: close')
        up.sendall(b'\r\n'.join(fwd) + b'\r\n\r\n' + body)

        uphead, uprest = read_headers(up)
        status = uphead.split(b'\r\n')[0]
        log('%s %s -> %s' % (method, path, status.decode('latin-1')))
        hdrs = []
        for ln in uphead.split(b'\r\n')[1:]:
            if not ln or b':' not in ln:
                continue
            k, v = ln.split(b':', 1)
            kl = k.strip().lower()
            if kl in (b'transfer-encoding', b'content-length', b'connection'):
                continue
            hdrs.append(b'%s: %s' % (k.strip(), v.strip()))
        is_sse = any(h.lower().startswith(b'content-type')
                     and b'event-stream' in h.lower() for h in hdrs)
        client.sendall(b'HTTP/1.1 ' + status.split(b' ', 1)[1] + b'\r\n'
                       + b'\r\n'.join(hdrs) + b'\r\nConnection: close\r\n\r\n')

        stats = {'bytes': 0, 'ka': 0, 'last': time.time()}

        def on_data(b):
            stats['bytes'] += len(b)
            stats['last'] = time.time()
            client.sendall(b)

        def on_idle():
            if not is_sse:
                return
            client.sendall(b': keepalive\n\n')
            stats['ka'] += 1
            log('  keepalive #%d (%d payload bytes so far)' % (stats['ka'], stats['bytes']))

        if b'transfer-encoding' in uphead.lower() and b'chunked' in uphead.lower():
            dechunk(up, uprest, on_data, on_idle)
        else:
            if uprest:
                on_data(uprest)
            up.settimeout(KEEPALIVE)
            while True:
                try:
                    chunk = up.recv(65536)
                except socket.timeout:
                    on_idle()
                    continue
                if not chunk:
                    break
                on_data(chunk)
        log('  done: %d payload bytes, %d keepalives' % (stats['bytes'], stats['ka']))
    except Exception as e:
        log('  error: %r' % (e,))
    finally:
        for s in (client,):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', LISTEN))
    srv.listen(8)
    log('keepalive proxy :%d -> :%d  (comment every %.0fs)' % (LISTEN, UPSTREAM, KEEPALIVE))
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client,), daemon=True).start()


main()
