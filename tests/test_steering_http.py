#!/usr/bin/env python3
"""Integration checks against a built binary and prepared Soura models.

python tests/test_steering_http.py --binary ./pocket-tts --models models --voices voices --voice example.wav
Uses isolated ephemeral-port workers; stops every process on exit.
"""
import argparse
import contextlib
import http.client
import json
import socket
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--voices', type=Path, required=True)
    parser.add_argument('--voice', default='example.wav')
    args = parser.parse_args()
    common = [str(args.binary.resolve()), '--models-dir', str(args.models.resolve()),
              '--tokenizer', str(args.models.resolve() / 'tokenizer.model'),
              '--voices-dir', str(args.voices.resolve()), '--threads-ar', '3',
              '--threads-dec', '2', '--threads-full', '2']
    base = {'text': 'Hello there. It is good to see you again.', 'voice': args.voice, 'format': 's16le'}

    @contextlib.contextmanager
    def server(enabled):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryFile() as log:
            process = subprocess.Popen(common + ['--server', '--port', str(port)] + (['--soura'] if enabled else []),
                                       stdout=log, stderr=log)
            try:
                for _ in range(200):
                    try:
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                        connection.request('GET', '/health')
                        response = connection.getresponse()
                        health = json.loads(response.read())
                        connection.close()
                        assert health['soura'] == enabled and health['sampling_seed']
                        break
                    except (OSError, http.client.HTTPException):
                        if process.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read().decode())
                        time.sleep(.1)
                else:
                    raise AssertionError('Worker did not become ready')
                yield port
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def request(port, body, expected=200, endpoint='/tts'):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
        try:
            connection.request('POST', endpoint, body if isinstance(body, str) else json.dumps(body), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            data = response.read()
            assert response.status == expected, (response.status, data[:200])
            if expected == 200:
                assert len(data) > 1000
            else:
                assert 'error' in json.loads(data)
            return data
        finally:
            connection.close()

    # Cache-only mode must create both caches, reuse fresh files, and report write failures.
    with tempfile.TemporaryDirectory() as temp:
        voices = Path(temp)
        shutil.copy2(args.voices / args.voice, voices / 'test.wav')
        command = common + ['--voices-dir', str(voices), '--prepare-voice', 'test.wav']
        first = subprocess.run(command, capture_output=True)
        assert first.returncode == 0 and not first.stdout, first.stderr
        caches = [voices / '.cache/test.emb', voices / '.cache/test.kv']
        assert all(path.stat().st_size > 0 for path in caches)
        timestamps = [path.stat().st_mtime_ns for path in caches]
        second = subprocess.run(command, capture_output=True)
        assert second.returncode == 0 and not second.stdout, second.stderr
        assert timestamps == [path.stat().st_mtime_ns for path in caches]
        shutil.rmtree(voices / '.cache')
        (voices / '.cache').write_text('not a directory')
        failed = subprocess.run(command, capture_output=True)
        assert failed.returncode != 0

    for flags in (['--emotion', 'happy'], ['--soura-vectors', 'unused.npy'],
                  ['--seed', '-1'], ['--seed', '0.5'], ['--seed', '1,\"x\":2'],
                  ['--soura', '--intensity', '1.20000001'],
                  ['--prepare-voice', args.voice, '--no-cache'],
                  ['--server', '--seed', '31'], ['--soura', '--soura-vectors', '/missing-vectors.npy', 'Hello.', args.voice, '/dev/null']):
        result = subprocess.run(common + flags, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert result.returncode != 0, flags
    with server(False) as port:
        request(port, base)
        request(port, {**base, 'emotion': 'happy', 'intensity': .8}, 400)
        request(port, {**base, 'emotion': 'happy', 'intensity': 0})
    with server(True) as port:
        for invalid in ({'emotion': 'surprise'}, {'emotion': 1}, {'emotion': ''}, {'emotion': None},
                        {'intensity': -.1}, {'intensity': 1.20000001}, {'intensity': '0.8'},
                        {'intensity': float('nan')}, {'seed': -1}, {'seed': .5}, {'seed': 9007199254740992}):
            request(port, {**base, **invalid}, 400)
        for label in ('angry', 'happy', 'sad', 'disgust', 'fear', 'neutral'):
            request(port, {**base, 'emotion': label, 'intensity': .8, 'seed': 31})
        request(port, {**base, 'seed': 31})  # no inherited emotion
        request(port, {'input': base['text'], 'voice': args.voice}, endpoint='/v1/audio/speech')
        interrupted = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
        interrupted.request('POST', '/tts', json.dumps({**base, 'text': base['text'] * 20, 'emotion': 'angry'}))
        response = interrupted.getresponse()
        assert response.status == 200
        response.read(2)
        response.close()
        interrupted.close()
        request(port, base)
    print('PASS: cache preparation/reuse/write failure, CLI validation, disabled/enabled HTTP controls, six emotions, omitted defaults, legacy endpoint, disconnect recovery')


if __name__ == '__main__':
    main()
