import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('assets', Path(__file__).resolve().parents[1] / 'check_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetCheckTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in assets.REQUIRED:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        (self.root / 'models/soura_vectors.derived.json').write_text(json.dumps({
            'vectors_sha256': hashlib.sha256(b'fixture').hexdigest(),
        }))

    def test_missing_default_model(self):
        self.assertEqual(assets.check(self.root), [])
        (self.root / 'models/flow_lm_main_delta_attn_flow_int8_soura.onnx').unlink()
        self.assertTrue(any('Missing or empty' in e for e in assets.check(self.root)))

    def test_changed_vectors(self):
        (self.root / 'models/soura_vectors.derived.npy').write_bytes(b'changed')
        self.assertTrue(any('SHA256' in e for e in assets.check(self.root)))

    def test_stale_sidecar_even_with_newer_timestamp(self):
        try:
            import brotli
        except ImportError:
            self.skipTest('brotli not installed')
        original = self.root / 'vendor/ptt/pocket_tts_wasm.wasm'
        sidecar = original.with_suffix('.wasm.br')
        sidecar.write_bytes(brotli.compress(b'old build'))
        self.assertTrue(any('Stale compressed' in e for e in assets.check(self.root)))
        sidecar.write_bytes(brotli.compress(original.read_bytes()))
        self.assertEqual(assets.check(self.root), [])
