"""Fast graph/vector validation: python -m unittest discover -s tests -p 'test_*.py'."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from make_soura_onnx import LABELS, TARGET, load_vectors, patch


class SouraGraphTests(unittest.TestCase):
    def graph(self, consumer='/out_norm/Identity'):
        inputs = [helper.make_tensor_value_info(k, TensorProto.FLOAT, [1, 1, 1024]) for k in ('a', 'b')]
        output = helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, 1, 1024])
        return helper.make_model(helper.make_graph([
            helper.make_node('Add', ['a', 'b'], [TARGET]),
            helper.make_node('Identity', [TARGET], ['out'], name=consumer),
        ], 'test', inputs, [output]), opset_imports=[helper.make_opsetid('', 17)])

    def test_patch_and_reject_double_patch(self):
        model = patch(self.graph())
        self.assertEqual(model.graph.input[-1].name, 'soura_shift')
        self.assertEqual(list(model.graph.node[1].input), [TARGET + '_unsteered', 'soura_shift'])
        with self.assertRaises(ValueError):
            patch(model)

    def test_reject_wrong_injection_site(self):
        with self.assertRaises(ValueError):
            patch(self.graph('/transformer/layers.6/attention'))

    def test_vectors_preserve_magnitude_bypass_neutral(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.npy'
            rows = np.full((6, 1024), .25, np.float32)
            np.save(path, rows)
            actual = load_vectors(path)
            np.testing.assert_array_equal(actual[0], 0)
            np.testing.assert_array_equal(actual[1:], rows[1:])
            archive = Path(temp) / 'vectors.npz'
            np.savez(archive, **dict(zip(LABELS, rows)))
            np.testing.assert_array_equal(load_vectors(archive), actual)
            for invalid in (rows.astype(np.float64), rows[:5], rows * np.nan):
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    load_vectors(path)


if __name__ == '__main__':
    unittest.main()
