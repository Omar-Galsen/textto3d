import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path
import trimesh
from pipeline import generate_batch, prepare_mesh, SpaceGenerator

class FakeGenerator:
    def __init__(self, fail=False): self.seeds = []; self.fail = fail
    def generate(self, prompt, image, seed, steps, directory):
        self.seeds.append(seed)
        if self.fail: raise RuntimeError('GPU quota reached')
        path = directory / 'raw.glb'
        trimesh.creation.box(extents=[1, 2, .5]).export(path)
        return path

class PipelineTests(unittest.TestCase):
    def test_client_auth_parameter(self):
        with patch("gradio_client.Client") as client:
            SpaceGenerator(token="test")
            client.assert_called_once_with("hysts/Shap-E", hf_token="test", verbose=False)

    def test_batch_seeds_exports_manifest(self):
        fake = FakeGenerator()
        rows, archive = generate_batch(fake, 'ninja', count=4, seed=42)
        self.assertEqual(fake.seeds, [42,43,44,45])
        self.assertEqual(len(rows), 4)
        for row in rows:
            mesh = trimesh.load(row['files'][0], force='scene').to_geometry()
            self.assertAlmostEqual(mesh.extents[1], 1.8, places=5)
            self.assertAlmostEqual(mesh.bounds[0,1], 0, places=5)
        with zipfile.ZipFile(archive) as z:
            self.assertIn('variation-04/ninja.stl', z.namelist())
            self.assertEqual(json.loads(z.read('manifest.json'))['requested'], 4)
    def test_quota_stops_after_one_call(self):
        fake = FakeGenerator(fail=True)
        rows, archive = generate_batch(fake, 'ninja', count=4)
        self.assertEqual(fake.seeds, [0])
        self.assertIn('quota', rows[0]['error'])
        self.assertTrue(Path(archive).is_file())
    def test_invalid_count_never_calls_provider(self):
        fake = FakeGenerator()
        with self.assertRaises(ValueError): generate_batch(fake, 'ninja', count=100)
        self.assertEqual(fake.seeds, [])
if __name__ == '__main__': unittest.main()
