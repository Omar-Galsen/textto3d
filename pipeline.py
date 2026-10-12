"""Sequential, reproducible Shap-E batches; never silently retry GPU jobs."""
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import trimesh

DEFAULT_PROMPT = ('A full-body realistic ninja wearing a charcoal hood, face mask, fitted cloth tunic, '
                  'leather belt, wrapped arms and legs, split-toe boots, symmetric A-pose, no weapons')

def prepare_mesh(source, directory):
    scene = trimesh.load(source, force='scene')
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('The generator returned no triangle mesh.')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('The mesh contains invalid coordinates.')
    original = len(mesh.faces)
    # Preserve colors. Remove only duplicate and degenerate faces, then unused vertices.
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError('No usable faces remain after cleanup.')
    bounds = mesh.bounds
    height = float(bounds[1, 1] - bounds[0, 1])
    if height < 1e-8:
        raise ValueError('The mesh has zero height.')
    mesh.apply_translation([-mesh.centroid[0], -bounds[0, 1], -mesh.centroid[2]])
    mesh.apply_scale(1.8 / height)
    outputs = []
    for extension in ('glb', 'obj', 'stl'):
        path = directory / ('ninja.' + extension)
        mesh.export(path)
        outputs.append(str(path))
    report = {'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
              'removed_faces': original - len(mesh.faces),
              'watertight': bool(mesh.is_watertight), 'height_m': 1.8,
              'rigged': False, 'semantic_quality': 'Requires human review'}
    return outputs, report

class SpaceGenerator:
    def __init__(self, space='hysts/Shap-E', token=None):
        from gradio_client import Client
        self.client = Client(space, token=token or None, verbose=False)

    def generate(self, prompt, image, seed, steps, directory):
        from gradio_client import handle_file
        if image:
            result = self.client.predict(handle_file(image), seed, 3.0, steps, api_name='/image-to-3d')
        else:
            result = self.client.predict(prompt, seed, 15.0, steps, api_name='/text-to-3d')
        if isinstance(result, (tuple, list)):
            result = result[0]
        if isinstance(result, dict):
            result = result.get('path')
        if not result or not Path(result).is_file():
            raise ValueError('No downloaded model returned by the Space.')
        target = directory / 'raw.glb'
        shutil.copyfile(result, target)
        return target

def generate_batch(generator, prompt, image=None, count=4, seed=0, steps=64, progress=None):
    count, seed, steps = int(count), int(seed), int(steps)
    if count not in (1, 2, 3, 4):
        raise ValueError('Choose 1–4 variations.')
    if not 0 <= seed <= 2147483643 or not 2 <= steps <= 100:
        raise ValueError('Invalid seed or inference steps.')
    if not image and not prompt.strip():
        raise ValueError('Provide a prompt or reference image.')
    if image and not Path(image).is_file():
        raise ValueError('Reference image is missing.')
    root = Path(tempfile.mkdtemp(prefix='ninja3d-'))
    results = []
    for index in range(count):
        directory = root / f'variation-{index + 1:02}'
        directory.mkdir()
        if progress:
            progress(index / count, desc=f'Generating variation {index + 1}/{count}')
        try:
            source = generator.generate(prompt, image, seed + index, steps, directory)
            files, report = prepare_mesh(source, directory)
            results.append({'seed': seed + index, 'files': files, 'report': report, 'error': None})
        except Exception as error:
            # Stop at first provider/mesh failure: avoid repeating calls on quota or auth errors.
            results.append({'seed': seed + index, 'files': [], 'report': {},
                            'error': str(error)[:500]})
            break
    manifest = {'prompt': prompt, 'mode': 'image' if image else 'text',
                'steps': steps, 'requested': count, 'results': results,
                'notice': 'Seed variations do not train or improve the model. Review anatomy manually.'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = root / 'ninja-batch.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path != archive:
                z.write(path, path.relative_to(root))
    if progress:
        progress(1, desc='Batch finished')
    return results, str(archive)
