"""Run original OpenAI Shap-E text-to-3D, without the shared demo API."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor' / 'shap-e'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt', default='A full body ninja wearing black clothing')
    p.add_argument('--count', type=int, choices=range(1,5), default=1)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'shap-e-test')
    p.add_argument('--verify-source', action='store_true', help='Verify upstream hashes without downloading weights')
    args = p.parse_args()
    if args.verify_source:
        import hashlib
        root = ROOT / 'vendor' / 'shap-e'
        record = json.loads((root / 'UPSTREAM.json').read_text())
        for name, digest in record['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise SystemExit('Modified upstream file: ' + name)
        print(f'Verified {len(record["files"])} files at upstream commit {record["commit"]}')
        return
    if not args.prompt.strip() or not 2 <= args.steps <= 100:
        p.error('Provide a prompt and 2–100 inference steps.')
    import torch
    from shap_e.models.download import load_model, load_config
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable. Install CUDA-enabled PyTorch or choose --device cpu.')
    if device.type == 'cpu':
        print('CPU inference may be very slow. No paid API or remote inference is used.')
    args.output.mkdir(parents=True, exist_ok=True)
    print('Loading original transmitter and text300M weights; first run requires downloads.')
    transmitter = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))
    files = []
    for index in range(args.count):
        seed = args.seed + index
        torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        latent = sample_latents(batch_size=1, model=model, diffusion=diffusion,
            guidance_scale=15.0, model_kwargs={'texts':[args.prompt]}, progress=True,
            clip_denoised=True, use_fp16=device.type=='cuda', use_karras=True,
            karras_steps=args.steps, sigma_min=1e-3, sigma_max=160, s_churn=0)[0]
        mesh = decode_latent_mesh(transmitter, latent).tri_mesh()
        path = args.output / f'ninja-{index+1:02}-seed-{seed}.obj'
        with path.open('w') as f: mesh.write_obj(f)
        files.append({'file':path.name, 'seed':seed})
        print('Saved', path)
    (args.output/'manifest.json').write_text(json.dumps({'prompt':args.prompt,'steps':args.steps,
        'device':str(device),'files':files,'upstream_commit':json.loads(
        (ROOT/'vendor/shap-e/UPSTREAM.json').read_text())['commit']},indent=2))

if __name__ == '__main__': main()
