import os
from pathlib import Path
import gradio as gr
from pipeline import DEFAULT_PROMPT, SpaceGenerator, generate_batch


def run(prompt, image, count, seed, steps, consent, progress=gr.Progress()):
    if not consent:
        raise gr.Error('Confirm that your prompt/reference can be sent to the public Hugging Face Space.')
    try:
        generator = SpaceGenerator(token=os.getenv('HF_TOKEN'))
        results, archive = generate_batch(generator, prompt, image, count, seed, steps, progress)
    except Exception as error:
        raise gr.Error('Could not connect to the generator: ' + str(error)[:400]) from error
    previews = [row['files'][0] if row['files'] else None for row in results]
    previews += [None] * (4 - len(previews))
    summary = []
    for i, row in enumerate(results, 1):
        if row['error']:
            summary.append(f'Variation {i}: failed; batch stopped. {row["error"]}')
        else:
            report = row['report']
            summary.append(f'Variation {i} · seed {row["seed"]}: {report["vertices"]} vertices, '
                           f'{report["triangles"]} triangles. Watertight: {report["watertight"]}.')
    return (*previews, archive, '\n\n'.join(summary))

with gr.Blocks(title='Ninja3D Studio', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# Ninja3D Studio\nGenerate up to four ninja meshes, compare them, and download a batch.')
    gr.Markdown('Free open-source workflow powered by [Shap-E](https://github.com/openai/shap-e) '
                'and [hysts’ shared demo](https://huggingface.co/spaces/hysts/Shap-E). '
                'Shared GPU queues/quotas apply. This does not train the model or guarantee realistic characters.')
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(value=DEFAULT_PROMPT, label='Character description', lines=4)
            image = gr.Image(type='filepath', label='Optional front-view reference (takes priority over text)')
            refs = sorted(Path('references').glob('*.png'))
            if refs:
                gr.Examples(examples=[[str(p)] for p in refs], inputs=image, label='Our realistic ninja references')
            count = gr.Slider(1, 4, value=1, step=1, label='Variations (start with 1 on the free demo)')
            seed = gr.Number(value=0, precision=0, label='Starting seed; each variation adds 1')
            steps = gr.Slider(2, 100, value=64, step=1, label='Inference steps')
            consent = gr.Checkbox(label='Send this prompt/image to hysts’ public Hugging Face generator')
            generate = gr.Button('Generate ninja variations', variant='primary')
        with gr.Column(scale=2):
            with gr.Row():
                a = gr.Model3D(label='Variation 1')
                b = gr.Model3D(label='Variation 2')
            with gr.Row():
                c = gr.Model3D(label='Variation 3')
                d = gr.Model3D(label='Variation 4')
            archive = gr.File(label='Download all GLB / OBJ / STL files + generation manifest')
            status = gr.Textbox(label='Mesh checks and provider errors', interactive=False, lines=5)
    gr.Markdown('**Review before choosing:** face, hands, separated limbs, feet, clothing silhouette. '
                'Cleanup removes invalid/duplicate faces and sets height to 1.8 m; it does not repair anatomy. '
                'GLB preserves mesh colors. OBJ/STL are geometry exports. No skeleton, animation, or FBX exporter yet.')
    generate.click(run, [prompt, image, count, seed, steps, consent], [a, b, c, d, archive, status],
                   concurrency_limit=1, api_name=False)

if __name__ == '__main__':
    demo.queue(max_size=8).launch(server_name=os.getenv('HOST', '127.0.0.1'),
                                server_port=int(os.getenv('PORT', '7860')))
