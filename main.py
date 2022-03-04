import gradio as gr
import torch

model_map = torch.hub.load('nateraw/image-generation:main', 'model_map')


class InferenceWrapper:
    def __init__(self, model):
        self.model = model
        self.pipe = torch.hub.load('nateraw/image-generation:main', 'styleganv3', pretrained=self.model, videos=True)

    def __call__(self, seed1, seed2, seed3, w_frames, model):
        if model != self.model:
            print(f"Loading model: {model}")
            self.model = model
            self.pipe = torch.hub.load('nateraw/image-generation:main', 'styleganv3', pretrained=self.model,
                                       videos=True)
        else:
            print(f"Model '{model}' already loaded, reusing it.")
        return self.pipe([seed1, seed2, seed3], w_frames=w_frames)


wrapper = InferenceWrapper('wikiart-1024')


def fn(s1, s2, s3, w_frames, model):
    return wrapper(s1, s2, s3, w_frames, model)


gr.Interface(
    fn,
    inputs=[
        gr.inputs.Slider(minimum=0, maximum=999999999, step=1, default=0, label='Random Seed For Image 1'),
        gr.inputs.Slider(minimum=0, maximum=999999999, step=1, default=0, label='Random Seed For Image 2'),
        gr.inputs.Slider(minimum=0, maximum=999999999, step=1, default=0, label='Random Seed For Image 3'),
        gr.inputs.Radio([60, 120, 240], type="value", default=60, label='Frames'),
        gr.inputs.Radio(list(model_map), type="value", default='wikiart-1024', label='Pretrained Model')
    ],
    outputs='video',
    examples=[[0, 1, 2, 60, 'landscapes-256']],
    enable_queue=True
).launch()
