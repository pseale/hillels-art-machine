# @title Licensed under the MIT License

# Copyright (c) 2021 Katherine Crowson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#@title Parameters { run: "auto", display-mode: "form" }
#@markdown ### Basic options:
#@markdown ↓ The prompt to you want to try.


#@markdown ↓ How big you want the image to be. This can be up to 700, but if it's smaller, your images will appear *much* faster.

#@markdown ↓ If checked, will automatically bundle and download the images when the engine is done.

download_after_finishing = False #@param {type:"boolean"}

#@markdown ---

#@markdown ### Advanced Options
#@markdown *You don't need to change any of these things, but you can if you want.*



#@markdown ↓ How many steps you want the generator to run. Best between 400 and 1000 for default weirdness. 

total_steps = 400 #@param {type:"integer"}

#@markdown ↓ How many steps you want between images. Use a lower number if you want to make videos.

steps_per_image = 50 #@param {type:"integer"}

#@markdown ↓ Different flavors give you very different art for the same prompt.

flavor = 'cumin' #@param ["rosewater", "cumin"]

#@markdown ↓ If nonempty, the generator will try to mimick the style of the image. Should be a url. Leave empty if you just want a prompt.

style_url = "" #@param {type:"string"}
#@markdown ↓ How fast and sloppy the AI should be. 2 is a good balance.

weirdness = 2 #@param {type:"slider", min:1, max:11, step:1}
#@markdown ↓ Changing this number will give you slightly different results for the same prompt.
seed = 1 #@param {type:"integer"}


#@title low-level config you don't need to worry about { display-mode: "form" }
#@markdown You can click "show code" if you want to mess with it, but you shouldn't need to.

# TODO why is this even argparse this makes no sense!
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--text-prompt', type=str)
parser.add_argument('--image-prompt', type=str, default='')
parser.add_argument('--total-iterations', type=int, default=400)
parser.add_argument('--image-size', type=int, default=400)



args = parser.parse_args()
args.prompts = [args.text_prompt]
args.image_prompts = [args.image_prompt] if args.image_prompt != '' else []
args.noise_prompt_seeds=[]
args.noise_prompt_weights=[]
args.tv_weight=0.1
args.step_size=0.05*(weirdness if weirdness != 11 else 22)
args.weight_decay=0.
args.cutn=64
args.cut_pow=1.
args.display_freq=steps_per_image
#args.total_iterations=total_steps
args.seed=seed


#@title More, different setup { display-mode: "form" }
import math
import io
from pathlib import Path
import sys
import time

# from IPython import display
from omegaconf import OmegaConf
from PIL import Image
import requests
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import kornia.augmentation as K


import warnings
warnings.simplefilter("ignore") # Avoid spookin people for Cumin

from clip import clip
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
    

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        headers={"user-agent": "AIArtMachineBot/0.0 (https://is.gd/aiartmachine; h@hillelwayne.com) generic-library/0.0"}
        r = requests.get(url_or_path, headers=headers)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = [prompt]
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean()


class MakeCutoutsDefault(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

class MakeCutoutsCumin(nn.Module):
    """from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ"""
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),
            
)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.encoder, model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vqgan_config='vqgan_imagenet_f16_1024.yaml'
vqgan_checkpoint='vqgan_imagenet_f16_1024.ckpt'
model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
perceptor = clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(device)

cut_size = perceptor.visual.input_resolution
e_dim = model.quantize.e_dim
f = 2**(model.decoder.num_resolutions - 1)
flavordict = {
    "default": MakeCutoutsDefault,
    "cumin": MakeCutoutsCumin,
    "rosewater": MakeCutoutsDefault
}
make_cutouts = flavordict[flavor](cut_size, args.cutn, cut_pow=args.cut_pow)
n_toks = model.quantize.n_e
toksX, toksY = args.image_size // f, args.image_size // f
sideX, sideY = toksX * f, toksY * f

if args.seed is not None:
    torch.manual_seed(args.seed)

logits = torch.randn([toksY * toksX, n_toks], device=device, requires_grad=True)
opt = optim.AdamW([logits], lr=args.step_size, weight_decay=args.weight_decay)

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

pMs = []

from re import sub

# I hate using underscores for names, but this way is better for people using utf-8
p_filename = sub("\W", "", args.text_prompt.lower().replace(" ","_"))

### make the output path for dumping the images 
img_path = Path("img") / "grouped" / p_filename
all_path = Path("img/all")
img_path.mkdir(parents = True, exist_ok = True)
all_path.mkdir(parents = True, exist_ok = True)

for prompt in args.prompts:
    txt, weight, stop = parse_prompt(prompt)
    embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for prompt in args.image_prompts:
    path, weight, stop = parse_prompt(prompt)
    img = resize_image(Image.open(fetch(path)).convert('RGB'), (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img)[None].to(device))
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))

def synth(one_hot):
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    return clamp_with_grad(model.decode(z).add(1).div(2), 0, 1)

@torch.no_grad()
def checkin(i, losses):
    tqdm.write(f'iterations: {i}, prompt: {args.text_prompt}')
    one_hot = F.one_hot(logits.argmax(1), n_toks).to(logits.dtype)
    out = synth(one_hot)
    out_img = TF.to_pil_image(out[0].cpu())
    out_img.save(all_path / f"{p_filename}-{i:0=4}.png")
    out_img.save(img_path / f"{i:0=4}.png")

    # display.display(display.Image('progress.png'))

def ascend_txt():
    probs = logits.softmax(1)
    one_hot = F.one_hot(probs.multinomial(1)[..., 0], n_toks).to(logits.dtype)
    one_hot = replace_grad(one_hot, probs)
    out = synth(one_hot)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.tv_weight:
        result.append(tv_loss(out) * args.tv_weight / 4)

    for prompt in pMs:
        result.append(prompt(iii))

    return result

def train(i):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt()
    if i % args.display_freq == 0:
        checkin(i, lossAll)
    loss = sum(lossAll)
    loss.backward()
    opt.step()


#@title Images will appear here! { display-mode: "form" }

i = 1
print("NOTE: First image will look random. This is normal.")
try:
    # with tqdm() as pbar:
        start = time.perf_counter()
        while i < args.total_iterations:
            # pbar.update()
            train(i)
            if i == 10:
              end = time.perf_counter()
              print(f"It will take about {int((end - start) * args.total_iterations // (60 * 10))} minutes to complete all {args.total_iterations} iterations.")
            i += 1
except KeyboardInterrupt:
    pass
except RuntimeError:
    print("ERROR! ERROR! ERROR!")
    print("The image size you chose was too big!")
else:
  print("Final image.")
  checkin(i, 0)

print("All done!")
