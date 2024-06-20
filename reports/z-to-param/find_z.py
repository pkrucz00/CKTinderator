import os
import sys
import errno

from pathlib import Path

from lightweight_gan import LightweightGAN
import torch

from time import time

from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX
try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir
    
LATENT_DIM = 256
IMG_SIZE = 256

torch.manual_seed(42)

# landmarks model

def load_file_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file


def load_landmarks_model():
    two_d_model_url = "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"
    return torch.jit.load(load_file_from_url(two_d_model_url)).to("cuda:0").eval()


# Generator
def load_generator() -> torch.nn.Module:   # TODO change this to a generator model
    model_filepath = Path("model/model_174.pt")
    model = torch.load(model_filepath)
    
    gan = LightweightGAN(latent_dim=LATENT_DIM,
                         image_size=IMG_SIZE, 
                         attn_res_layers=[32],
                         freq_chan_attn=False)
    gan.load_state_dict(model["GAN"], strict=False)
    gan.eval()
    gan.G.eval()

    return gan.G



class DifferentiableLandmarks(torch.nn.Module):
    def __init__(self, generator, landmarks_model, images: torch.Tensor = None):
        super().__init__()
        self.z = torch.nn.Parameter(pick_best_random_z(images, generator))
        
        self.landmarks_model = landmarks_model
        self.generator = generator
        self.generator.eval()
        self.landmarks_model.eval()
               
        for p in self.generator.parameters():
            p.requires_grad = False
            
        for p in self.landmarks_model.parameters():
            p.requires_grad = False
        
        self.curr_image = self.generator(self.z)
        
        
    def forward(self):
        self.curr_image = self.generator(self.z)
        return self.landmarks_model(self.curr_image)
    
    def get_z(self) -> torch.Tensor:
        return self.z.clone().detach().cpu()
    
    def print_are_training(self):
        print("Are training:")
        print(f"Generator: {self.generator.training}")
        print(f"Landmarks: {self.landmarks_model.training}")
        print(f"Model: {self.training}")
        
        
def pick_best_random_z(images: torch.Tensor, generator: torch.nn.Module, n=100, criterion=torch.nn.MSELoss()) -> torch.Tensor:
    batch_size = images.shape[0]
    z = torch.randn( (batch_size, LATENT_DIM), requires_grad=True, device="cuda")
    z_best = z.clone()
    loss_best = float("inf")
    
    for _ in range(n):
        z = torch.randn( (batch_size, LATENT_DIM), requires_grad=True, device="cuda")
        images_gen = generator(z)
        with torch.no_grad():
            loss = criterion(images, images_gen)
            if loss < loss_best:
                z_best = z.clone()
                loss_best = loss.clone()
            
    return z_best
    
        
def find_z(images: torch.Tensor, lr=1e-2, iters=500) -> torch.Tensor:   
    generator = load_generator().to("cuda:0")
    landmarks_model = load_landmarks_model()
    model = DifferentiableLandmarks(generator, landmarks_model, images)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    y_target = landmarks_model(images).detach()
    
    for _ in range(iters):
        y_pred = model()
        loss = criterion(y_target, y_pred) + 5 * criterion(images, model.curr_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return  model.get_z()

