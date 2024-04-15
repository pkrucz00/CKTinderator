import click
import numpy as np
import pickle

import src.dna_manipulation as dna_manipulation
from src.find_z import find_z, load_generator
from src.image_align import image_align
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2


IMG_SIZE = 256


def prep_image(image_path) -> torch.Tensor:
    aligned_image =  image_align(image_path)
    image_min_size = min(aligned_image.size)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(image_min_size),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])
    return transforms(aligned_image).to("cuda:0")


def load_normalizer():
    NORMALIZER_PATH = "models/normalizer.pickle"
    with open(NORMALIZER_PATH, 'rb') as f:
        normalizer = pickle.load(f)
        
    return normalizer


def load_encoder():
    ENCODER_PATH = "models/MLPRegressorZ.pickle"
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return encoder


def modify_dna(dna_path: str, parameters: np.ndarray) -> str:
    GENES_TO_CHANGE = ['eye_distance', 'eye_height', 'eye_angle', 'jaw_width', 'jaw_height']
    dna_text = dna_manipulation.load_dna(dna_path)
    params_dict = {gene: value for gene, value in zip(GENES_TO_CHANGE, parameters)}
    return  dna_manipulation.change_dna(dna_text, params_dict)


@click.command()
@click.option('--image', prompt='Image file', help='Image file to be processed', type=click.Path(exists=True))
@click.option('--dna', prompt='DNA file', help='DNA file used as a template')
@click.option('--output', prompt='Output file', default="result-dna.txt", help='Output file to be saved', type=click.Path())
def main(image, dna, output):
    preprocessed_image = prep_image(image)
    normalizer = load_normalizer()
    encoder = load_encoder()
    generator = load_generator()
    
    z = find_z(preprocessed_image)
    generated_image = generator(torch.from_numpy(z))
    plt.imshow(generated_image[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()
    print("Z:", z.shape)
    parameters = np.rint(encoder.predict(normalizer.transform(z)))[0]    
    print("Parameters:", parameters)
    modified_dna = modify_dna(dna, parameters)
    
    print(f"Saving output to {output}")
    with open(output, "w") as f:
        f.write(modified_dna)
        
    print("Done.")
    

if __name__=="__main__":
    main()