import click
import numpy as np

import src.dna_manipulation as dna_manipulation
from src.find_z import find_z, load_generator
from src.image_align import image_align
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
from joblib import load

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


def load_encoder():
    ENCODER_PATH = "models/MLPRegressor_100000_Z.joblib"
    return load(ENCODER_PATH)


def modify_dna(dna_path: str, parameters: np.ndarray) -> str:
    GENES_TO_CHANGE = ['jaw_height', 'jaw_width',  'eye_angle', 'eye_height', 'eye_distance']
    dna_text = dna_manipulation.load_dna(dna_path)
    params_dict = {gene: value for gene, value in zip(GENES_TO_CHANGE, parameters)}
    return  dna_manipulation.change_dna(dna_text, params_dict)


@click.command()
@click.option('--image', prompt='Image file', help='Image file to be processed', type=click.Path(exists=True))
@click.option('--dna', prompt='DNA file', help='DNA file used as a template')
@click.option('--output', prompt='Output file', default="result-dna.txt", help='Output file to be saved', type=click.Path())
@click.option('--generated', prompt='Generated file path', default="result-generated.png", help='Generated file path', type=click.Path())
def main(image, dna, output, generated):
    preprocessed_image = prep_image(image)
    encoder = load_encoder()
    generator = load_generator()
    
    z = find_z(preprocessed_image)
    generated_image = generator(torch.from_numpy(z))
    plt.imshow(generated_image[0].detach().cpu().numpy().transpose(1, 2, 0))
    plt.savefig(generated)
    
    print("Z:", z.shape)
    parameters = np.rint(encoder.predict(z))[0]    
    print("Parameters:", parameters)
    modified_dna = modify_dna(dna, parameters)
    
    print(f"Saving output to {output}")
    with open(output, "w") as f:
        f.write(modified_dna)
        
    print("Done.")
    

if __name__=="__main__":
    main()