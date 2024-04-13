import click
import numpy as np
import pickle

import src.dna_manipulation as dna_manipulation

def prep_image(image_path):
    pass


def load_encoder():
    ENCODER_PATH = "models/LinearRegressionZ.pickle"
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return encoder


def find_z(image):
    pass


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
    # preprocessed_image = prep_image(image)
    
    encoder = load_encoder()
    
    # z = find_z(preprocessed_image)
    z = np.random.rand(1, 256)
    parameters = np.rint(encoder.predict(z))[0]    
    modified_dna = modify_dna(dna, parameters)
    
    print(f"Saving output to {output}")
    with open(output, "w") as f:
        f.write(modified_dna)
        
    print("Done.")
    

if __name__=="__main__":
    main()