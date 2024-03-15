import json
import numpy as np

from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import v2

from find_z import find_z

IDX_TO_VALUE = {"0": 35, "1": 66, "2": 87, "3": 104, "4": 120, "5": 135, "6": 151, "7": 168, "8": 189, "9": 220} 
IMG_SIZE = 256
BATCH_SIZE = 1000


def get_vec_from_pathname(pathname: Path) -> np.ndarray:
    coded_vector = pathname.stem[-5:]
    return np.array([IDX_TO_VALUE[c] for c in coded_vector])


def prep_image(filename)-> torch.tensor:
    image = Image.open(filename).convert("RGB")
    image_min_size = min(image.size)
    transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(image_min_size),
        v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True)
    ])
    return transforms(image).cuda(0)


if __name__ == '__main__':
    path_names = [Path(p) for p in sorted(glob("/media/pawel/DATA/tmp/freddie_mercuries/en_face/aligned/*"))]
    print("Loaded path names")
    # out_vec = np.array([get_vec_from_pathname(p) for p in tqdm(path_names)])

    # np.save("./data/outvec.npy", out_vec)
    
    # with open("./data/idx_to_value.json", "w") as f:
    #     json.dump(IDX_TO_VALUE, f)
        
    # with open("./data/position_to_gene.json", "w") as f:
    #     json.dump({0: "gene_jaw_height", 1:"gene_jaw_width", 2: "gene_eye_angle", 3: "gene_eye_height", 4: "gene_eye_distance"}, f)
                
    # print("Converting images to naive input...")
    # naive_input = np.array([np.array(Image.open(p).convert("L").resize((256, 256))).flatten() for p in tqdm(path_names)])
    # print("Done.")
    
    # for i in tqdm(range( len(naive_input) // BATCH_SIZE )):
    #     idx_from, idx_to = BATCH_SIZE*i, BATCH_SIZE*(i+1)
    #     np.save(f"/media/pawel/DATA/tmp/freddie_mercuries/naive_input/{i}.npy", naive_input[:, idx_from:idx_to])
    
    z = np.array([find_z(prep_image(p)).numpy() for p in tqdm(path_names[60:70])])
    np.save("/media/pawel/DATA/tmp/freddie_mercuries/z4.npy", z)
    
    
    
    
    