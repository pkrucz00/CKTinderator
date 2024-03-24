import json
import numpy as np

from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import v2

from find_z import find_z

from time import time

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
    z_list = []
    for i in tqdm(range(250, 350)):
        batch = torch.cat([prep_image(p).unsqueeze(0) for p in path_names[4*i:4*(i+1)]])
        z = find_z(batch).numpy()
        z_list.append(z)
    
    z_list = np.array(z_list).reshape((-1, 256))
    np.save("/media/pawel/DATA/tmp/freddie_mercuries/z_less_iter_2.npy", z_list)
    # This needs to be faster but I don't have too much ideas on how to make this work as necessary. 
    
    # First of all I need to make a small framework that will show that his at least has some merit in it. I need to compare G(z) and the original. Then I want to create a small random forest that would find a regression from z space to R^5 After that a distributed (randomized) algorithm for inferring different items is needed. And I will also need to find people who would want to help me with this. I'll ask Adam and some other people. Maybe Michał Idzik? But I don't want to take all his computational power... Maybe Lukasz? He has his compuer standing as it is. May he can help. I will also need to store this data somewhere. The target 100 vectors now take aproximately . Maybe Google Colab wil be enough for some of the computations? A tleast batching needs to be p checked.
    
    
    
    
    