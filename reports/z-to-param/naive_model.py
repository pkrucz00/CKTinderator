import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


from glob import glob

Z_VEC_LOC = "/media/pawel/DATA/tmp/freddie_mercuries/en_face/vectors"


def concatenate_zs(zs: list):
    return np.concatenate(zs, axis=0)

def load_data():
    
    z = concatenate_zs([np.load(f) for f in sorted(glob(f"{Z_VEC_LOC}/*.npy"))])
    permute = np.load("permuted_100000.npy")
    naive_input = np.load("/media/pawel/DATA/tmp/freddie_mercuries/naive_input.npy")[permute]
    out_vec = np.load("./data/outvec.npy")[permute]
    
    return z, out_vec


def random_forest_train_and_eval(naive_input, out_vec):
    naive_input, out_vec = scale(naive_input), scale(out_vec)
    X_train, X_test, y_train, y_test = train_test_split(naive_input, out_vec, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    print(f"Train: {model.score(X_train, y_train)}")
    print(f"Test: {model.score(X_test, y_test)}")

def MLP_train_and_eval(naive_input, out_vec):
    naive_input, out_vec = scale(naive_input), scale(out_vec)
    X_train, X_test, y_train, y_test = train_test_split(naive_input, out_vec, test_size=0.2, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
    model.fit(X_train, y_train)
    
    print(f"Train: {model.score(X_train, y_train)}")
    print(f"Test: {model.score(X_test, y_test)}")
    
    
def GaussianProcess_train_and_eval(naive_input, out_vec):
    naive_input, out_vec = scale(naive_input), scale(out_vec)
    X_train, X_test, y_train, y_test = train_test_split(naive_input, out_vec, test_size=0.2, random_state=42)
    model = GaussianProcessRegressor()
    model.fit(X_train, y_train)
    
    print(f"Train: {model.score(X_train, y_train)}")
    print(f"Test: {model.score(X_test, y_test)}")
    

def scale(X):
    skaler = StandardScaler()
    skaler.fit(X)
    X = skaler.transform(X)
    return X

if __name__ == '__main__':
    z, out_vec = load_data()
    zs = [z[:i] for i in [100, 500, 1000, 2000, 5000]]
    out_vecs = [out_vec[:i] for i in [100, 500, 1000, 2000, 5000]]
    # out_vec = out_vec[:len(z)]
    # naive_input, z, out_vec = naive_input[:100], z[:100], out_vec[:100]
    print("Data has been loaded")
    print(z.shape, out_vec.shape)
    
    
    # naive_input = scale(naive_input)
    z = scale(z)
    out_vec = scale(out_vec)
    print("Data has been scaled")
    
    
    print("RANDOM FOREST  --- Z")
    for X, Y in zip(zs, out_vecs):
        print(f"Size: {X.shape[0]}")
        # print("Random Forest")
        # random_forest_train_and_eval(X, Y)
        print("MLP")
        MLP_train_and_eval(X, Y)
        # print("Gaussian Process")
        # GaussianProcess_train_and_eval(X, Y)
        
    # for size in 
    
    # random_forest_train_and_eval(naive_input, out_vec)
    
    # print("RANDOM FOREST  --- Z")
    # random_forest_train_and_eval(z, out_vec)
    
    
    
    