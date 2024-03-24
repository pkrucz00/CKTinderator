import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Z_VEC_LOC = "/media/pawel/DATA/tmp/freddie_mercuries/z_less_iter.npy"


def load_data():
    naive_input = np.load("/media/pawel/DATA/tmp/freddie_mercuries/naive_input.npy")
    z = np.load(Z_VEC_LOC)
    out_vec = np.load("./data/outvec.npy")
    
    return naive_input, z, out_vec


def random_forest_train_and_eval(naive_input, out_vec):
    naive_input, out_vec = scale(naive_input), scale(out_vec)
    X_train, X_test, y_train, y_test = train_test_split(naive_input, out_vec, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    print(f"Train: {model.score(X_train, y_train)}")
    print(f"Test: {model.score(X_test, y_test)}")

def scale(X):
    skaler = StandardScaler()
    skaler.fit(X)
    X = skaler.transform(X)
    return X

if __name__ == '__main__':
    naive_input, z, out_vec = load_data()
    naive_inputs = [naive_input[:i] for i in [100]]
    out_vecs = [out_vec[:i] for i in [100]]
    # naive_input, z, out_vec = naive_input[:100], z[:100], out_vec[:100]
    print("Data has been loaded")
    print(naive_input.shape, z.shape, out_vec.shape)
    
    
    naive_input = scale(naive_input)
    z = scale(z)
    out_vec = scale(out_vec)
    print("Data has been scaled")
    
    
    print("RANDOM FOREST  --- NAIVE INPUT")
    for X, Y in zip(naive_inputs, out_vecs):
        print(f"Size: {X.shape[0]}")
        random_forest_train_and_eval(naive_input, out_vec)
    # random_forest_train_and_eval(naive_input, out_vec)
    
    # print("RANDOM FOREST  --- Z")
    # random_forest_train_and_eval(z, out_vec)
    
    
    
    