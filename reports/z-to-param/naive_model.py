import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    naive_input = np.load("/media/pawel/DATA/tmp/freddie_mercuries/naive_input.npy")
    z = np.load("/media/pawel/DATA/tmp/freddie_mercuries/z.npy")
    out_vec = np.load("./data/outvec.npy")
    
    return naive_input, z, out_vec


if __name__ == '__main__':
    naive_input, z, out_vec = load_data()
    print("Loaded data")
    
    skaler = StandardScaler()
    skaler.fit(naive_input)
    skaler.fit(z)
    print("Data scaled")
    
    X_train, X_test, y_train, y_test = train_test_split(naive_input, out_vec, test_size=0.2, random_state=42)
    print("Data split")
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train[:100]. y_train[:100])
    
    print(f"Train: {model.score(X_train, y_train)}")
    print(f"Test: {model.score(X_test, y_test)}")
    
    
    