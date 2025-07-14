import numpy as np

for i in range(41, 61):
    vp_file = np.load(f"data/velocity_models/vp_{i}.npy")
    density = 0.31 * np.power(vp_file, 0.25)
    np.save(f"data/velocity_models/density_{i}.npy", density)
    
