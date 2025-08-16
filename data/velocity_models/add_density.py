import numpy as np

for i in range(101, 109):
    vp_file = np.load(f"data/test/velocity_models/vp_{i}.npy")
    density = 0.31 * np.power(vp_file, 0.25)
    np.save(f"data/test/velocity_models/density_{i}.npy", density)
    
