# config.py

class CFG():
    epochs = 10
    batch_size = 64
    SGD_or_SA = "SGD_SA" # "SGD" or "SA" or "SGD_SA"
    can = 10 # only if SA
    T = 0.01 # only if SA
    c = 0.95 # only if SA
    noise_scale = 0.003 # only if SA
    name = "exp01_SA10_T0.01_c0.95_noise0.003" # for save file name
