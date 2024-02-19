

def get_tcnn_config(mlp_depth, mlp_width, output_activation="Sigmoid"):
    tinycudann_meta = {
        "encoding": {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": output_activation,
            "n_neurons": mlp_width,
            "n_hidden_layers": mlp_depth
        }}
    return tinycudann_meta