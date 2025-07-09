import os
import yaml

CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)

default_configs = {
    "ui_config.yaml": {
        "appearance_mode": "light",
        "theme_color": "green",
        "window_size": {
            "width": 1800,
            "height": 1200
        }
    },
    "path_config.yaml": {
        "paths": {
            "upload_dir": "uploaded",
            "output_dir": "data",
            "preprocessed_dir": "preprocessing_data",
            "result_dir": "result"
        }
    },
    "model_config.yaml": {
        "model": {
            "hidden_dim": 64,
            "output_dim": 2
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.01
        }
    },
    "preprocess_config.yaml": {
        "preprocessing": {
            "amplification_factor": 2.0,
            "band_options": [10, 20, 50, 80, 100, 224],
            "default_mode": "SNV"
        }
    },
    "data_mapping.yaml": {
        "mapping": {
            "COMPACT100C_RT_roi_25x500.npy": [1.0, 0.0],
            "COMPACT100P_RT_roi_25x500.npy": [0.0, 1.0],
            "COMPACT5050_RT_roi_25x500.npy": [0.5, 0.5],
            "MVS100C_RT_roi_25x500.npy": [1.0, 0.0],
            "MVS100P_RT_roi_25x500.npy": [0.0, 1.0],
            "MVS5050_RT_roi_25x500.npy": [0.5, 0.5],
            "OE100C_RT_roi_25x500.npy": [1.0, 0.0],
            "OE100P_RT_roi_25x500.npy": [0.0, 1.0],
            "OE5050_RT_roi_25x500.npy": [0.5, 0.5]
        }
    }
}

def write_default_configs():
    for filename, content in default_configs.items():
        path = os.path.join(CONFIG_DIR, filename)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(content, f, allow_unicode=True)
            print(f"✅ Created default config: {path}")
        else:
            print(f"✅ Found existing config: {path}")
