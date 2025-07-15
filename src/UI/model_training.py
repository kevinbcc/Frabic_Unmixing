import os
import shutil
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.train import train_model, select_train_test_samples
from src.utils import save_predictions_by_source, calculate_rmse_by_source, print_avg_predicted_ratios
from models.module import SimpleCNN_MLP
from src.preprocessing import Preprocessing

OUTPUT_NPY_PREPROCESSING_DIR = "./preprocessing_data"
os.makedirs(OUTPUT_NPY_PREPROCESSING_DIR, exist_ok=True)

def snv(input_data):
    '''
    Apply Standard Normal Variate (SNV) transformation to input data

    Parameters:
        input_data : numpy.ndarray, input data with shape (..., bands)

    Returns:
        numpy.ndarray : SNV-transformed data with same shape as input
    '''
    d = len(input_data.shape) - 1
    mean = input_data.mean(d)
    std = input_data.std(d)
    res = (input_data - mean[..., None]) / std[..., None]
    return res

def run_model_training(mode: str, band_num: int):
    '''
    Train the model and evaluate predictions

    Parameters:
        mode : str, preprocessing mode (SNV, PCA_BANDSELECT, SNV_PCABANDSELECT)
        band_num : int, number of bands for preprocessing

    Returns:
        tuple : (results_by_source, rmse_by_source), containing prediction results and RMSE by source
    '''
    original_data_path = "data"
    preprocessing_data_path = "preprocessing_data"
    amplification_factor = 2.0

    if os.path.exists(preprocessing_data_path):
        shutil.rmtree(preprocessing_data_path)
    os.makedirs(preprocessing_data_path, exist_ok=True)

    file_paths = sorted(glob.glob(f'./{original_data_path}/*.npy'))
    preprocessing_paths = [os.path.join(f'./{preprocessing_data_path}', os.path.basename(path)) for path in file_paths]

    preprocessor = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)
    preprocessor.preprocess(file_paths, preprocessing_paths)


    # 檢查預處理後的數據形狀
    for path in preprocessing_paths:
        if os.path.exists(path):
            data = np.load(path)
            print(f"Processed {path}: shape = {data.shape}")

    g_mapping = {
        f"./{preprocessing_data_path}/COMPACT100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessing_data_path}/COMPACT100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessing_data_path}/COMPACT5050_RT_roi_25x500.npy": (0.5, 0.5),
        f"./{preprocessing_data_path}/MVS100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessing_data_path}/MVS100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessing_data_path}/MVS5050_RT_roi_25x500.npy": (0.5, 0.5),
        f"./{preprocessing_data_path}/OE100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessing_data_path}/OE100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessing_data_path}/OE5050_RT_roi_25x500.npy": (0.5, 0.5),
    }

    train_X, train_Y, test_X, test_Y, test_sources = select_train_test_samples(
        proportion_mode=(0.1, "train"), g_mapping=g_mapping
    )


    # 檢查訓練數據形狀
    print("train_X shape:", train_X.shape)
    print("test_X shape:", test_X.shape)


    model = SimpleCNN_MLP(input_channels=1, input_dim=band_num, hidden_dim=64, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    model, best_loss, final_lr = train_model(
        model, train_X, train_Y,
        epochs=100, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, batch_size=64
    )

    model.eval()
    with torch.no_grad():
        pred_Y = model(test_X)

    # 儲存預測結果並分類
    results_by_source = save_predictions_by_source(test_sources, pred_Y, test_Y)

    # 計算並顯示/儲存 RMSE
    rmse_by_source = calculate_rmse_by_source(results_by_source, save_csv_path="result/sourcewise_rmse.csv")

    # 顯示每一種紗種的真實成分平均比例（Cotton / Poly）
    print_avg_predicted_ratios(results_by_source)

    output_json = {}
    for source, records in results_by_source.items():
        cotton_preds = [r['Predicted_cotton'] for r in records]
        poly_preds = [r['Predicted_poly'] for r in records]
        avg_cotton = np.mean(cotton_preds) * 100
        avg_poly = np.mean(poly_preds) * 100

        source_name = os.path.splitext(os.path.basename(source))[0]
        output_json[source_name] = {
            "avg_predicted": {"cotton": round(avg_cotton, 2), "poly": round(avg_poly, 2)},
            "rmse": {
                "cotton": round(rmse_by_source[source]['RMSE_cotton'] * 100, 2),
                "poly": round(rmse_by_source[source]['RMSE_poly'] * 100, 2)
            }
        }

    with open("config/output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    return results_by_source, rmse_by_source