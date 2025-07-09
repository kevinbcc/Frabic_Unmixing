import os
from typing import Dict, List
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import csv

def split_array_into_blocks(file_path: str, block_size: int) -> List[np.ndarray]:
    """
    將 .npy 文件中的 3D 數組切割為小塊。

    參數：
        file_path (str): .npy 文件的路徑。
        block_size (int): 每個塊的目標高度和寬度（正方形塊）。

    返回：
        List[np.ndarray]: 包含所有分塊的列表。
    """
    # 加載數據
    R = np.load(file_path)

    # 檢查數據形狀
    if len(R.shape) != 3:
        raise ValueError("The input array must be a 3D array.")

    H, W, D = R.shape  # 獲取數據的形狀
    blocks = []

    # 遍歷分塊
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            # 計算塊的範圍，確保邊界情況下不越界
            end_i = min(i + block_size, H)
            end_j = min(j + block_size, W)

            # 提取當前小塊
            block = R[i:end_i, j:end_j, :]
            blocks.append(block)

    return blocks



def flatten_3d_to_2d(array: np.ndarray) -> np.ndarray:
    """
    将 3D 矩阵展平为 2D 矩阵 (H * W, D)。

    参数：
        array (np.ndarray): 输入的 3D 矩阵，形状为 (H, W, D)。

    返回：
        np.ndarray: 转换后的 2D 矩阵，形状为 (H * W, D)。
    """
    if len(array.shape) != 3:
        raise ValueError("The input array must be a 3D matrix.")

    # 展平为二维矩阵 (H * W, D)
    return array.reshape(-1, array.shape[2])

def mean_spectral(array: np.ndarray) -> np.ndarray:
    """
    展平 3D 矩阵并对展平后的 2D 矩阵按列求平均值。

    参数：
        array (np.ndarray): 输入的 3D 矩阵，形状为 (H, W, D)。

    返回：
        np.ndarray: 展平并按列平均后的 1D 向量，长度为 D。
    """
    # 调用 flatten_3d_to_2d 函数
    flattened_array = flatten_3d_to_2d(array)

    # 按列求平均值
    column_means = np.mean(flattened_array, axis=0)

    return column_means


def save_predictions_by_source(test_sources, pred_Y, test_Y, output_dir="result"):
    results_by_source = defaultdict(list)

    for i, source_path in enumerate(test_sources):
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        results_by_source[source_name].append({
            'Predicted_cotton': pred_Y[i, 0].item(),
            'Predicted_poly': pred_Y[i, 1].item(),
            'Actual_cotton': test_Y[i, 0].item(),
            'Actual_poly': test_Y[i, 1].item()
        })

    os.makedirs(output_dir, exist_ok=True)

    for source_name, rows in results_by_source.items():
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{output_dir}/{source_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",  # 支援中文
            sep=",",
            quoting=csv.QUOTE_NONNUMERIC  # 數字加引號保護格式
        )

    print(f"分類預測結果已儲存到資料夾：{output_dir}")
    return results_by_source


def save_predictions_by_source_classification(test_sources, pred_classes, test_Y, output_dir="result"):
    results_by_source = defaultdict(list)

    for i, source_path in enumerate(test_sources):
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        predicted_class = int(pred_classes[i].item())  # ← 直接用 class index
        actual_class = int(test_Y[i].item())

        results_by_source[source_name].append({
            'Predicted_class': predicted_class,
            'Actual_class': actual_class
        })

    os.makedirs(output_dir, exist_ok=True)

    for source_name, rows in results_by_source.items():
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{output_dir}/{source_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
            sep=",",
            quoting=csv.QUOTE_NONNUMERIC
        )

    print(f"分類預測結果已儲存到資料夾：{output_dir}")
    return results_by_source


def calculate_rmse_by_source(results_by_source, save_csv_path=None):
    rmse_by_source = {}

    for source_name, rows in results_by_source.items():
        df = pd.DataFrame(rows)

        actual_cotton = df['Actual_cotton'].values
        predicted_cotton = df['Predicted_cotton'].values
        actual_poly = df['Actual_poly'].values
        predicted_poly = df['Predicted_poly'].values

        rmse_cotton = np.sqrt(mean_squared_error(actual_cotton, predicted_cotton))
        rmse_poly = np.sqrt(mean_squared_error(actual_poly, predicted_poly))

        rmse_by_source[source_name] = {
            'RMSE_cotton': (rmse_cotton),
            'RMSE_poly': (rmse_poly)
        }

    print("\n===== 各來源的 RMSE 結果 =====")
    for source_name, rmse in rmse_by_source.items():
        print(f"{source_name} - RMSE_cotton: {rmse['RMSE_cotton']:.4f}, RMSE_poly: {rmse['RMSE_poly']:.4f}")

    if save_csv_path:
        rmse_df = pd.DataFrame.from_dict(rmse_by_source, orient='index')
        rmse_df.index.name = "Source"
        rmse_df.to_csv(
            save_csv_path,
            encoding="utf-8-sig",
            sep=",",
            quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"RMSE 結果已儲存至：{save_csv_path}")

    return rmse_by_source


def print_avg_predicted_ratios(results_by_source):
    print("\n===== 各紗種的預測成分平均比例（百分比） =====")
    for source_name, rows in results_by_source.items():
        df = pd.DataFrame(rows)
        avg_pred_cotton = df['Predicted_cotton'].mean() * 100
        avg_pred_poly = df['Predicted_poly'].mean() * 100
        print(f"{source_name} - Predicted Cotton: {avg_pred_cotton:.2f}%, Predicted Poly: {avg_pred_poly:.2f}%")



def process_npy_to_blocks(folder: str, block_size: int) -> Dict[str, List[np.ndarray]]:
    """
    Read .npy files from the folder, split into blocks, and apply mean_spectral.

    Returns:
        Dict[str, List[np.ndarray]]: Mapping from file path to list of processed blocks.
    """
    data_dir = f"./{folder}/"
    file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

    all_blocks: Dict[str, List[np.ndarray]] = {}
    for file_name in file_list:
        blocks = split_array_into_blocks(file_name, block_size)
        all_blocks[file_name] = [mean_spectral(block) for block in blocks]

    return all_blocks