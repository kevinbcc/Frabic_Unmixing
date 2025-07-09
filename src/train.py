
import numpy as np
import torch
import random
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def select_train_test_samples(
        proportion_mode: Tuple[float, str],
        g_mapping: Dict[str, Tuple[float, float]],
        avg_block_size: int = 100  # 每幾個 pixel 做一次平均
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proportion, mode = proportion_mode

    train_X_list, train_Y_list = [], []
    test_X_list, test_Y_list = [], []
    test_sources: List[str] = []

    for file_name, g_value_tuple in g_mapping.items():
        samples = np.load(file_name)  # (num_pixels, num_bands)

        if samples.ndim != 2:
            raise ValueError(f"Unexpected sample shape in file {file_name}. Expected 2D, got {samples.ndim}D.")

        g_value = torch.tensor(g_value_tuple, dtype=torch.float32).to(device)

        # 打亂像素順序以避免區域性偏差
        np.random.shuffle(samples)

        num_pixels, num_bands = samples.shape
        num_blocks = num_pixels // avg_block_size

        block_averages = []

        for i in range(num_blocks):
            block = samples[i * avg_block_size:(i + 1) * avg_block_size]
            avg_spectrum = np.mean(block, axis=0)  # (num_bands,)
            block_averages.append(avg_spectrum)

        for avg_spectrum in block_averages:
            if random.random() < proportion:
                if mode == "train":
                    train_X_list.append(avg_spectrum)
                    train_Y_list.append(g_value.unsqueeze(0))
                else:
                    test_X_list.append(avg_spectrum)
                    test_Y_list.append(g_value.unsqueeze(0))
                    test_sources.append(file_name)
            else:
                if mode == "train":
                    test_X_list.append(avg_spectrum)
                    test_Y_list.append(g_value.unsqueeze(0))
                    test_sources.append(file_name)
                else:
                    train_X_list.append(avg_spectrum)
                    train_Y_list.append(g_value.unsqueeze(0))

    # 整理成 tensor
    train_X = torch.tensor(np.array(train_X_list), dtype=torch.float32).unsqueeze(1).to(device)
    train_Y = torch.cat(train_Y_list, dim=0).to(device) if train_Y_list else torch.empty((0, 2)).to(device)
    test_X = torch.tensor(np.array(test_X_list), dtype=torch.float32).unsqueeze(1).to(device)
    test_Y = torch.cat(test_Y_list, dim=0).to(device) if test_Y_list else torch.empty((0, 2)).to(device)

    return train_X, train_Y, test_X, test_Y, test_sources

def select_train_test_samples_classification(
        proportion: float,
        g_mapping: Dict[str, int],
        avg_block_size: int = 50
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X_list, train_Y_list = [], []
    test_X_list, test_Y_list = [], []
    test_sources: List[str] = []

    for file_name, class_label in g_mapping.items():
        samples = np.load(file_name)
        if samples.ndim != 2:
            raise ValueError(f"Unexpected sample shape in file {file_name}. Expected 2D, got {samples.ndim}D.")
        np.random.shuffle(samples)
        num_pixels, num_bands = samples.shape
        num_blocks = num_pixels // avg_block_size

        block_averages = [
            np.mean(samples[i * avg_block_size:(i + 1) * avg_block_size], axis=0)
            for i in range(num_blocks)
        ]

        for avg_spectrum in block_averages:
            if random.random() < proportion:
                train_X_list.append(avg_spectrum)
                train_Y_list.append(class_label)
            else:
                test_X_list.append(avg_spectrum)
                test_Y_list.append(class_label)
                test_sources.append(file_name)

    train_X = torch.tensor(np.array(train_X_list), dtype=torch.float32).unsqueeze(1).to(device)
    train_Y = torch.tensor(train_Y_list, dtype=torch.long).to(device)
    test_X = torch.tensor(np.array(test_X_list), dtype=torch.float32).unsqueeze(1).to(device)
    test_Y = torch.tensor(test_Y_list, dtype=torch.long).to(device)

    return train_X, train_Y, test_X, test_Y, test_sources


def train_model(model, train_X, train_Y, epochs, criterion, optimizer, scheduler, batch_size=32, device=None):
    """
    Training loop with GPU support, batch size, and progress bar.

    Args:
        model (nn.Module): Your PyTorch model.
        train_X (torch.Tensor): Input features.
        train_Y (torch.Tensor): Ground truth labels.
        epochs (int): Number of training epochs.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        batch_size (int): Training batch size.
        device (torch.device, optional): Device to use.

    Returns:
        model, best_loss, final_lr
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    best_loss = float('inf')
    final_lr = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_X, batch_Y in progress_bar:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)

            # ✅ 自動處理分類情況
            if isinstance(criterion, nn.CrossEntropyLoss):
                batch_Y = batch_Y.long()

            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch+1}/{epochs}] Total Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")

        final_lr = optimizer.param_groups[0]['lr']

    return model, best_loss, final_lr

def select_train_test_samples_grouped(
    grouped_g_mapping: Dict[str, Dict[str, List]],
    proportion_mode: Tuple[float, str],
    avg_block_size: int = 100,
    random_seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proportion, mode = proportion_mode

    all_X = []
    all_Y = []
    all_sources = []

    # 收集所有平均後的光譜與對應 label
    for group_name, group_info in grouped_g_mapping.items():
        files = group_info["files"]
        g_values = group_info["g_values"]

        for file_path, g_val in zip(files, g_values):
            samples = np.load(file_path)

            if samples.ndim != 2:
                raise ValueError(f"{file_path} has shape {samples.shape}, expected (pixels, bands)")

            np.random.shuffle(samples)
            num_pixels = samples.shape[0]
            num_blocks = num_pixels // avg_block_size

            for i in range(num_blocks):
                block = samples[i * avg_block_size : (i + 1) * avg_block_size]
                avg_spectrum = np.mean(block, axis=0)

                all_X.append(avg_spectrum)
                all_Y.append(g_val)
                all_sources.append(file_path)

    # 轉成 numpy array
    all_X = np.array(all_X, dtype=np.float32)
    all_Y = np.array(all_Y, dtype=np.float32)
    all_sources = np.array(all_sources)

    # 使用 sklearn 的 train_test_split 做穩定比例切分
    train_X_np, test_X_np, train_Y_np, test_Y_np, train_sources, test_sources = train_test_split(
        all_X, all_Y, all_sources, train_size=proportion, random_state=random_seed, shuffle=True
    )

    # 根據 mode 傳回對應資料
    if mode == "train":
        X = torch.tensor(train_X_np).unsqueeze(1).to(device)
        Y = torch.tensor(train_Y_np).to(device)
        sources = train_sources.tolist()
        test_X = torch.tensor(test_X_np).unsqueeze(1).to(device)
        test_Y = torch.tensor(test_Y_np).to(device)
        test_sources = test_sources.tolist()
    else:
        X = torch.tensor(test_X_np).unsqueeze(1).to(device)
        Y = torch.tensor(test_Y_np).to(device)
        sources = test_sources.tolist()
        test_X = torch.tensor(train_X_np).unsqueeze(1).to(device)
        test_Y = torch.tensor(train_Y_np).to(device)
        test_sources = train_sources.tolist()

    return X, Y, test_X, test_Y, test_sources
