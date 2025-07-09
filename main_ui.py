from src.config_loader import write_default_configs
write_default_configs()
import customtkinter as ctk
from tkinter import filedialog, messagebox
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from spectral import envi
import torch
import torch.optim as optim
import torch.nn as nn
import glob, os
from src.train import train_model, select_train_test_samples
from src.utils import save_predictions_by_source, calculate_rmse_by_source, print_avg_predicted_ratios
from models.module import SimpleCNN_MLP
from src.preprocessing import Preprocessing
from matplotlib.widgets import RectangleSelector
import matplotlib


matplotlib.use('TkAgg')

# 初始化樣式
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

UPLOAD_DIR = "./uploaded"
OUTPUT_DIR = "./data"
OUTPUT_NPY_PREPROCESSING_DIR = "./preprocessing_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_NPY_PREPROCESSING_DIR, exist_ok=True)

x, y = 333, 220
fixed_width = 25
fixed_height = 500

def sam(hsi_cube, d_point):
    '''
    Calculate Spectral Angle Mapper (SAM) for hyperspectral image data

    Parameters:
        hsi_cube : numpy.ndarray, hyperspectral image data with shape (height, width, bands)
        d_point : numpy.ndarray, reference spectrum with shape (bands,)

    Returns:
        numpy.ndarray : SAM map with shape (height, width)
    '''
    h, w, b = hsi_cube.shape
    r = hsi_cube.reshape(-1, b).T
    rd = np.dot(d_point, r)
    r_abs = np.linalg.norm(r, axis=0)
    d_abs = np.linalg.norm(d_point)
    tmp = rd / (r_abs * d_abs + 1e-8)
    tmp = np.clip(tmp, -1.0, 1.0)
    sam_rd = np.arccos(tmp)
    return sam_rd.reshape(h, w)

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

def process_roi_for_uploaded_hdrs(progress, total_files):
    '''
    Process Region of Interest (ROI) for uploaded HDR files

    Parameters:
        progress : ctk.CTkProgressBar, progress bar widget to update processing status
        total_files : int, total number of HDR files to process
    '''
    processed = 0
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".hdr"):
            hdr_path = os.path.join(UPLOAD_DIR, filename)
            label_result.configure(text=f"處理中: {filename}")
            app.update_idletasks()

            data = envi.open(hdr_path)
            np_data = np.asarray(data.open_memmap(writable=True))
            d_point = np_data[y, x, :]
            sam_map = sam(np_data, d_point)

            fig, ax = plt.subplots()
            img = ax.imshow(sam_map, cmap='jet')
            plt.colorbar(img)
            ax.set_title(f"SAM Map - {filename}\n請點選一次框選固定 ROI ({fixed_width}x{fixed_height})")

            rect_coords = {}

            def onselect(eclick, erelease):
                '''
                Handle rectangle selection for ROI

                Parameters:
                    eclick : matplotlib.backend_bases.MouseEvent, click event data
                    erelease : matplotlib.backend_bases.MouseEvent, release event data
                '''
                x1, y1 = int(eclick.xdata), int(eclick.ydata)
                x2 = x1 + fixed_width
                y2 = y1 + fixed_height

                # 若超出邊界，調整 x1/y1，使得 x2/y2 不會超過影像尺寸
                if x2 > sam_map.shape[1]:
                    x1 = sam_map.shape[1] - fixed_width
                    x2 = sam_map.shape[1]
                if y2 > sam_map.shape[0]:
                    y1 = sam_map.shape[0] - fixed_height
                    y2 = sam_map.shape[0]

                rect_coords['x1'], rect_coords['y1'] = x1, y1
                rect_coords['x2'], rect_coords['y2'] = x2, y2

                rect = plt.Rectangle((x1, y1), fixed_width, fixed_height, edgecolor='red', facecolor='none', lw=2)
                ax.add_patch(rect)
                fig.canvas.draw()
                plt.pause(1)
                plt.close()

            toggle_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], spancoords='pixels', interactive=False)
            plt.show()

            if rect_coords:
                x1, y1 = rect_coords['x1'], rect_coords['y1']
                x2, y2 = rect_coords['x2'], rect_coords['y2']
                mask = np.zeros(sam_map.shape, dtype=bool)
                mask[y1:y2, x1:x2] = True
                roi_spectra = np_data[mask, :]
                npy_name = filename.replace(".hdr", f"_roi_{fixed_width}x{fixed_height}.npy")
                np.save(os.path.join(OUTPUT_DIR, npy_name), roi_spectra)

            processed += 1
            progress.set(int((processed / total_files) * 100))
            app.update_idletasks()

def upload_and_process_files():
    '''
    Upload and process HDR and RAW files, then extract ROI

    Returns:
        None
    '''
    file_paths = filedialog.askopenfilenames(title="選擇 HDR + RAW 檔", filetypes=[("HDR/RAW files", "*.hdr *.raw")])
    if not file_paths:
        messagebox.showerror("錯誤", "未選擇任何檔案")
        return

    hdr_files = [f for f in file_paths if f.endswith(".hdr")]
    raw_files = [f for f in file_paths if f.endswith(".raw")]
    raw_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_files}
    paired_files = []

    for hdr in hdr_files:
        base = os.path.splitext(os.path.basename(hdr))[0]
        if base in raw_dict:
            paired_files.append((hdr, raw_dict[base]))
        else:
            messagebox.showerror("錯誤", f"缺少與 {base}.hdr 配對的 .raw 檔案")
            return

    if not paired_files:
        messagebox.showerror("錯誤", "未找到可配對的檔案")
        return

    label_result.configure(text="開始上傳檔案...")
    progress_bar.set(0)
    app.update_idletasks()

    for i, (hdr, raw) in enumerate(paired_files, start=1):
        shutil.copy(hdr, os.path.join(UPLOAD_DIR, os.path.basename(hdr)))
        shutil.copy(raw, os.path.join(UPLOAD_DIR, os.path.basename(raw)))
        percent = int((i / len(paired_files)) * 100)
        progress_bar.set(percent)
        label_result.configure(text=f"上傳中: {i}/{len(paired_files)} 組（{percent}%）")
        app.update_idletasks()

    label_result.configure(text="✅ 上傳完成，開始 ROI 分析...")
    process_roi_for_uploaded_hdrs(progress_bar, len(paired_files))
    progress_bar.set(100)
    label_result.configure(text="✅ 所有檔案處理完成！")
    messagebox.showinfo("完成", "全部 HDR 已完成 ROI 處理並儲存為 .npy")

def run_model_training(mode, band_num):
    '''
    Train the model and evaluate predictions

    Parameters:
        mode : str, preprocessing mode (SNV, PCA_BANDSELECT, SNV_PCABANDSELECT)
        band_num : int, number of bands for preprocessing

    Returns:
        tuple : (results_by_source, rmse_by_source), containing prediction results and RMSE by source
    '''
    orignal_data_path = "data"
    preprocessin_data_path = "preprocessing_data"
    amplification_factor = 2.0

    # 清空並重新建立預處理資料夾
    if os.path.exists(preprocessin_data_path):
        shutil.rmtree(preprocessin_data_path)
    os.makedirs(preprocessin_data_path, exist_ok=True)

    file_paths = sorted(glob.glob(f'./{orignal_data_path}/*.npy'))
    preprocessing_paths = [os.path.join(f'./{preprocessin_data_path}', os.path.basename(path)) for path in file_paths]

    preprocessor = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)
    preprocessor.preprocess(file_paths, preprocessing_paths)

    # 檢查預處理後的數據形狀
    for path in preprocessing_paths:
        if os.path.exists(path):
            data = np.load(path)
            print(f"Processed {path}: shape = {data.shape}")

    g_mapping = {
        f"./{preprocessin_data_path}/COMPACT100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessin_data_path}/COMPACT100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessin_data_path}/COMPACT5050_RT_roi_25x500.npy": (0.5, 0.5),
        f"./{preprocessin_data_path}/MVS100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessin_data_path}/MVS100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessin_data_path}/MVS5050_RT_roi_25x500.npy": (0.5, 0.5),
        f"./{preprocessin_data_path}/OE100C_RT_roi_25x500.npy": (1.0, 0.0),
        f"./{preprocessin_data_path}/OE100P_RT_roi_25x500.npy": (0.0, 1.0),
        f"./{preprocessin_data_path}/OE5050_RT_roi_25x500.npy": (0.5, 0.5),
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

    results_by_source = save_predictions_by_source(test_sources, pred_Y, test_Y)
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
            "avg_predicted": {
                "cotton": round(avg_cotton, 2),
                "poly": round(avg_poly, 2)
            },
            "rmse": {
                "cotton": round(rmse_by_source[source]['RMSE_cotton'] * 100, 2),
                "poly": round(rmse_by_source[source]['RMSE_poly'] * 100, 2)
            }
        }

    with open("config/output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)


    return results_by_source, rmse_by_source

def plot_spectra_in_gui(mode, band_num):
    '''
    Plot original and processed spectra in the GUI

    Parameters:
        mode : str, preprocessing mode (SNV, PCA_BANDSELECT, SNV_PCABANDSELECT)
        band_num : int, number of bands for preprocessing
    '''
    npy_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")])
    if not npy_files:
        messagebox.showerror("錯誤", "找不到 .npy 檔案")
        return

    file_paths = [os.path.join(OUTPUT_DIR, f) for f in npy_files]
    file_names = [os.path.splitext(f)[0] for f in npy_files]

    orig_spectra = []
    processed_spectra = []

    # 使用 Preprocessing 類處理所有情況
    amplification_factor = 2.0  # 與 run_model_training 一致
    preprocessor = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)

    for path in file_paths:
        arr = np.load(path)
        mean_orig = np.mean(arr, axis=0)
        orig_spectra.append(mean_orig)

        # 使用 Preprocessing 處理數據
        temp_path = os.path.join(OUTPUT_NPY_PREPROCESSING_DIR, "temp.npy")
        temp_processed_path = os.path.join(OUTPUT_NPY_PREPROCESSING_DIR, "temp_processed.npy")
        np.save(temp_path, arr)
        preprocessor.preprocess([temp_path], [temp_processed_path])
        if os.path.exists(temp_processed_path):
            processed_data = np.load(temp_processed_path)
            mean_processed = np.mean(processed_data, axis=0)
        else:
            mean_processed = mean_orig  # 如果處理失敗，顯示原始數據
            print(f"Warning: Preprocessing failed for {path}, using original data.")
        # 清理臨時檔案
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_processed_path):
            os.remove(temp_processed_path)

        processed_spectra.append(mean_processed)

    for widget in frame_spectra.winfo_children():
        widget.destroy()

    left_frame = ctk.CTkFrame(frame_spectra)
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    right_frame = ctk.CTkFrame(frame_spectra)
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for spec, name in zip(orig_spectra, file_names):
        ax1.plot(spec, label=name)
    ax1.set_title("Mean Spectral Reflectance (Original)")
    ax1.set_xlabel("Band")
    ax1.set_ylabel("Reflectance")
    ax1.legend()
    ax1.grid(True)

    canvas1 = FigureCanvasTkAgg(fig1, master=left_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(expand=True, fill="both")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for spec, name in zip(processed_spectra, file_names):
        ax2.plot(spec, label=name)
    # 根據模式設置標題
    # if mode == "SNV_PCA_BANDSELECT":
    #     ax2.set_title("平均光譜反射率 (SNV + PCA_BANDSELECT)")
    # else:
    ax2.set_title(f"Mean Spectral Reflectance ({mode})")
    ax2.set_xlabel("Band")
    ax2.set_ylabel("Reflectance")
    ax2.legend()
    ax2.grid(True)

    canvas2 = FigureCanvasTkAgg(fig2, master=right_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(expand=True, fill="both")



def display_ratio_analysis(mode_selector, band_num_selector):
    '''
    Display ratio analysis and RMSE in the GUI

    Parameters:
        mode_selector : ctk.CTkOptionMenu, widget for selecting preprocessing mode
        band_num_selector : ctk.CTkOptionMenu, widget for selecting number of bands
    '''
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    for widget in frame_other1.winfo_children():
        widget.destroy()

    # 標題
    title_label = ctk.CTkLabel(frame_other1,
                               text="比例與誤差分析",
                               font=ctk.CTkFont(size=32, weight="bold"))
    title_label.pack(pady=(20, 10))

    # 訓練按鈕
    def train_and_plot():
        '''
        Train model and plot results
        '''
        mode = mode_selector.get()
        band_num = int(band_num_selector.get())

        try:
            results_by_source, rmse_by_source = run_model_training(mode, band_num)
        except Exception as e:
            messagebox.showerror("錯誤", f"模型訓練失敗: {str(e)}")
            return

        # 清空結果區域
        for widget in result_frame.winfo_children():
            widget.destroy()

        if not results_by_source:
            messagebox.showerror("錯誤", "無訓練結果，請檢查資料來源")
            return

        # 按類別組織來源
        categories = {
            'MVS': {'100C': None, '100P': None, '5050': None},
            'OE': {'100C': None, '100P': None, '5050': None},
            'COMPACT': {'100C': None, '100P': None, '5050': None}
        }
        for source in results_by_source.keys():
            source_name = os.path.basename(source)
            source_name_lower = source_name.lower()
            for category in categories:
                if category.lower() in source_name_lower:
                    if '100c' in source_name_lower:
                        categories[category]['100C'] = source
                    elif '100p' in source_name_lower:
                        categories[category]['100P'] = source
                    elif '5050' in source_name_lower:
                        categories[category]['5050'] = source

        # 計算未正規化的平均比例（與 print_avg_predicted_ratios 一致）
        avg_ratios_by_category = {}
        for category, proportions in categories.items():
            avg_ratios_by_category[category] = {}
            for prop, source in proportions.items():
                if source and source in results_by_source:
                    data = results_by_source[source]
                    cotton_preds = [row['Predicted_cotton'] for row in data]
                    poly_preds = [row['Predicted_poly'] for row in data]
                    avg_cotton = np.mean(cotton_preds) * 100  # 轉為百分比
                    avg_poly = np.mean(poly_preds) * 100      # 轉為百分比
                    avg_ratios_by_category[category][prop] = np.array([avg_cotton, avg_poly])
                else:
                    avg_ratios_by_category[category][prop] = np.array([50.0, 50.0])

        # 類別選擇下拉選單
        def update_pie_charts(category):
            '''
            Update pie charts for selected category

            Parameters:
                category : str, category name (MVS, OE, COMPACT)
            '''
            for widget in chart_frame.winfo_children():
                widget.destroy()

            ratios = avg_ratios_by_category.get(category, {})
            if not ratios:
                messagebox.showerror("錯誤", f"無 {category} 類別資料")
                return

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
            labels = ['Cotton', 'Polyester']
            colors = ['#66b3ff', '#ff9999']

            # 100C 圓餅圖
            ratio_100c = ratios.get('100C', np.array([50.0, 50.0]))
            # 使用原始比例，但手動計算顯示的百分比
            total_100c = np.sum(ratio_100c)
            if total_100c > 0:
                autopct_100c = [f'{ratio_100c[0]:.2f}%', f'{ratio_100c[1]:.2f}%']
            else:
                autopct_100c = ['50.00%', '50.00%']
            ax1.pie(ratio_100c, labels=labels, colors=colors, autopct=lambda p: autopct_100c.pop(0), startangle=90, textprops={'fontsize': 10})
            ax1.set_title(f"{category} 100C 預測比例", fontsize=12)

            # 100P 圓餅圖
            ratio_100p = ratios.get('100P', np.array([50.0, 50.0]))
            total_100p = np.sum(ratio_100p)
            if total_100p > 0:
                autopct_100p = [f'{ratio_100p[0]:.2f}%', f'{ratio_100p[1]:.2f}%']
            else:
                autopct_100p = ['50.00%', '50.00%']
            ax2.pie(ratio_100p, labels=labels, colors=colors, autopct=lambda p: autopct_100p.pop(0), startangle=90, textprops={'fontsize': 10})
            ax2.set_title(f"{category} 100P 預測比例", fontsize=12)

            # 5050 圓餅圖
            ratio_5050 = ratios.get('5050', np.array([50.0, 50.0]))
            total_5050 = np.sum(ratio_5050)
            if total_5050 > 0:
                autopct_5050 = [f'{ratio_5050[0]:.2f}%', f'{ratio_5050[1]:.2f}%']
            else:
                autopct_5050 = ['50.00%', '50.00%']
            ax3.pie(ratio_5050, labels=labels, colors=colors, autopct=lambda p: autopct_5050.pop(0), startangle=90, textprops={'fontsize': 10})
            ax3.set_title(f"{category} 5050 預測比例", fontsize=12)

            plt.subplots_adjust(wspace=0.4)
            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill="both")
            plt.close(fig)

        # 下拉選單
        category_selector = ctk.CTkOptionMenu(
            result_frame,
            values=['MVS', 'OE', 'COMPACT'],
            command=update_pie_charts,
            width=150
        )
        category_selector.set('MVS')
        category_selector.pack(pady=10)

        # 圓餅圖顯示區域
        chart_frame = ctk.CTkFrame(result_frame)
        chart_frame.pack(fill="x", padx=20, pady=10)

        # 初始顯示 MVS 類別
        update_pie_charts('MVS')

        # RMSE 圖形化顯示
        rmse_frame = ctk.CTkFrame(result_frame)
        rmse_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(rmse_frame,
                     text="各來源 RMSE（柱狀圖）",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0, 10))

        if not rmse_by_source:
            ctk.CTkLabel(rmse_frame,
                         text="無 RMSE 數據，請檢查模型訓練結果",
                         font=ctk.CTkFont(size=16)).pack(pady=10)
        else:
            sources = [os.path.basename(source) for source in rmse_by_source.keys()]
            rmse_cotton = [rmse.get('RMSE_cotton', 0.0) * 100 for rmse in rmse_by_source.values()]
            rmse_poly = [rmse.get('RMSE_poly', 0.0) * 100 for rmse in rmse_by_source.values()]

            fig, ax = plt.subplots(figsize=(10, 4))
            bar_width = 0.35
            index = np.arange(len(sources))

            bars1 = ax.bar(index, rmse_cotton, bar_width, label='Cotton', color='#66b3ff')
            bars2 = ax.bar(index + bar_width, rmse_poly, bar_width, label='Polyester', color='#ff9999')

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}%', ha='center', va='bottom', fontsize=8)

            ax.set_ylabel('RMSE (%)', fontsize=10)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            max_rmse = max(max(rmse_cotton), max(rmse_poly))
            ax.set_ylim(0, max_rmse * 1.2)

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=rmse_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill="both")
            plt.close(fig)

    btn_train = ctk.CTkButton(frame_other1,
                              text="訓練並分析",
                              width=250,
                              height=60,
                              font=ctk.CTkFont(size=18, weight="bold"),
                              command=train_and_plot)
    btn_train.pack(pady=20)

    result_frame = ctk.CTkFrame(frame_other1)
    result_frame.pack(fill="both", expand=True, padx=20, pady=10)

    ctk.CTkLabel(result_frame,
                 text="請點擊「訓練並分析」按鈕開始模型訓練",
                 font=ctk.CTkFont(size=16)).pack(pady=20)

def switch_page(choice, mode_selector, band_num_selector):
    '''
    Switch between GUI pages

    Parameters:
        choice : str, selected page ("顯示光譜" or "顯示比例")
        mode_selector : ctk.CTkOptionMenu, widget for selecting preprocessing mode
        band_num_selector : ctk.CTkOptionMenu, widget for selecting number of bands
    '''
    frame_main.pack_forget()
    frame_other1.pack_forget()

    if choice == "顯示光譜":
        frame_main.pack(fill="both", expand=True, padx=20, pady=20)
    elif choice == "顯示比例":
        frame_other1.pack(fill="both", expand=True, padx=20, pady=20)
        display_ratio_analysis(mode_selector, band_num_selector)

app = ctk.CTk()
app.title("紗線高光譜辨識系統")
app.geometry("1800x1200")

CONTENT_PADX = 20
CONTENT_PADY = 20

start_frame = ctk.CTkFrame(app)
start_frame.pack(fill="both", expand=True)

start_title = ctk.CTkLabel(start_frame,
                           text="紗線高光譜辨識系統",
                           font=ctk.CTkFont(size=42, weight="bold"))
start_title.pack(pady=(200, 20))

start_subtitle = ctk.CTkLabel(start_frame,
                              text="按下開始，進入分析系統",
                              font=ctk.CTkFont(size=20, weight="bold"))
start_subtitle.pack(pady=(0, 40))

btn_start = ctk.CTkButton(start_frame,
                          text="開始",
                          width=200,
                          height=60,
                          font=ctk.CTkFont(size=22, weight="bold"),
                          command=lambda: show_main_screen())
btn_start.pack()

nav_frame = ctk.CTkFrame(app, height=40, corner_radius=0)
frame_main = ctk.CTkFrame(app, fg_color="transparent")
frame_other1 = ctk.CTkFrame(app, corner_radius=0)

def show_main_screen():
    '''
    Display the main GUI screen
    '''
    start_frame.pack_forget()
    nav_frame.pack(side="top", fill="x", padx=CONTENT_PADX, pady=(5, 0))
    frame_main.pack(fill="both", expand=True, padx=CONTENT_PADX, pady=CONTENT_PADY)
    build_main_ui()

def build_main_ui():
    '''
    Build the main UI components
    '''
    # mode 下拉選單
    mode_frame = ctk.CTkFrame(nav_frame)
    mode_frame.pack(side="left", padx=10, pady=5)
    ctk.CTkLabel(mode_frame, text="mode:", font=ctk.CTkFont(size=14)).pack(side="left")
    mode_selector = ctk.CTkOptionMenu(mode_frame,
                                      values=["SNV", "PCA_BANDSELECT", "SNV_PCABANDSELECT"],
                                      width=120)
    mode_selector.set("SNV_PCABANDSELECT")
    mode_selector.pack(side="left", padx=5)

    # band_num 下拉選單
    band_num_frame = ctk.CTkFrame(nav_frame)
    band_num_frame.pack(side="left", padx=10, pady=5)
    ctk.CTkLabel(band_num_frame, text="band_num:", font=ctk.CTkFont(size=14)).pack(side="left")
    band_num_selector = ctk.CTkOptionMenu(band_num_frame,
                                          values=["10", "20", "50", "80", "100", "224"],
                                          width=80)
    band_num_selector.set("50")
    band_num_selector.pack(side="left", padx=5)

    page_selector = ctk.CTkOptionMenu(nav_frame,
                                      values=["顯示光譜", "顯示比例"],
                                      command=lambda choice: switch_page(choice, mode_selector, band_num_selector),
                                      width=150)
    page_selector.set("顯示光譜")
    page_selector.pack(side="left", padx=10, pady=5)

    title_frame = ctk.CTkFrame(frame_main, fg_color="transparent")
    title_frame.pack(fill="x", pady=(30, 10))

    title_label = ctk.CTkLabel(title_frame,
                               text="紗線高光譜辨識系統",
                               font=ctk.CTkFont(size=32, weight="bold"))
    title_label.pack()

    subtitle_label = ctk.CTkLabel(title_frame,
                                  text="請上傳 HDR + RAW 並框選 ROI",
                                  font=ctk.CTkFont(size=20, weight="bold"))
    subtitle_label.pack(pady=(0, 20))

    btn_frame = ctk.CTkFrame(frame_main, fg_color="transparent")
    btn_frame.pack(pady=10, fill="x")
    btn_frame.grid_columnconfigure((0, 1), weight=1)

    btn_upload = ctk.CTkButton(btn_frame,
                               text="📁 上傳並 ROI",
                               width=250,
                               height=60,
                               font=ctk.CTkFont(size=18, weight="bold"),
                               command=upload_and_process_files)
    btn_upload.grid(row=0, column=0, padx=(0, 15), pady=20, sticky="e")

    btn_plot = ctk.CTkButton(btn_frame,
                             text="📊 顯示光譜圖",
                             width=250,
                             height=60,
                             font=ctk.CTkFont(size=18, weight="bold"),
                             command=lambda: plot_spectra_in_gui(mode_selector.get(), int(band_num_selector.get())))
    btn_plot.grid(row=0, column=1, padx=(15, 0), pady=20, sticky="w")

    result_frame = ctk.CTkFrame(frame_main, fg_color="transparent")
    result_frame.pack(fill="x", pady=10)

    global label_result, progress_bar
    label_result = ctk.CTkLabel(result_frame,
                                text="尚未處理任何檔案",
                                wraplength=800,
                                font=ctk.CTkFont(size=20, weight="bold"))
    label_result.pack(anchor="center")

    progress_bar = ctk.CTkProgressBar(result_frame, width=600)
    progress_bar.set(0)
    progress_bar.pack(anchor="center", pady=10)

    global frame_spectra
    frame_spectra = ctk.CTkFrame(frame_main, corner_radius=0)
    frame_spectra.pack(pady=10, fill="both", expand=True)

app.mainloop()