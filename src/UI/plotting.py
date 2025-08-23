import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import messagebox
from src.preprocessing import Preprocessing
from src.UI.model_training import run_model_training

OUTPUT_DIR = "./data"
OUTPUT_NPY_PREPROCESSING_DIR = "./preprocessing_data"

def plot_spectra_in_gui(mode: str, band_num: int, frame_main):
    '''
    Plot original and processed spectra in the GUI

    Parameters:
        mode : str, preprocessing mode (SNV, PCA_BANDSELECT, SNV_PCABANDSELECT)
        band_num : int, number of bands for preprocessing
    '''
    # 1) 收集檔案
    npy_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")])
    if not npy_files:
        messagebox.showerror("錯誤", "找不到 .npy 檔案")
        return

    file_paths = [os.path.join(OUTPUT_DIR, f) for f in npy_files]
    file_names = [os.path.splitext(f)[0] for f in npy_files]

    # 2) 原始平均光譜
    orig_spectra = [np.mean(np.load(p), axis=0) for p in file_paths]

    # 3) 批次前處理（關鍵差異：一次把所有檔案丟進 preprocess）
    amplification_factor = 2.0
    preprocessor = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)

    # 清乾淨暫存輸出資料夾，避免殘留舊檔
    os.makedirs(OUTPUT_NPY_PREPROCESSING_DIR, exist_ok=True)
    for _f in os.listdir(OUTPUT_NPY_PREPROCESSING_DIR):
        _p = os.path.join(OUTPUT_NPY_PREPROCESSING_DIR, _f)
        try:
            os.remove(_p)
        except IsADirectoryError:
            shutil.rmtree(_p)

    # 產生對應輸出路徑，與原始檔名一一對應
    proc_paths = [os.path.join(OUTPUT_NPY_PREPROCESSING_DIR, os.path.basename(p)) for p in file_paths]

    # ★ 一次性呼叫，PCA 會用 vstack(全部資料) 來決定同一組 band
    preprocessor.preprocess(file_paths, proc_paths)

    # 4) 前處理後的平均光譜
    processed_spectra = [np.mean(np.load(p), axis=0) for p in proc_paths]

    # 5) 繪圖（沿用你原本的 GUI 佈局）
    frame_spectra = frame_main.winfo_children()[-1]
    for widget in frame_spectra.winfo_children():
        widget.destroy()

    left_frame = ctk.CTkFrame(frame_spectra)
    left_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)

    right_frame = ctk.CTkFrame(frame_spectra)
    right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    # 原始
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

    # 前處理後（PCA_BANDSELECT / SNV_PCABANDSELECT 等）
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for spec, name in zip(processed_spectra, file_names):
        ax2.plot(spec, label=name)
    ax2.set_title(f"Mean Spectral Reflectance ({mode})")
    ax2.set_xlabel("Band")
    ax2.set_ylabel("Reflectance")
    ax2.legend()
    ax2.grid(True)
    canvas2 = FigureCanvasTkAgg(fig2, master=right_frame)
    canvas2.draw()
    canvas2.get_tk_widget().pack(expand=True, fill="both")


def display_ratio_analysis(mode_selector, band_num_selector, frame_other1):
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

    title_label = ctk.CTkLabel(frame_other1,
                             text="比例與誤差分析",
                             font=ctk.CTkFont(size=32, weight="bold"))
    title_label.pack(pady=(20, 10))

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

        for widget in result_frame.winfo_children():
            widget.destroy()

        if not results_by_source:
            messagebox.showerror("錯誤", "無訓練結果，請檢查資料來源")
            return

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

        avg_ratios_by_category = {}
        for category, proportions in categories.items():
            avg_ratios_by_category[category] = {}
            for prop, source in proportions.items():
                if source and source in results_by_source:
                    data = results_by_source[source]
                    cotton_preds = [row['Predicted_cotton'] for row in data]
                    poly_preds = [row['Predicted_poly'] for row in data]
                    avg_cotton = np.mean(cotton_preds) * 100
                    avg_poly = np.mean(poly_preds) * 100
                    avg_ratios_by_category[category][prop] = np.array([avg_cotton, avg_poly])
                else:
                    avg_ratios_by_category[category][prop] = np.array([50.0, 50.0])

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

            ratio_100c = ratios.get('100C', np.array([50.0, 50.0]))
            total_100c = np.sum(ratio_100c)
            autopct_100c = [f'{ratio_100c[0]:.2f}%', f'{ratio_100c[1]:.2f}%'] if total_100c > 0 else ['50.00%', '50.00%']
            ax1.pie(ratio_100c, labels=labels, colors=colors, autopct=lambda p: autopct_100c.pop(0), startangle=90, textprops={'fontsize': 10})
            ax1.set_title(f"{category} 100C 預測比例", fontsize=12)

            ratio_100p = ratios.get('100P', np.array([50.0, 50.0]))
            total_100p = np.sum(ratio_100p)
            autopct_100p = [f'{ratio_100p[0]:.2f}%', f'{ratio_100p[1]:.2f}%'] if total_100p > 0 else ['50.00%', '50.00%']
            ax2.pie(ratio_100p, labels=labels, colors=colors, autopct=lambda p: autopct_100p.pop(0), startangle=90, textprops={'fontsize': 10})
            ax2.set_title(f"{category} 100P 預測比例", fontsize=12)

            ratio_5050 = ratios.get('5050', np.array([50.0, 50.0]))
            total_5050 = np.sum(ratio_5050)
            autopct_5050 = [f'{ratio_5050[0]:.2f}%', f'{ratio_5050[1]:.2f}%'] if total_5050 > 0 else ['50.00%', '50.00%']
            ax3.pie(ratio_5050, labels=labels, colors=colors, autopct=lambda p: autopct_5050.pop(0), startangle=90, textprops={'fontsize': 10})
            ax3.set_title(f"{category} 5050 預測比例", fontsize=12)

            plt.subplots_adjust(wspace=0.4)
            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill="both")
            plt.close(fig)

        category_selector = ctk.CTkOptionMenu(result_frame,
                                            values=['MVS', 'OE', 'COMPACT'],
                                            command=update_pie_charts,
                                            width=150)
        category_selector.set('MVS')
        category_selector.pack(pady=10)

        chart_frame = ctk.CTkFrame(result_frame)
        chart_frame.pack(fill="x", padx=20, pady=10)
        update_pie_charts('MVS')

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