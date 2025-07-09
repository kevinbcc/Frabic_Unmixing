import os
import shutil
from tkinter import filedialog, messagebox
import numpy as np
from spectral import envi
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import customtkinter as ctk

UPLOAD_DIR = "./uploaded"
OUTPUT_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def process_roi_for_uploaded_hdrs(progress: ctk.CTkProgressBar, total_files: int, label_result: ctk.CTkLabel, app):
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

def upload_and_process_files(label_result, progress_bar, app):
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
    process_roi_for_uploaded_hdrs(progress_bar, len(paired_files), label_result, app)
    progress_bar.set(100)
    label_result.configure(text="✅ 所有檔案處理完成！")
    messagebox.showinfo("完成", "全部 HDR 已完成 ROI 處理並儲存為 .npy")