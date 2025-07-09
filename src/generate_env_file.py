import os

# 設定 .env 存放資料夾
CONFIG_DIR = "config"
os.makedirs(CONFIG_DIR, exist_ok=True)

# .env 內容字串（可依需求動態產生）
env_content = """# --- UI 設定 ---
APPEARANCE_MODE=light
THEME_COLOR=green
WINDOW_WIDTH=1800
WINDOW_HEIGHT=1200

# --- 資料夾路徑 ---
UPLOAD_DIR=uploaded
OUTPUT_DIR=data
PREPROCESSED_DIR=preprocessing_data
RESULT_DIR=result

# --- 模型參數 ---
HIDDEN_DIM=64
OUTPUT_DIM=2

# --- 訓練參數 ---
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.01

# --- 前處理參數 ---
AMPLIFICATION_FACTOR=2.0
BAND_OPTIONS=10,20,50,80,100,224
DEFAULT_MODE=SNV
"""

# 寫入 config/.env 檔案
env_path = os.path.join(CONFIG_DIR, ".env")
with open(env_path, "w", encoding="utf-8") as f:
    f.write(env_content)

print(f"✅ .env 檔案已產生：{env_path}")
