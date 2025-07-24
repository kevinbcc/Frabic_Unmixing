import customtkinter as ctk
from src.UI.gui import HyperspectralGUI
from src.config_loader import write_default_configs


write_default_configs()

# 初始化樣式
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

# 調整視窗大小
app = ctk.CTk()
app.title("紗線高光譜辨識系統")
app.geometry("1800x1200")

gui = HyperspectralGUI(app)
gui.setup_start_screen()

app.mainloop()
