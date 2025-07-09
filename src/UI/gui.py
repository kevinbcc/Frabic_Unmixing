import customtkinter as ctk
from src.UI.data_processing import upload_and_process_files
from src.UI.plotting import plot_spectra_in_gui, display_ratio_analysis

class HyperspectralGUI:
    '''
    A GUI class for hyperspectral yarn identification system

    Parameters:
        app : ctk.CTk, the main application window
    '''
    def __init__(self, app):
        self.app = app
        self.CONTENT_PADX = 20
        self.CONTENT_PADY = 20
        
        self.nav_frame = ctk.CTkFrame(app, height=40, corner_radius=0)
        self.frame_main = ctk.CTkFrame(app, fg_color="transparent")
        self.frame_other1 = ctk.CTkFrame(app, corner_radius=0)
        self.start_frame = ctk.CTkFrame(app)
        
        self.label_result = None
        self.progress_bar = None
        self.frame_spectra = None
        
    def setup_start_screen(self):
        '''
        Set up the initial start screen with title and start button
        '''
        self.start_frame.pack(fill="both", expand=True)
        
        start_title = ctk.CTkLabel(self.start_frame,
                                 text="紗線高光譜辨識系統",
                                 font=ctk.CTkFont(size=42, weight="bold"))
        start_title.pack(pady=(200, 20))
        
        start_subtitle = ctk.CTkLabel(self.start_frame,
                                    text="按下開始，進入分析系統",
                                    font=ctk.CTkFont(size=20, weight="bold"))
        start_subtitle.pack(pady=(0, 40))
        
        btn_start = ctk.CTkButton(self.start_frame,
                                text="開始",
                                width=200,
                                height=60,
                                font=ctk.CTkFont(size=22, weight="bold"),
                                command=self.show_main_screen)
        btn_start.pack()
    
    def show_main_screen(self):
        '''
        Display the main GUI screen with navigation and main content frames
        '''
        self.start_frame.pack_forget()
        self.nav_frame.pack(side="top", fill="x", padx=self.CONTENT_PADX, pady=(5, 0))
        self.frame_main.pack(fill="both", expand=True, padx=self.CONTENT_PADX, pady=self.CONTENT_PADY)
        self.build_main_ui()
    
    def build_main_ui(self):
        '''
        Build the main UI components including mode selection, band selection, and buttons
        '''
        mode_frame = ctk.CTkFrame(self.nav_frame)
        mode_frame.pack(side="left", padx=10, pady=5)
        ctk.CTkLabel(mode_frame, text="mode:", font=ctk.CTkFont(size=14)).pack(side="left")
        self.mode_selector = ctk.CTkOptionMenu(mode_frame,
                                             values=["SNV", "PCA_BANDSELECT", "SNV_PCABANDSELECT"],
                                             width=120)
        self.mode_selector.set("SNV_PCABANDSELECT")
        self.mode_selector.pack(side="left", padx=5)
        
        band_num_frame = ctk.CTkFrame(self.nav_frame)
        band_num_frame.pack(side="left", padx=10, pady=5)
        ctk.CTkLabel(band_num_frame, text="band_num:", font=ctk.CTkFont(size=14)).pack(side="left")
        self.band_num_selector = ctk.CTkOptionMenu(band_num_frame,
                                                 values=["10", "20", "50", "80", "100", "224"],
                                                 width=80)
        self.band_num_selector.set("50")
        self.band_num_selector.pack(side="left", padx=5)
        
        page_selector = ctk.CTkOptionMenu(self.nav_frame,
                                        values=["顯示光譜", "顯示比例"],
                                        command=lambda choice: self.switch_page(choice),
                                        width=150)
        page_selector.set("顯示光譜")
        page_selector.pack(side="left", padx=10, pady=5)
        
        title_frame = ctk.CTkFrame(self.frame_main, fg_color="transparent")
        title_frame.pack(fill="x", pady=(30, 10))
        
        ctk.CTkLabel(title_frame,
                    text="紗線高光譜辨識系統",
                    font=ctk.CTkFont(size=32, weight="bold")).pack()
        ctk.CTkLabel(title_frame,
                    text="請上傳 HDR + RAW 並框選 ROI",
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0, 20))
        
        btn_frame = ctk.CTkFrame(self.frame_main, fg_color="transparent")
        btn_frame.pack(pady=10, fill="x")
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        
        btn_upload = ctk.CTkButton(btn_frame,
                                 text="📁 上傳並 ROI",
                                 width=250,
                                 height=60,
                                 font=ctk.CTkFont(size=18, weight="bold"),
                                 command=lambda: upload_and_process_files(
                                 self.label_result, self.progress_bar, self.app
                                )
                            )
        btn_upload.grid(row=0, column=0, padx=(0, 15), pady=20, sticky="e")
        
        btn_plot = ctk.CTkButton(btn_frame,
                               text="📊 顯示光譜圖",
                               width=250,
                               height=60,
                               font=ctk.CTkFont(size=18, weight="bold"),
                               command=lambda: plot_spectra_in_gui(self.mode_selector.get(), int(self.band_num_selector.get()), self.frame_main))
        btn_plot.grid(row=0, column=1, padx=(15, 0), pady=20, sticky="w")
        
        result_frame = ctk.CTkFrame(self.frame_main, fg_color="transparent")
        result_frame.pack(fill="x", pady=10)
        
        self.label_result = ctk.CTkLabel(result_frame,
                                       text="尚未處理任何檔案",
                                       wraplength=800,
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.label_result.pack(anchor="center")
        
        self.progress_bar = ctk.CTkProgressBar(result_frame, width=600)
        self.progress_bar.set(0)
        self.progress_bar.pack(anchor="center", pady=10)
        
        self.frame_spectra = ctk.CTkFrame(self.frame_main, corner_radius=0)
        self.frame_spectra.pack(pady=10, fill="both", expand=True)
    
    def switch_page(self, choice):
        '''
        Switch between GUI pages

        Parameters:
            choice : str, selected page ("顯示光譜" or "顯示比例")
            mode_selector : ctk.CTkOptionMenu, widget for selecting preprocessing mode
            band_num_selector : ctk.CTkOptionMenu, widget for selecting number of bands
        '''
        self.frame_main.pack_forget()
        self.frame_other1.pack_forget()
        
        if choice == "顯示光譜":
            self.frame_main.pack(fill="both", expand=True, padx=self.CONTENT_PADX, pady=self.CONTENT_PADY)
        elif choice == "顯示比例":
            self.frame_other1.pack(fill="both", expand=True, padx=self.CONTENT_PADX, pady=self.CONTENT_PADY)
            display_ratio_analysis(self.mode_selector, self.band_num_selector, self.frame_other1)