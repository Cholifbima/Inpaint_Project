"""
Manga Text Inpainting Tool - Tabbed Interface
Tab 1: Restoration Mode | Tab 2: Evaluation Mode
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict

from inpainter import HybridInpainter

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARNING] scikit-image not found. pip install scikit-image")


class MangaCleanerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manga Inpainting - Adaptive Hybrid Pyramid")
        self.root.geometry("1280x850")
        self.root.configure(bg='#f0f0f0')
        
        # Restoration state
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.display_mask: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        self.scale_factor = 1.0
        self.max_display_width, self.max_display_height = 750, 600
        self.is_drawing, self.brush_size = False, 15
        self.last_x, self.last_y = None, None
        self.auto_snap_enabled, self.stroke_bbox = True, None
        self.mask_history, self.history_index, self.max_history = [], -1, 50
        self.show_result = False
        self.zoom_level, self.zoom_min, self.zoom_max = 1.0, 0.5, 4.0
        self.pan_offset_x, self.pan_offset_y = 0, 0
        self.is_panning, self.pan_start_x, self.pan_start_y = False, 0, 0
        self.inpainter = HybridInpainter(target_proc_size=450, patch_size=9)
        self.is_processing = False
        self.progress_queue = queue.Queue()
        self.canvas_image_id, self.tk_image = None, None
        
        # Evaluation state
        self.eval_ground_truth: Optional[np.ndarray] = None
        self.eval_mask: Optional[np.ndarray] = None
        self.eval_damaged: Optional[np.ndarray] = None
        self.eval_results: Dict[str, Dict] = {}
        self.eval_is_running = False
        self.eval_queue = queue.Queue()
        
        self._setup_styles()
        self._create_notebook_ui()
        self._check_progress_queue()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab', 
                  background=[('selected', '#3498db'), ('!selected', '#bdc3c7')],
                  foreground=[('selected', 'white'), ('!selected', '#2c3e50')])
        style.configure('Treeview', font=('Segoe UI', 10), rowheight=30)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))
    
    def _create_notebook_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.restoration_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.restoration_frame, text='  Mode Restorasi  ')
        self._create_restoration_tab()
        
        self.evaluation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_frame, text='  Mode Evaluasi  ')
        self._create_evaluation_tab()
    
    def _create_restoration_tab(self):
        main = tk.Frame(self.restoration_frame, bg='#f0f0f0')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        left = tk.Frame(main, bg='#ffffff', relief='solid', bd=1)
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.upload_frame = tk.Frame(left, bg='#ffffff')
        self.upload_frame.pack(fill='both', expand=True)
        tk.Label(self.upload_frame, text="Klik untuk memuat gambar manga", font=('Segoe UI', 14), bg='#ffffff', fg='#7f8c8d').pack(expand=True)
        tk.Button(self.upload_frame, text="Buka Gambar", font=('Segoe UI', 12, 'bold'), bg='#3498db', fg='white', padx=30, pady=10, relief='flat', command=self._upload_image).pack(pady=20)
        
        self.canvas_frame = tk.Frame(left, bg='#ffffff')
        self.canvas_label = tk.Label(self.canvas_frame, text="Gambar mask di area yang ingin dihapus", font=('Segoe UI', 10), bg='#ffffff', fg='#7f8c8d')
        self.canvas_label.pack(pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg='#e0e0e0', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
        
        right = tk.Frame(main, bg='#ecf0f1', width=280)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        self._create_restoration_controls(right)
        
        bottom = tk.Frame(self.restoration_frame, bg='#f0f0f0')
        bottom.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        self.progress_bar = ttk.Progressbar(bottom, mode='indeterminate', length=300)
        self.progress_bar.pack(side='left', padx=(0, 10))
        self.status_label = tk.Label(bottom, text="Siap. Muat gambar untuk mulai.", font=('Segoe UI', 10), bg='#f0f0f0', fg='#2c3e50')
        self.status_label.pack(side='left')
    
    def _create_restoration_controls(self, p):
        tk.Label(p, text="Pengaturan", font=('Segoe UI', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=(15, 10))
        bf = tk.Frame(p, bg='#ecf0f1'); bf.pack(pady=8, padx=15, fill='x')
        tk.Label(bf, text="Ukuran Kuas", font=('Segoe UI', 10), bg='#ecf0f1', fg='#34495e').pack(anchor='w')
        self.brush_size_var = tk.IntVar(value=15)
        ttk.Scale(bf, from_=5, to=50, orient='horizontal', variable=self.brush_size_var, command=self._on_brush_size_change).pack(fill='x', pady=5)
        self.brush_size_label = tk.Label(bf, text="15 px", font=('Segoe UI', 9), bg='#ecf0f1', fg='#7f8c8d'); self.brush_size_label.pack(anchor='e')
        self.auto_snap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bf, text="Deteksi Otomatis", variable=self.auto_snap_var, font=('Segoe UI', 9), bg='#ecf0f1', command=self._on_auto_snap_toggle).pack(anchor='w', pady=(8, 0))
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        hf = tk.Frame(p, bg='#ecf0f1'); hf.pack(pady=8, padx=15, fill='x')
        self.undo_btn = tk.Button(hf, text="Urungkan", font=('Segoe UI', 9), bg='#95a5a6', fg='white', relief='flat', command=self._undo, state='disabled'); self.undo_btn.pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.redo_btn = tk.Button(hf, text="Ulangi", font=('Segoe UI', 9), bg='#95a5a6', fg='white', relief='flat', command=self._redo, state='disabled'); self.redo_btn.pack(side='right', expand=True, fill='x', padx=(4, 0))
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        tk.Label(p, text="Zoom", font=('Segoe UI', 10), bg='#ecf0f1', fg='#34495e').pack(pady=(0, 5), padx=15, anchor='w')
        zf = tk.Frame(p, bg='#ecf0f1'); zf.pack(pady=5, padx=15, fill='x')
        tk.Button(zf, text="−", font=('Segoe UI', 12, 'bold'), bg='#95a5a6', fg='white', width=2, relief='flat', command=self._zoom_out).pack(side='left', padx=(0, 4))
        self.zoom_label = tk.Label(zf, text="100%", font=('Segoe UI', 10), bg='#ecf0f1', fg='#34495e', width=8); self.zoom_label.pack(side='left', padx=4)
        tk.Button(zf, text="+", font=('Segoe UI', 12, 'bold'), bg='#95a5a6', fg='white', width=2, relief='flat', command=self._zoom_in).pack(side='left', padx=(4, 0))
        tk.Button(zf, text="Reset", font=('Segoe UI', 8), bg='#bdc3c7', fg='white', relief='flat', command=self._zoom_reset).pack(side='right')
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        self.run_btn = tk.Button(p, text="Proses", font=('Segoe UI', 11, 'bold'), bg='#3498db', fg='white', pady=10, relief='flat', command=self._run_inpainting); self.run_btn.pack(pady=8, padx=15, fill='x')
        tk.Button(p, text="Hapus Mask", font=('Segoe UI', 10), bg='#5dade2', fg='white', pady=8, relief='flat', command=self._clear_mask).pack(pady=4, padx=15, fill='x')
        self.toggle_btn = tk.Button(p, text="Lihat Hasil", font=('Segoe UI', 10), bg='#85c1e9', fg='white', pady=8, relief='flat', command=self._toggle_view, state='disabled'); self.toggle_btn.pack(pady=4, padx=15, fill='x')
        self.save_btn = tk.Button(p, text="Simpan Hasil", font=('Segoe UI', 10), bg='#85c1e9', fg='white', pady=8, relief='flat', command=self._save_image, state='disabled'); self.save_btn.pack(pady=4, padx=15, fill='x')
        tk.Button(p, text="Gambar Baru", font=('Segoe UI', 10), bg='#95a5a6', fg='white', pady=8, relief='flat', command=self._new_image).pack(pady=4, padx=15, fill='x')
    
    def _create_evaluation_tab(self):
        main = tk.Frame(self.evaluation_frame, bg='#f0f0f0')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        left = tk.Frame(main, bg='#ecf0f1', width=320); left.pack(side='left', fill='y', padx=(0, 10)); left.pack_propagate(False)
        self._create_evaluation_controls(left)
        right = tk.Frame(main, bg='#ffffff', relief='solid', bd=1); right.pack(side='right', fill='both', expand=True)
        self._create_evaluation_display(right)
    
    def _create_evaluation_controls(self, p):
        tk.Label(p, text="Evaluasi Kuantitatif", font=('Segoe UI', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=(15, 5))
        tk.Label(p, text="Perbandingan PSNR, SSIM, dan Waktu", font=('Segoe UI', 9), bg='#ecf0f1', fg='#7f8c8d').pack(pady=(0, 15))
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=5, padx=15)
        s1 = tk.Frame(p, bg='#ecf0f1'); s1.pack(pady=10, padx=15, fill='x')
        tk.Label(s1, text="1. Ground Truth (Gambar Bersih)", font=('Segoe UI', 10, 'bold'), bg='#ecf0f1', fg='#34495e').pack(anchor='w')
        self.gt_btn = tk.Button(s1, text="Muat Ground Truth", font=('Segoe UI', 10), bg='#27ae60', fg='white', pady=8, relief='flat', command=self._load_ground_truth); self.gt_btn.pack(fill='x', pady=(5, 0))
        self.gt_status = tk.Label(s1, text="Belum dimuat", font=('Segoe UI', 9), bg='#ecf0f1', fg='#e74c3c'); self.gt_status.pack(anchor='w', pady=(3, 0))
        s2 = tk.Frame(p, bg='#ecf0f1'); s2.pack(pady=10, padx=15, fill='x')
        tk.Label(s2, text="2. Mask (Area yang Akan Dievaluasi)", font=('Segoe UI', 10, 'bold'), bg='#ecf0f1', fg='#34495e').pack(anchor='w')
        self.mask_btn = tk.Button(s2, text="Muat Mask", font=('Segoe UI', 10), bg='#27ae60', fg='white', pady=8, relief='flat', command=self._load_eval_mask); self.mask_btn.pack(fill='x', pady=(5, 0))
        self.mask_status = tk.Label(s2, text="Belum dimuat", font=('Segoe UI', 9), bg='#ecf0f1', fg='#e74c3c'); self.mask_status.pack(anchor='w', pady=(3, 0))
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=15, padx=15)
        self.eval_run_btn = tk.Button(p, text="Jalankan Analisis Komparatif", font=('Segoe UI', 11, 'bold'), bg='#3498db', fg='white', pady=12, relief='flat', command=self._run_evaluation, state='disabled'); self.eval_run_btn.pack(pady=10, padx=15, fill='x')
        self.eval_progress = ttk.Progressbar(p, mode='determinate', length=250); self.eval_progress.pack(pady=5, padx=15, fill='x')
        self.eval_status = tk.Label(p, text="Siap untuk evaluasi", font=('Segoe UI', 10), bg='#ecf0f1', fg='#7f8c8d'); self.eval_status.pack(pady=5, padx=15)
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=15, padx=15)
        self.export_btn = tk.Button(p, text="Ekspor Hasil (PNG + CSV)", font=('Segoe UI', 10), bg='#9b59b6', fg='white', pady=8, relief='flat', command=self._export_results, state='disabled'); self.export_btn.pack(pady=5, padx=15, fill='x')
    
    def _create_evaluation_display(self, p):
        tf = tk.Frame(p, bg='#ffffff'); tf.pack(fill='x', padx=10, pady=10)
        tk.Label(tf, text="Hasil Perbandingan Metode", font=('Segoe UI', 12, 'bold'), bg='#ffffff', fg='#2c3e50').pack(anchor='w', pady=(0, 10))
        cols = ('method', 'time', 'psnr', 'ssim')
        self.results_tree = ttk.Treeview(tf, columns=cols, show='headings', height=4)
        for c, t, w in [('method', 'Metode', 200), ('time', 'Waktu (s)', 120), ('psnr', 'PSNR (dB)', 120), ('ssim', 'SSIM', 120)]:
            self.results_tree.heading(c, text=t); self.results_tree.column(c, width=w, anchor='w' if c == 'method' else 'center')
        self.results_tree.pack(fill='x')
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=10)
        vf = tk.Frame(p, bg='#ffffff'); vf.pack(fill='both', expand=True, padx=10, pady=10)
        tk.Label(vf, text="Perbandingan Visual", font=('Segoe UI', 12, 'bold'), bg='#ffffff', fg='#2c3e50').pack(anchor='w', pady=(0, 10))
        self.thumbnails_frame = tk.Frame(vf, bg='#f5f5f5'); self.thumbnails_frame.pack(fill='both', expand=True)
        self.thumb_labels = {}
        for i, name in enumerate(['Ground Truth', 'Damaged', 'Telea', 'Criminisi', 'Proposed']):
            f = tk.Frame(self.thumbnails_frame, bg='#e0e0e0', relief='solid', bd=1); f.grid(row=0, column=i, padx=5, pady=5, sticky='nsew')
            il = tk.Label(f, bg='#e0e0e0', width=15, height=8); il.pack(padx=5, pady=5)
            tk.Label(f, text=name, font=('Segoe UI', 9), bg='#e0e0e0').pack(pady=(0, 5))
            self.thumb_labels[name] = il
            self.thumbnails_frame.columnconfigure(i, weight=1)
    
    # EVALUATION METHODS
    def _load_ground_truth(self):
        fp = filedialog.askopenfilename(title="Pilih Ground Truth", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            self.eval_ground_truth = cv2.imread(fp)
            if self.eval_ground_truth is not None:
                h, w = self.eval_ground_truth.shape[:2]
                self.gt_status.config(text=f"✓ {w}×{h}", fg='#27ae60')
                self._update_eval_thumbnail('Ground Truth', self.eval_ground_truth)
                self._check_eval_ready()
    
    def _load_eval_mask(self):
        if self.eval_ground_truth is None: messagebox.showwarning("Peringatan", "Muat Ground Truth dulu!"); return
        fp = filedialog.askopenfilename(title="Pilih Mask", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            m = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                gh, gw = self.eval_ground_truth.shape[:2]
                if m.shape[:2] != (gh, gw): m = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_NEAREST)
                self.eval_mask = (m > 127).astype(np.uint8) * 255
                self.mask_status.config(text=f"✓ {np.count_nonzero(self.eval_mask)} px", fg='#27ae60')
                self.eval_damaged = self.eval_ground_truth.copy()
                self.eval_damaged[self.eval_mask > 0] = [255, 255, 255]
                self._update_eval_thumbnail('Damaged', self.eval_damaged)
                self._check_eval_ready()
    
    def _check_eval_ready(self):
        if self.eval_ground_truth is not None and self.eval_mask is not None:
            self.eval_run_btn.config(state='normal', bg='#3498db')
        else: self.eval_run_btn.config(state='disabled', bg='#bdc3c7')
    
    def _update_eval_thumbnail(self, name, img):
        if name not in self.thumb_labels: return
        h, w = img.shape[:2]; s = 150 / max(h, w); nw, nh = int(w * s), int(h * s)
        t = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(t))
        self.thumb_labels[name].config(image=tk_img, width=nw, height=nh)
        self.thumb_labels[name].image = tk_img
    
    def _run_evaluation(self):
        if not SKIMAGE_AVAILABLE: messagebox.showerror("Error", "Install scikit-image!"); return
        if self.eval_is_running: return
        self.eval_is_running = True
        self.eval_run_btn.config(state='disabled', bg='#bdc3c7')
        self.eval_status.config(text="Menjalankan...", fg='#e67e22')
        self.eval_progress['value'] = 0
        for item in self.results_tree.get_children(): self.results_tree.delete(item)
        self.eval_results = {}
        threading.Thread(target=self._evaluation_worker, daemon=True).start()
        self._check_eval_queue()
    
    def _evaluation_worker(self):
        try:
            inp = HybridInpainter(target_proc_size=450, patch_size=9)
            methods = [("Telea (OpenCV)", "telea"), ("Criminisi (Standard)", "criminisi_standard"), ("Proposed (Hybrid)", "adaptive")]
            for i, (name, key) in enumerate(methods):
                self.eval_queue.put(('status', f"Menjalankan: {name}..."))
                self.eval_queue.put(('progress', (i / 3) * 100))
                t0 = time.time()
                res = inp.inpaint(self.eval_damaged.copy(), self.eval_mask.copy(), method=key)
                elapsed = time.time() - t0
                psnr_v = psnr(self.eval_ground_truth, res)
                ssim_v = ssim(self.eval_ground_truth, res, channel_axis=2, data_range=255)
                self.eval_results[name] = {'result': res, 'time': elapsed, 'psnr': psnr_v, 'ssim': ssim_v}
                self.eval_queue.put(('result', (name, elapsed, psnr_v, ssim_v, res)))
            self.eval_queue.put(('progress', 100))
            self.eval_queue.put(('complete', None))
        except Exception as e: self.eval_queue.put(('error', str(e)))
    
    def _check_eval_queue(self):
        try:
            while not self.eval_queue.empty():
                t, d = self.eval_queue.get_nowait()
                if t == 'status': self.eval_status.config(text=d, fg='#e67e22')
                elif t == 'progress': self.eval_progress['value'] = d
                elif t == 'result':
                    n, el, pv, sv, res = d
                    self.results_tree.insert('', 'end', values=(n, f"{el:.2f}", f"{pv:.2f}", f"{sv:.4f}"))
                    tn = 'Telea' if 'Telea' in n else ('Criminisi' if 'Criminisi' in n else 'Proposed')
                    self._update_eval_thumbnail(tn, res)
                elif t == 'complete':
                    self.eval_is_running = False
                    self.eval_run_btn.config(state='normal', bg='#3498db')
                    self.eval_status.config(text="✓ Evaluasi selesai!", fg='#27ae60')
                    self.export_btn.config(state='normal')
                    self._print_markdown_table()
                elif t == 'error':
                    self.eval_is_running = False
                    self.eval_run_btn.config(state='normal', bg='#3498db')
                    self.eval_status.config(text=f"Error: {d}", fg='#e74c3c')
        except: pass
        if self.eval_is_running: self.root.after(100, self._check_eval_queue)
    
    def _print_markdown_table(self):
        print("\n" + "="*70 + "\nHASIL EVALUASI (Markdown)\n" + "="*70)
        print("| Metode | Waktu (s) | PSNR (dB) | SSIM |")
        print("|--------|-----------|-----------|------|")
        for n, d in self.eval_results.items(): print(f"| {n} | {d['time']:.2f} | {d['psnr']:.2f} | {d['ssim']:.4f} |")
        print("="*70)
    
    def _export_results(self):
        if not self.eval_results: return
        od = filedialog.askdirectory(title="Pilih Folder Output")
        if not od: return
        import os
        cv2.imwrite(os.path.join(od, "ground_truth.png"), self.eval_ground_truth)
        cv2.imwrite(os.path.join(od, "mask.png"), self.eval_mask)
        cv2.imwrite(os.path.join(od, "damaged.png"), self.eval_damaged)
        for n, d in self.eval_results.items():
            sn = n.replace(" ", "_").replace("(", "").replace(")", "")
            cv2.imwrite(os.path.join(od, f"result_{sn}.png"), d['result'])
        with open(os.path.join(od, "results.csv"), 'w') as f:
            f.write("Method,Time,PSNR,SSIM\n")
            for n, d in self.eval_results.items(): f.write(f"{n},{d['time']:.4f},{d['psnr']:.4f},{d['ssim']:.6f}\n")
        messagebox.showinfo("Berhasil", f"Hasil diekspor ke:\n{od}")
    
    # RESTORATION METHODS
    def _upload_image(self):
        fp = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            self.original_image = cv2.imread(fp)
            if self.original_image is None: return
            h, w = self.original_image.shape[:2]
            self.mask = np.zeros((h, w), dtype=np.uint8)
            self.result_image = None
            self.mask_history = [self.mask.copy()]; self.history_index = 0
            self.show_result = False; self.zoom_level = 1.0; self.pan_offset_x = self.pan_offset_y = 0
            self.scale_factor = min(self.max_display_width / w, self.max_display_height / h, 1.0)
            dh, dw = int(h * self.scale_factor), int(w * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (dw, dh), interpolation=cv2.INTER_AREA)
            self.display_mask = np.zeros((dh, dw), dtype=np.uint8)
            self.upload_frame.pack_forget(); self.canvas_frame.pack(fill='both', expand=True)
            self._update_canvas_display()
            self.status_label.config(text=f"Gambar dimuat: {w}×{h}")
    
    def _on_brush_size_change(self, v): self.brush_size = int(float(v)); self.brush_size_label.config(text=f"{self.brush_size} px")
    def _on_auto_snap_toggle(self): self.auto_snap_enabled = self.auto_snap_var.get()
    
    def _on_mouse_down(self, e):
        if e.state & 0x0001: self.is_panning = True; self.pan_start_x, self.pan_start_y = e.x, e.y; self.canvas.config(cursor='fleur')
        else:
            self.is_drawing = True; self.last_x, self.last_y = e.x, e.y
            ox = int((e.x - self.pan_offset_x) / (self.zoom_level * self.scale_factor))
            oy = int((e.y - self.pan_offset_y) / (self.zoom_level * self.scale_factor))
            self.stroke_bbox = [ox, oy, ox, oy]
            self._save_mask_state(); self._draw_point(e.x, e.y)
    
    def _on_mouse_drag(self, e):
        if self.is_panning:
            self.pan_offset_x += e.x - self.pan_start_x; self.pan_offset_y += e.y - self.pan_start_y
            self.pan_start_x, self.pan_start_y = e.x, e.y; self._update_canvas_display()
        elif self.is_drawing:
            if self.last_x is not None: self._draw_line(self.last_x, self.last_y, e.x, e.y)
            self.last_x, self.last_y = e.x, e.y
    
    def _on_mouse_up(self, e):
        if self.is_panning: self.is_panning = False; self.canvas.config(cursor='crosshair')
        else:
            if self.is_drawing and self.auto_snap_enabled and self.stroke_bbox: self._refine_mask_auto_snap()
            self.is_drawing = False; self.last_x = self.last_y = None; self.stroke_bbox = None
    
    def _draw_point(self, x, y):
        if self.display_mask is None: return
        ix, iy = int((x - self.pan_offset_x) / self.zoom_level), int((y - self.pan_offset_y) / self.zoom_level)
        h, w = self.display_mask.shape
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(self.display_mask, (ix, iy), self.brush_size // 2, 255, -1)
            ox, oy = int(ix / self.scale_factor), int(iy / self.scale_factor)
            obs = max(1, int(self.brush_size / self.scale_factor))
            if self.stroke_bbox:
                r = obs // 2
                self.stroke_bbox[0] = min(self.stroke_bbox[0], ox - r); self.stroke_bbox[1] = min(self.stroke_bbox[1], oy - r)
                self.stroke_bbox[2] = max(self.stroke_bbox[2], ox + r); self.stroke_bbox[3] = max(self.stroke_bbox[3], oy + r)
            ho, wo = self.mask.shape
            if 0 <= ox < wo and 0 <= oy < ho: cv2.circle(self.mask, (ox, oy), obs // 2, 255, -1)
            self._update_canvas_display()
    
    def _draw_line(self, x1, y1, x2, y2):
        if self.display_mask is None: return
        ix1, iy1 = int((x1 - self.pan_offset_x) / self.zoom_level), int((y1 - self.pan_offset_y) / self.zoom_level)
        ix2, iy2 = int((x2 - self.pan_offset_x) / self.zoom_level), int((y2 - self.pan_offset_y) / self.zoom_level)
        cv2.line(self.display_mask, (ix1, iy1), (ix2, iy2), 255, self.brush_size)
        ox1, oy1 = int(ix1 / self.scale_factor), int(iy1 / self.scale_factor)
        ox2, oy2 = int(ix2 / self.scale_factor), int(iy2 / self.scale_factor)
        obs = max(1, int(self.brush_size / self.scale_factor))
        if self.stroke_bbox:
            r = obs // 2
            self.stroke_bbox[0] = min(self.stroke_bbox[0], min(ox1, ox2) - r); self.stroke_bbox[1] = min(self.stroke_bbox[1], min(oy1, oy2) - r)
            self.stroke_bbox[2] = max(self.stroke_bbox[2], max(ox1, ox2) + r); self.stroke_bbox[3] = max(self.stroke_bbox[3], max(oy1, oy2) + r)
        cv2.line(self.mask, (ox1, oy1), (ox2, oy2), 255, obs)
        self._update_canvas_display()
    
    def _refine_mask_auto_snap(self):
        if not self.stroke_bbox or self.original_image is None: return
        try:
            x1, y1, x2, y2 = self.stroke_bbox; ho, wo = self.mask.shape; pad = 10
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(wo, x2 + pad), min(ho, y2 + pad)
            if x2 - x1 < 5 or y2 - y1 < 5: return
            roi = self.original_image[y1:y2, x1:x2].copy(); rm = self.mask[y1:y2, x1:x2].copy()
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            refined = cv2.bitwise_and(binary, rm)
            k = np.ones((5, 5), np.uint8); refined = cv2.dilate(refined, k, iterations=3)
            refined = cv2.bitwise_and(refined, rm)
            op, rp = np.sum(rm > 0), np.sum(refined > 0)
            if rp > 0 and rp < op * 2:
                self.mask[y1:y2, x1:x2] = refined
                dx1, dy1 = int(x1 * self.scale_factor), int(y1 * self.scale_factor)
                dx2, dy2 = int(x2 * self.scale_factor), int(y2 * self.scale_factor)
                rd = cv2.resize(refined, (dx2 - dx1, dy2 - dy1), interpolation=cv2.INTER_NEAREST)
                self.display_mask[dy1:dy2, dx1:dx2] = rd
                self._update_canvas_display()
                print(f"[AUTO-SNAP] ✓ {op} → {rp} px")
        except Exception as e: print(f"[AUTO-SNAP] Error: {e}")
    
    def _update_canvas_display(self):
        if self.display_image is None: return
        if self.is_processing and self.result_image is not None:
            d = cv2.resize(self.result_image, (self.display_image.shape[1], self.display_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        elif self.show_result and self.result_image is not None:
            d = cv2.resize(self.result_image, (self.display_image.shape[1], self.display_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            d = self.display_image.copy()
            if not self.show_result and not self.is_processing:
                mo = self.display_mask > 0
                if np.any(mo): d[mo] = d[mo] * 0.5 + np.array([0, 0, 255]) * 0.5
        if self.zoom_level != 1.0:
            h, w = d.shape[:2]; d = cv2.resize(d, (int(w * self.zoom_level), int(h * self.zoom_level)), interpolation=cv2.INTER_LINEAR)
        dr = cv2.cvtColor(d.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pi = Image.fromarray(dr)
        if self.pan_offset_x != 0 or self.pan_offset_y != 0 or self.zoom_level != 1.0:
            cw, ch = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
            iw, ih = pi.size
            l, t, r, b = max(0, -self.pan_offset_x), max(0, -self.pan_offset_y), min(iw, cw - self.pan_offset_x), min(ih, ch - self.pan_offset_y)
            if r > l and b > t: pi = pi.crop((l, t, r, b))
        self.tk_image = ImageTk.PhotoImage(pi)
        xp, yp = max(0, self.pan_offset_x), max(0, self.pan_offset_y)
        if self.canvas_image_id is None: self.canvas_image_id = self.canvas.create_image(xp, yp, anchor='nw', image=self.tk_image)
        else: self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image); self.canvas.coords(self.canvas_image_id, xp, yp)
    
    def _clear_mask(self):
        if self.mask is not None:
            self._save_mask_state(); self.mask.fill(0); self.display_mask.fill(0)
            self.result_image = None; self.show_result = False
            self.save_btn.config(state='disabled'); self.toggle_btn.config(state='disabled', text="Lihat Hasil")
            self._update_canvas_display(); self.status_label.config(text="Mask dihapus.")
    
    def _run_inpainting(self):
        if self.is_processing or self.mask is None or np.count_nonzero(self.mask) == 0:
            if self.mask is None or np.count_nonzero(self.mask) == 0: messagebox.showwarning("Peringatan", "Gambar mask dulu!")
            return
        self.is_processing = True; self.run_btn.config(state='disabled', bg='#bdc3c7'); self.progress_bar.start(10)
        self.status_label.config(text="Memproses...")
        threading.Thread(target=self._inpainting_worker, daemon=True).start()
    
    def _inpainting_worker(self):
        try:
            res = self.inpainter.inpaint(self.original_image.copy(), self.mask.copy(), padding=20, progress_callback=self._progress_callback)
            self.progress_queue.put(('complete', res))
        except Exception as e: self.progress_queue.put(('error', str(e)))
    
    def _progress_callback(self, img, it, rem, pct): self.progress_queue.put(('preview', {'image': img.copy(), 'iteration': it, 'remaining': rem, 'percent': pct}))
    
    def _check_progress_queue(self):
        try:
            while not self.progress_queue.empty():
                t, d = self.progress_queue.get_nowait()
                if t == 'preview':
                    self.result_image = d['image']
                    self.status_label.config(text=f"Proses... It:{d['iteration']} | Sisa:{d['remaining']}px | {d['percent']:.1f}%")
                    self._update_canvas_display()
                elif t == 'complete':
                    self.result_image = d; self._update_canvas_display()
                    self.is_processing = False; self.progress_bar.stop()
                    self.run_btn.config(state='normal', bg='#3498db')
                    self.save_btn.config(state='normal'); self.toggle_btn.config(state='normal')
                    self.status_label.config(text="Selesai!")
                elif t == 'error':
                    self.is_processing = False; self.progress_bar.stop()
                    self.run_btn.config(state='normal', bg='#3498db')
                    self.status_label.config(text="Error!"); messagebox.showerror("Error", d)
        except: pass
        self.root.after(100, self._check_progress_queue)
    
    def _save_image(self):
        if self.result_image is None: return
        fp = filedialog.asksaveasfilename(title="Simpan", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if fp: cv2.imwrite(fp, self.result_image); messagebox.showinfo("Berhasil", f"Disimpan: {fp}")
    
    def _save_mask_state(self):
        if self.mask is None: return
        self.mask_history = self.mask_history[:self.history_index + 1]
        self.mask_history.append(self.mask.copy()); self.history_index += 1
        if len(self.mask_history) > self.max_history: self.mask_history.pop(0); self.history_index -= 1
        self._update_history_buttons()
    
    def _undo(self):
        if self.history_index > 0:
            self.history_index -= 1; self.mask = self.mask_history[self.history_index].copy()
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self._update_canvas_display(); self._update_history_buttons()
    
    def _redo(self):
        if self.history_index < len(self.mask_history) - 1:
            self.history_index += 1; self.mask = self.mask_history[self.history_index].copy()
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self._update_canvas_display(); self._update_history_buttons()
    
    def _update_history_buttons(self):
        self.undo_btn.config(state='normal' if self.history_index > 0 else 'disabled', bg='#95a5a6' if self.history_index > 0 else '#bdc3c7')
        self.redo_btn.config(state='normal' if self.history_index < len(self.mask_history) - 1 else 'disabled', bg='#95a5a6' if self.history_index < len(self.mask_history) - 1 else '#bdc3c7')
    
    def _zoom_in(self):
        if self.zoom_level < self.zoom_max: self.zoom_level = min(self.zoom_level + 0.25, self.zoom_max); self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%"); self._update_canvas_display()
    def _zoom_out(self):
        if self.zoom_level > self.zoom_min: self.zoom_level = max(self.zoom_level - 0.25, self.zoom_min); self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%"); self._update_canvas_display()
    def _zoom_reset(self): self.zoom_level = 1.0; self.pan_offset_x = self.pan_offset_y = 0; self.zoom_label.config(text="100%"); self._update_canvas_display()
    def _on_mouse_wheel(self, e): self._zoom_in() if e.delta > 0 else self._zoom_out()
    
    def _toggle_view(self):
        if self.result_image is None: return
        self.show_result = not self.show_result
        self.toggle_btn.config(text="Lihat Asli" if self.show_result else "Lihat Hasil")
        self.status_label.config(text="Melihat hasil" if self.show_result else "Melihat asli")
        self._update_canvas_display()
    
    def _new_image(self):
        if self.is_processing: return
        self.original_image = self.display_image = self.mask = self.display_mask = self.result_image = None
        self.mask_history = []; self.history_index = -1
        self.zoom_level = 1.0; self.pan_offset_x = self.pan_offset_y = 0; self.show_result = False
        self.canvas_frame.pack_forget(); self.upload_frame.pack(fill='both', expand=True)
        self.canvas_image_id = None
        self.status_label.config(text="Siap. Muat gambar untuk mulai.")
    
    def run(self): self.root.mainloop()


def main():
    root = tk.Tk()
    app = MangaCleanerApp(root)
    app.run()


if __name__ == "__main__":
    main()
