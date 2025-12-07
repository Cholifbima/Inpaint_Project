try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, Callable

from inpainter import HybridInpainter

try:
    from skimage.metrics import structural_similarity as calc_ssim
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[scikit-image not found. pip install scikit-image")


class CanvasController:
    def __init__(self, canvas: tk.Canvas, on_update: Callable):
        self.canvas = canvas
        self.on_update = on_update
        
        # Image data
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.display_mask: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        
        # Display scaling
        self.scale_factor = 1.0
        self.max_display_width = 700
        self.max_display_height = 550
        
        # Drawing state
        self.is_drawing = False
        self.brush_size = 15
        self.last_x, self.last_y = None, None
        self.auto_snap_enabled = True
        self.stroke_bbox = None
        
        # History
        self.mask_history = []
        self.history_index = -1
        self.max_history = 50
        
        # Zoom/Pan
        self.zoom_level = 1.0
        self.zoom_min, self.zoom_max = 0.5, 4.0
        self.pan_offset_x, self.pan_offset_y = 0, 0
        self.is_panning = False
        self.pan_start_x, self.pan_start_y = 0, 0
        
        # Canvas state
        self.canvas_image_id = None
        self.tk_image = None
        self.show_result = False
        
        # Bind events
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
    
    def load_image(self, image: np.ndarray):
        self.original_image = image.copy()
        h, w = image.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.result_image = None
        self.show_result = False
        
        # Calculate scale
        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h
        self.scale_factor = min(scale_w, scale_h, 1.0)
        
        # Create display versions
        dh, dw = int(h * self.scale_factor), int(w * self.scale_factor)
        self.display_image = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
        self.display_mask = np.zeros((dh, dw), dtype=np.uint8)
        
        # Reset state
        self.zoom_level = 1.0
        self.pan_offset_x, self.pan_offset_y = 0, 0
        self.mask_history = [self.mask.copy()]
        self.history_index = 0
        
        self.update_display()
        return w, h
    
    def _on_mouse_down(self, event):
        if self.display_image is None:
            return
        if event.state & 0x0001:  # Shift = pan
            self.is_panning = True
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.canvas.config(cursor='fleur')
        else:
            self.is_drawing = True
            self.last_x, self.last_y = event.x, event.y
            ox = int((event.x - self.pan_offset_x) / (self.zoom_level * self.scale_factor))
            oy = int((event.y - self.pan_offset_y) / (self.zoom_level * self.scale_factor))
            self.stroke_bbox = [ox, oy, ox, oy]
            self._save_state()
            self._draw_point(event.x, event.y)
    
    def _on_mouse_drag(self, event):
        if self.display_image is None:
            return
        if self.is_panning:
            self.pan_offset_x += event.x - self.pan_start_x
            self.pan_offset_y += event.y - self.pan_start_y
            self.pan_start_x, self.pan_start_y = event.x, event.y
            self.update_display()
        elif self.is_drawing and self.last_x is not None:
            self._draw_line(self.last_x, self.last_y, event.x, event.y)
            self.last_x, self.last_y = event.x, event.y
    
    def _on_mouse_up(self, event):
        if self.is_panning:
            self.is_panning = False
            self.canvas.config(cursor='crosshair')
        else:
            if self.is_drawing and self.auto_snap_enabled and self.stroke_bbox:
                self._refine_auto_snap()
            self.is_drawing = False
            self.last_x = self.last_y = None
            self.stroke_bbox = None
    
    def _on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def _draw_point(self, x, y):
        if self.display_mask is None:
            return
        ix = int((x - self.pan_offset_x) / self.zoom_level)
        iy = int((y - self.pan_offset_y) / self.zoom_level)
        h, w = self.display_mask.shape
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(self.display_mask, (ix, iy), self.brush_size // 2, 255, -1)
            ox, oy = int(ix / self.scale_factor), int(iy / self.scale_factor)
            obs = max(1, int(self.brush_size / self.scale_factor))
            if self.stroke_bbox:
                r = obs // 2
                self.stroke_bbox[0] = min(self.stroke_bbox[0], ox - r)
                self.stroke_bbox[1] = min(self.stroke_bbox[1], oy - r)
                self.stroke_bbox[2] = max(self.stroke_bbox[2], ox + r)
                self.stroke_bbox[3] = max(self.stroke_bbox[3], oy + r)
            ho, wo = self.mask.shape
            if 0 <= ox < wo and 0 <= oy < ho:
                cv2.circle(self.mask, (ox, oy), obs // 2, 255, -1)
            self.update_display()
    
    def _draw_line(self, x1, y1, x2, y2):
        if self.display_mask is None:
            return
        ix1 = int((x1 - self.pan_offset_x) / self.zoom_level)
        iy1 = int((y1 - self.pan_offset_y) / self.zoom_level)
        ix2 = int((x2 - self.pan_offset_x) / self.zoom_level)
        iy2 = int((y2 - self.pan_offset_y) / self.zoom_level)
        cv2.line(self.display_mask, (ix1, iy1), (ix2, iy2), 255, self.brush_size)
        
        ox1, oy1 = int(ix1 / self.scale_factor), int(iy1 / self.scale_factor)
        ox2, oy2 = int(ix2 / self.scale_factor), int(iy2 / self.scale_factor)
        obs = max(1, int(self.brush_size / self.scale_factor))
        if self.stroke_bbox:
            r = obs // 2
            self.stroke_bbox[0] = min(self.stroke_bbox[0], min(ox1, ox2) - r)
            self.stroke_bbox[1] = min(self.stroke_bbox[1], min(oy1, oy2) - r)
            self.stroke_bbox[2] = max(self.stroke_bbox[2], max(ox1, ox2) + r)
            self.stroke_bbox[3] = max(self.stroke_bbox[3], max(oy1, oy2) + r)
        cv2.line(self.mask, (ox1, oy1), (ox2, oy2), 255, obs)
        self.update_display()
    
    def _refine_auto_snap(self):
        if not self.stroke_bbox or self.original_image is None:
            return
        try:
            x1, y1, x2, y2 = self.stroke_bbox
            ho, wo = self.mask.shape
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(wo, x2 + pad), min(ho, y2 + pad)
            if x2 - x1 < 5 or y2 - y1 < 5:
                return
            
            roi = self.original_image[y1:y2, x1:x2].copy()
            rm = self.mask[y1:y2, x1:x2].copy()
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            refined = cv2.bitwise_and(binary, rm)
            k = np.ones((5, 5), np.uint8)
            refined = cv2.dilate(refined, k, iterations=3)
            refined = cv2.bitwise_and(refined, rm)
            
            op, rp = np.sum(rm > 0), np.sum(refined > 0)
            if rp > 0 and rp < op * 2:
                self.mask[y1:y2, x1:x2] = refined
                dx1, dy1 = int(x1 * self.scale_factor), int(y1 * self.scale_factor)
                dx2, dy2 = int(x2 * self.scale_factor), int(y2 * self.scale_factor)
                rd = cv2.resize(refined, (dx2 - dx1, dy2 - dy1), interpolation=cv2.INTER_NEAREST)
                self.display_mask[dy1:dy2, dx1:dx2] = rd
                self.update_display()
                print(f"[AUTO-SNAP] ✓ {op} → {rp} px")
        except Exception as e:
            print(f"[AUTO-SNAP] Error: {e}")
    
    def update_display(self):
        if self.display_image is None:
            return
        
        # Choose what to display
        if self.show_result and self.result_image is not None:
            d = cv2.resize(self.result_image, 
                          (self.display_image.shape[1], self.display_image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
        else:
            d = self.display_image.copy()
            # Overlay mask in red
            mo = self.display_mask > 0
            if np.any(mo):
                d[mo] = d[mo] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        # Apply zoom
        if self.zoom_level != 1.0:
            h, w = d.shape[:2]
            d = cv2.resize(d, (int(w * self.zoom_level), int(h * self.zoom_level)),
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert to PIL
        dr = cv2.cvtColor(d.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pi = Image.fromarray(dr)
        
        # Apply pan/crop
        if self.pan_offset_x != 0 or self.pan_offset_y != 0 or self.zoom_level != 1.0:
            cw = self.canvas.winfo_width() or 700
            ch = self.canvas.winfo_height() or 550
            iw, ih = pi.size
            l = max(0, -self.pan_offset_x)
            t = max(0, -self.pan_offset_y)
            r = min(iw, cw - self.pan_offset_x)
            b = min(ih, ch - self.pan_offset_y)
            if r > l and b > t:
                pi = pi.crop((l, t, r, b))
        
        self.tk_image = ImageTk.PhotoImage(pi)
        xp, yp = max(0, self.pan_offset_x), max(0, self.pan_offset_y)
        
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(xp, yp, anchor='nw', image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
            self.canvas.coords(self.canvas_image_id, xp, yp)
        
        self.on_update()
    
    def _save_state(self):
        if self.mask is None:
            return
        self.mask_history = self.mask_history[:self.history_index + 1]
        self.mask_history.append(self.mask.copy())
        self.history_index += 1
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.mask = self.mask_history[self.history_index].copy()
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self.update_display()
            return True
        return False
    
    def redo(self):
        if self.history_index < len(self.mask_history) - 1:
            self.history_index += 1
            self.mask = self.mask_history[self.history_index].copy()
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            self.update_display()
            return True
        return False
    
    def clear_mask(self):
        if self.mask is not None:
            self._save_state()
            self.mask.fill(0)
            self.display_mask.fill(0)
            self.result_image = None
            self.show_result = False
            self.update_display()
    
    def zoom_in(self):
        if self.zoom_level < self.zoom_max:
            self.zoom_level = min(self.zoom_level + 0.25, self.zoom_max)
            self.update_display()
    
    def zoom_out(self):
        if self.zoom_level > self.zoom_min:
            self.zoom_level = max(self.zoom_level - 0.25, self.zoom_min)
            self.update_display()
    
    def zoom_reset(self):
        self.zoom_level = 1.0
        self.pan_offset_x = self.pan_offset_y = 0
        self.update_display()
    
    def can_undo(self):
        return self.history_index > 0
    
    def can_redo(self):
        return self.history_index < len(self.mask_history) - 1


class MangaCleanerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manga Inpainting - Adaptive Hybrid Pyramid")
        self.root.geometry("1300x900")
        self.root.configure(bg='#f0f0f0')
        
        # Inpainting
        self.inpainter = HybridInpainter(target_proc_size=450, patch_size=9)
        self.is_processing = False
        self.progress_queue = queue.Queue()
        
        # Tab 1 state
        self.tab1_controller: Optional[CanvasController] = None
        self.tab1_image_loaded = False
        
        # Tab 2 state
        self.tab2_controller: Optional[CanvasController] = None
        self.eval_ground_truth: Optional[np.ndarray] = None
        self.eval_input_image: Optional[np.ndarray] = None
        self.eval_results: Dict[str, Dict] = {}
        self.eval_is_running = False
        self.eval_queue = queue.Queue()
        
        self._setup_styles()
        self._create_ui()
        self._check_queues()
    
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', '#3498db'), ('!selected', '#bdc3c7')],
                  foreground=[('selected', 'white'), ('!selected', '#2c3e50')])
        style.configure('Treeview', font=('Segoe UI', 10), rowheight=28)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))
    
    def _create_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1
        self.tab1_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1_frame, text='  Mode Restorasi  ')
        self._create_tab1()
        
        # Tab 2
        self.tab2_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2_frame, text='  Mode Evaluasi  ')
        self._create_tab2()
    

    # TAB 1: RESTORATION MODE
    def _create_tab1(self):
        main = tk.Frame(self.tab1_frame, bg='#f0f0f0')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Canvas area
        left = tk.Frame(main, bg='#ffffff', relief='solid', bd=1)
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Upload prompt
        self.tab1_upload_frame = tk.Frame(left, bg='#ffffff')
        self.tab1_upload_frame.pack(fill='both', expand=True)
        tk.Label(self.tab1_upload_frame, text="Klik untuk memuat gambar manga",
                font=('Segoe UI', 14), bg='#ffffff', fg='#7f8c8d').pack(expand=True)
        tk.Button(self.tab1_upload_frame, text="Buka Gambar", font=('Segoe UI', 12, 'bold'),
                 bg='#3498db', fg='white', padx=30, pady=10, relief='flat',
                 command=self._tab1_upload_image).pack(pady=20)
        
        # Canvas frame
        self.tab1_canvas_frame = tk.Frame(left, bg='#ffffff')
        self.tab1_canvas_label = tk.Label(self.tab1_canvas_frame, 
            text="Gambar mask di area yang ingin dihapus",
            font=('Segoe UI', 10), bg='#ffffff', fg='#7f8c8d')
        self.tab1_canvas_label.pack(pady=5)
        
        self.tab1_canvas = tk.Canvas(self.tab1_canvas_frame, bg='#e0e0e0',
                                     highlightthickness=0, cursor='crosshair')
        self.tab1_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create controller
        self.tab1_controller = CanvasController(self.tab1_canvas, self._tab1_on_update)
        
        # Right: Controls
        right = tk.Frame(main, bg='#ecf0f1', width=280)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        self._create_tab1_controls(right)
        
        # Bottom: Progress
        bottom = tk.Frame(self.tab1_frame, bg='#f0f0f0')
        bottom.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        self.tab1_progress = ttk.Progressbar(bottom, mode='indeterminate', length=300)
        self.tab1_progress.pack(side='left', padx=(0, 10))
        self.tab1_status = tk.Label(bottom, text="Siap. Muat gambar untuk mulai.",
                                   font=('Segoe UI', 10), bg='#f0f0f0', fg='#2c3e50')
        self.tab1_status.pack(side='left')
        
        # Keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self._tab1_undo())
        self.root.bind('<Control-y>', lambda e: self._tab1_redo())
    
    def _create_tab1_controls(self, p):
        tk.Label(p, text="Pengaturan", font=('Segoe UI', 14, 'bold'),
                bg='#ecf0f1', fg='#2c3e50').pack(pady=(15, 10))
        
        # Brush size
        bf = tk.Frame(p, bg='#ecf0f1')
        bf.pack(pady=8, padx=15, fill='x')
        tk.Label(bf, text="Ukuran Kuas", font=('Segoe UI', 10),
                bg='#ecf0f1', fg='#34495e').pack(anchor='w')
        self.tab1_brush_var = tk.IntVar(value=15)
        ttk.Scale(bf, from_=5, to=50, orient='horizontal', variable=self.tab1_brush_var,
                 command=self._tab1_brush_change).pack(fill='x', pady=5)
        self.tab1_brush_label = tk.Label(bf, text="15 px", font=('Segoe UI', 9),
                                        bg='#ecf0f1', fg='#7f8c8d')
        self.tab1_brush_label.pack(anchor='e')
        
        # Auto-snap
        self.tab1_autosnap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bf, text="Deteksi Otomatis", variable=self.tab1_autosnap_var,
                      font=('Segoe UI', 9), bg='#ecf0f1',
                      command=self._tab1_autosnap_toggle).pack(anchor='w', pady=(8, 0))
        
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        
        # Undo/Redo
        hf = tk.Frame(p, bg='#ecf0f1')
        hf.pack(pady=8, padx=15, fill='x')
        self.tab1_undo_btn = tk.Button(hf, text="Urungkan", font=('Segoe UI', 9),
                                       bg='#bdc3c7', fg='white', relief='flat',
                                       command=self._tab1_undo, state='disabled')
        self.tab1_undo_btn.pack(side='left', expand=True, fill='x', padx=(0, 4))
        self.tab1_redo_btn = tk.Button(hf, text="Ulangi", font=('Segoe UI', 9),
                                       bg='#bdc3c7', fg='white', relief='flat',
                                       command=self._tab1_redo, state='disabled')
        self.tab1_redo_btn.pack(side='right', expand=True, fill='x', padx=(4, 0))
        
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        
        # Zoom
        tk.Label(p, text="Zoom", font=('Segoe UI', 10), bg='#ecf0f1', fg='#34495e').pack(padx=15, anchor='w')
        zf = tk.Frame(p, bg='#ecf0f1')
        zf.pack(pady=5, padx=15, fill='x')
        tk.Button(zf, text="−", font=('Segoe UI', 12, 'bold'), bg='#95a5a6', fg='white',
                 width=2, relief='flat', command=lambda: self.tab1_controller.zoom_out()).pack(side='left')
        self.tab1_zoom_label = tk.Label(zf, text="100%", font=('Segoe UI', 10),
                                        bg='#ecf0f1', fg='#34495e', width=8)
        self.tab1_zoom_label.pack(side='left', padx=4)
        tk.Button(zf, text="+", font=('Segoe UI', 12, 'bold'), bg='#95a5a6', fg='white',
                 width=2, relief='flat', command=lambda: self.tab1_controller.zoom_in()).pack(side='left')
        tk.Button(zf, text="Reset", font=('Segoe UI', 8), bg='#bdc3c7', fg='white',
                 relief='flat', command=lambda: self.tab1_controller.zoom_reset()).pack(side='right')
        
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=12, padx=15)
        
        # Action buttons
        self.tab1_run_btn = tk.Button(p, text="Proses", font=('Segoe UI', 11, 'bold'),
                                      bg='#3498db', fg='white', pady=10, relief='flat',
                                      command=self._tab1_run_inpainting)
        self.tab1_run_btn.pack(pady=8, padx=15, fill='x')
        
        tk.Button(p, text="Hapus Mask", font=('Segoe UI', 10), bg='#5dade2', fg='white',
                 pady=8, relief='flat', command=self._tab1_clear_mask).pack(pady=4, padx=15, fill='x')
        
        self.tab1_toggle_btn = tk.Button(p, text="Lihat Hasil", font=('Segoe UI', 10),
                                         bg='#85c1e9', fg='white', pady=8, relief='flat',
                                         command=self._tab1_toggle_view, state='disabled')
        self.tab1_toggle_btn.pack(pady=4, padx=15, fill='x')
        
        self.tab1_save_btn = tk.Button(p, text="Simpan Hasil", font=('Segoe UI', 10),
                                       bg='#85c1e9', fg='white', pady=8, relief='flat',
                                       command=self._tab1_save_image, state='disabled')
        self.tab1_save_btn.pack(pady=4, padx=15, fill='x')
        
        tk.Button(p, text="Gambar Baru", font=('Segoe UI', 10), bg='#95a5a6', fg='white',
                 pady=8, relief='flat', command=self._tab1_new_image).pack(pady=4, padx=15, fill='x')
    
    def _tab1_upload_image(self):
        fp = filedialog.askopenfilename(title="Select Image",
                                        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            img = cv2.imread(fp)
            if img is not None:
                w, h = self.tab1_controller.load_image(img)
                self.tab1_upload_frame.pack_forget()
                self.tab1_canvas_frame.pack(fill='both', expand=True)
                self.tab1_image_loaded = True
                self.tab1_status.config(text=f"Gambar dimuat: {w}×{h}")
    
    def _tab1_brush_change(self, v):
        self.tab1_controller.brush_size = int(float(v))
        self.tab1_brush_label.config(text=f"{self.tab1_controller.brush_size} px")
    
    def _tab1_autosnap_toggle(self):
        self.tab1_controller.auto_snap_enabled = self.tab1_autosnap_var.get()
    
    def _tab1_on_update(self):
        if self.tab1_controller:
            self.tab1_zoom_label.config(text=f"{int(self.tab1_controller.zoom_level * 100)}%")
            self.tab1_undo_btn.config(state='normal' if self.tab1_controller.can_undo() else 'disabled',
                                      bg='#95a5a6' if self.tab1_controller.can_undo() else '#bdc3c7')
            self.tab1_redo_btn.config(state='normal' if self.tab1_controller.can_redo() else 'disabled',
                                      bg='#95a5a6' if self.tab1_controller.can_redo() else '#bdc3c7')
    
    def _tab1_undo(self):
        if self.tab1_controller:
            self.tab1_controller.undo()
    
    def _tab1_redo(self):
        if self.tab1_controller:
            self.tab1_controller.redo()
    
    def _tab1_clear_mask(self):
        if self.tab1_controller:
            self.tab1_controller.clear_mask()
            self.tab1_toggle_btn.config(state='disabled')
            self.tab1_save_btn.config(state='disabled')
            self.tab1_status.config(text="Mask dihapus.")
    
    def _tab1_toggle_view(self):
        if self.tab1_controller and self.tab1_controller.result_image is not None:
            self.tab1_controller.show_result = not self.tab1_controller.show_result
            self.tab1_toggle_btn.config(text="Lihat Asli" if self.tab1_controller.show_result else "Lihat Hasil")
            self.tab1_controller.update_display()
    
    def _tab1_run_inpainting(self):
        if self.is_processing:
            return
        if not self.tab1_controller or self.tab1_controller.mask is None:
            return
        if np.count_nonzero(self.tab1_controller.mask) == 0:
            messagebox.showwarning("Peringatan", "Gambar mask dulu!")
            return
        
        self.is_processing = True
        self.tab1_run_btn.config(state='disabled', bg='#bdc3c7')
        self.tab1_progress.start(10)
        self.tab1_status.config(text="Memproses...")
        
        threading.Thread(target=self._tab1_worker, daemon=True).start()
    
    def _tab1_worker(self):
        try:
            result = self.inpainter.inpaint(
                self.tab1_controller.original_image.copy(),
                self.tab1_controller.mask.copy(),
                padding=20,
                progress_callback=self._tab1_progress_cb
            )
            self.progress_queue.put(('tab1_complete', result))
        except Exception as e:
            self.progress_queue.put(('tab1_error', str(e)))
    
    def _tab1_progress_cb(self, img, it, rem, pct):
        self.progress_queue.put(('tab1_preview', {'image': img.copy(), 'it': it, 'rem': rem, 'pct': pct}))
    
    def _tab1_save_image(self):
        if not self.tab1_controller or self.tab1_controller.result_image is None:
            return
        fp = filedialog.asksaveasfilename(title="Simpan", defaultextension=".png",
                                          filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if fp:
            cv2.imwrite(fp, self.tab1_controller.result_image)
            messagebox.showinfo("Berhasil", f"Disimpan: {fp}")
    
    def _tab1_new_image(self):
        if self.is_processing:
            return
        self.tab1_canvas_frame.pack_forget()
        self.tab1_upload_frame.pack(fill='both', expand=True)
        self.tab1_controller.canvas_image_id = None
        self.tab1_image_loaded = False
        self.tab1_toggle_btn.config(state='disabled')
        self.tab1_save_btn.config(state='disabled')
        self.tab1_status.config(text="Siap. Muat gambar untuk mulai.")
    
    # TAB 2: EVALUATION MODE
    def _create_tab2(self):
        main = tk.Frame(self.tab2_frame, bg='#f0f0f0')
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top: Load buttons
        top = tk.Frame(main, bg='#ecf0f1')
        top.pack(fill='x', pady=(0, 10))
        
        tk.Label(top, text="1. Ground Truth:", font=('Segoe UI', 10, 'bold'),
                bg='#ecf0f1', fg='#34495e').pack(side='left', padx=(10, 5))
        self.tab2_gt_btn = tk.Button(top, text="Muat GT", font=('Segoe UI', 10),
                                     bg='#27ae60', fg='white', padx=15, relief='flat',
                                     command=self._tab2_load_gt)
        self.tab2_gt_btn.pack(side='left', padx=5)
        self.tab2_gt_status = tk.Label(top, text="❌ Belum", font=('Segoe UI', 9),
                                       bg='#ecf0f1', fg='#e74c3c')
        self.tab2_gt_status.pack(side='left', padx=(0, 20))
        
        tk.Label(top, text="2. Input Image:", font=('Segoe UI', 10, 'bold'),
                bg='#ecf0f1', fg='#34495e').pack(side='left', padx=(10, 5))
        self.tab2_input_btn = tk.Button(top, text="Muat Input", font=('Segoe UI', 10),
                                        bg='#27ae60', fg='white', padx=15, relief='flat',
                                        command=self._tab2_load_input)
        self.tab2_input_btn.pack(side='left', padx=5)
        self.tab2_input_status = tk.Label(top, text="❌ Belum", font=('Segoe UI', 9),
                                          bg='#ecf0f1', fg='#e74c3c')
        self.tab2_input_status.pack(side='left')
        
        # Center: Canvas
        center = tk.Frame(main, bg='#f0f0f0')
        center.pack(fill='both', expand=True)
        
        canvas_frame = tk.Frame(center, bg='#ffffff', relief='solid', bd=1)
        canvas_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(canvas_frame, text="3. Gambar mask pada area teks/SFX",
                font=('Segoe UI', 10), bg='#ffffff', fg='#7f8c8d').pack(pady=5)
        
        self.tab2_canvas = tk.Canvas(canvas_frame, bg='#e0e0e0',
                                     highlightthickness=0, cursor='crosshair')
        self.tab2_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create controller
        self.tab2_controller = CanvasController(self.tab2_canvas, self._tab2_on_update)
        
        # Right: Controls & Results
        right = tk.Frame(center, bg='#ecf0f1', width=320)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        self._create_tab2_controls(right)
        
        # Bottom: Progress
        bottom = tk.Frame(self.tab2_frame, bg='#f0f0f0')
        bottom.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        self.tab2_progress = ttk.Progressbar(bottom, mode='determinate', length=400)
        self.tab2_progress.pack(side='left', padx=(0, 10))
        self.tab2_status = tk.Label(bottom, text="Muat GT dan Input untuk mulai.",
                                   font=('Segoe UI', 10), bg='#f0f0f0', fg='#2c3e50')
        self.tab2_status.pack(side='left')
    
    def _create_tab2_controls(self, p):
        # Brush controls
        tk.Label(p, text="Pengaturan Kuas", font=('Segoe UI', 12, 'bold'),
                bg='#ecf0f1', fg='#2c3e50').pack(pady=(15, 10))
        
        bf = tk.Frame(p, bg='#ecf0f1')
        bf.pack(pady=5, padx=15, fill='x')
        tk.Label(bf, text="Ukuran:", font=('Segoe UI', 10), bg='#ecf0f1', fg='#34495e').pack(anchor='w')
        self.tab2_brush_var = tk.IntVar(value=15)
        ttk.Scale(bf, from_=5, to=50, orient='horizontal', variable=self.tab2_brush_var,
                 command=self._tab2_brush_change).pack(fill='x', pady=5)
        self.tab2_brush_label = tk.Label(bf, text="15 px", font=('Segoe UI', 9),
                                        bg='#ecf0f1', fg='#7f8c8d')
        self.tab2_brush_label.pack(anchor='e')
        
        self.tab2_autosnap_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bf, text="Deteksi Otomatis", variable=self.tab2_autosnap_var,
                      font=('Segoe UI', 9), bg='#ecf0f1',
                      command=self._tab2_autosnap_toggle).pack(anchor='w')
        
        # Zoom
        zf = tk.Frame(p, bg='#ecf0f1')
        zf.pack(pady=5, padx=15, fill='x')
        tk.Button(zf, text="−", font=('Segoe UI', 10), bg='#95a5a6', fg='white',
                 width=2, relief='flat', command=lambda: self.tab2_controller.zoom_out()).pack(side='left')
        self.tab2_zoom_label = tk.Label(zf, text="100%", font=('Segoe UI', 9),
                                        bg='#ecf0f1', fg='#34495e', width=6)
        self.tab2_zoom_label.pack(side='left', padx=2)
        tk.Button(zf, text="+", font=('Segoe UI', 10), bg='#95a5a6', fg='white',
                 width=2, relief='flat', command=lambda: self.tab2_controller.zoom_in()).pack(side='left')
        tk.Button(zf, text="Clear", font=('Segoe UI', 8), bg='#e74c3c', fg='white',
                 relief='flat', command=lambda: self.tab2_controller.clear_mask()).pack(side='right')
        
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=10, padx=15)
        
        # Run benchmark
        self.tab2_run_btn = tk.Button(p, text="4. Jalankan Benchmark",
                                      font=('Segoe UI', 11, 'bold'),
                                      bg='#3498db', fg='white', pady=12, relief='flat',
                                      command=self._tab2_run_benchmark, state='disabled')
        self.tab2_run_btn.pack(pady=10, padx=15, fill='x')
        
        ttk.Separator(p, orient='horizontal').pack(fill='x', pady=10, padx=15)
        
        # Results table
        tk.Label(p, text="Hasil Benchmark", font=('Segoe UI', 11, 'bold'),
                bg='#ecf0f1', fg='#2c3e50').pack(pady=(5, 5))
        
        cols = ('method', 'time', 'psnr', 'ssim')
        self.tab2_tree = ttk.Treeview(p, columns=cols, show='headings', height=4)
        self.tab2_tree.heading('method', text='Metode')
        self.tab2_tree.heading('time', text='Time (s)')
        self.tab2_tree.heading('psnr', text='PSNR')
        self.tab2_tree.heading('ssim', text='SSIM')
        self.tab2_tree.column('method', width=110)
        self.tab2_tree.column('time', width=60, anchor='center')
        self.tab2_tree.column('psnr', width=60, anchor='center')
        self.tab2_tree.column('ssim', width=60, anchor='center')
        self.tab2_tree.pack(pady=5, padx=10, fill='x')
        
        # Export
        self.tab2_export_btn = tk.Button(p, text="Ekspor Hasil", font=('Segoe UI', 10),
                                         bg='#9b59b6', fg='white', pady=8, relief='flat',
                                         command=self._tab2_export, state='disabled')
        self.tab2_export_btn.pack(pady=10, padx=15, fill='x')
        
        # Show comparison
        self.tab2_show_btn = tk.Button(p, text="Tampilkan Perbandingan", font=('Segoe UI', 10),
                                       bg='#e67e22', fg='white', pady=8, relief='flat',
                                       command=self._tab2_show_comparison, state='disabled')
        self.tab2_show_btn.pack(pady=5, padx=15, fill='x')
    
    def _tab2_load_gt(self):
        fp = filedialog.askopenfilename(title="Pilih Ground Truth (Gambar Bersih)",
                                        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            self.eval_ground_truth = cv2.imread(fp)
            if self.eval_ground_truth is not None:
                h, w = self.eval_ground_truth.shape[:2]
                self.tab2_gt_status.config(text=f"✓ {w}×{h}", fg='#27ae60')
                self._tab2_check_ready()
    
    def _tab2_load_input(self):
        fp = filedialog.askopenfilename(title="Pilih Input Image (Dengan Teks/SFX)",
                                        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if fp:
            img = cv2.imread(fp)
            if img is not None:
                self.eval_input_image = img
                h, w = img.shape[:2]
                self.tab2_input_status.config(text=f"✓ {w}×{h}", fg='#27ae60')
                
                # Load into canvas for masking
                self.tab2_controller.load_image(img)
                self._tab2_check_ready()
    
    def _tab2_check_ready(self):
        if self.eval_ground_truth is not None and self.eval_input_image is not None:
            # Check dimensions match
            gt_shape = self.eval_ground_truth.shape[:2]
            in_shape = self.eval_input_image.shape[:2]
            if gt_shape != in_shape:
                messagebox.showwarning("Peringatan", 
                    f"Ukuran tidak cocok!\nGT: {gt_shape[1]}×{gt_shape[0]}\nInput: {in_shape[1]}×{in_shape[0]}")
            self.tab2_run_btn.config(state='normal', bg='#3498db')
            self.tab2_status.config(text="Siap! Gambar mask lalu klik 'Jalankan Benchmark'.")
        else:
            self.tab2_run_btn.config(state='disabled', bg='#bdc3c7')
    
    def _tab2_brush_change(self, v):
        self.tab2_controller.brush_size = int(float(v))
        self.tab2_brush_label.config(text=f"{self.tab2_controller.brush_size} px")
    
    def _tab2_autosnap_toggle(self):
        self.tab2_controller.auto_snap_enabled = self.tab2_autosnap_var.get()
    
    def _tab2_on_update(self):
        if self.tab2_controller:
            self.tab2_zoom_label.config(text=f"{int(self.tab2_controller.zoom_level * 100)}%")
    
    def _tab2_run_benchmark(self):
        if not SKIMAGE_AVAILABLE:
            messagebox.showerror("Error", "Install scikit-image!\npip install scikit-image")
            return
        
        if self.eval_is_running:
            return
        
        # Validate
        if self.eval_ground_truth is None or self.eval_input_image is None:
            messagebox.showwarning("Peringatan", "Muat GT dan Input dulu!")
            return
        
        if self.tab2_controller.mask is None or np.count_nonzero(self.tab2_controller.mask) == 0:
            messagebox.showwarning("Peringatan", "Gambar mask dulu pada area teks!")
            return
        
        self.eval_is_running = True
        self.tab2_run_btn.config(state='disabled', bg='#bdc3c7')
        self.tab2_progress['value'] = 0
        self.tab2_status.config(text="Menjalankan benchmark...")
        
        # Clear previous results
        for item in self.tab2_tree.get_children():
            self.tab2_tree.delete(item)
        self.eval_results = {}
        
        threading.Thread(target=self._tab2_worker, daemon=True).start()
        self._check_eval_queue()
    
    def _tab2_worker(self):
        try:
            inp = HybridInpainter(target_proc_size=450, patch_size=9)
            input_img = self.eval_input_image.copy()
            mask = self.tab2_controller.mask.copy()
            gt = self.eval_ground_truth
            
            # Store current method name for callback
            current_method = [""]
            
            # Callback for live visualization
            def benchmark_callback(image, iteration, remaining, percent):
                self.eval_queue.put(('eval_preview', {
                    'image': image.copy(),
                    'method': current_method[0],
                    'iteration': iteration,
                    'remaining': remaining,
                    'percent': percent
                }))
            
            methods = [
                ("Telea", "telea", False),
                ("Criminisi Std", "criminisi_standard", True),
                ("Hybrid (Ours)", "adaptive", True)
            ]
            
            num_methods = len(methods)
            for i, (name, key, use_callback) in enumerate(methods):
                current_method[0] = name
                self.eval_queue.put(('status', f"🔄 Running: {name}..."))
                self.eval_queue.put(('progress', (i / num_methods) * 100))
                self.eval_queue.put(('method_start', name))
                
                t0 = time.time()
                
                if use_callback:
                    # Run with live preview callback
                    result = inp.inpaint(
                        input_img.copy(), mask.copy(), 
                        method=key,
                        progress_callback=benchmark_callback
                    )
                else:
                    # Run without callback (Telea is instant)
                    result = inp.inpaint(input_img.copy(), mask.copy(), method=key)
                    # Show final result immediately
                    self.eval_queue.put(('eval_preview', {
                        'image': result.copy(),
                        'method': name,
                        'iteration': 1,
                        'remaining': 0,
                        'percent': 100.0
                    }))
                
                elapsed = time.time() - t0
                
                # Calculate metrics against GT
                psnr_val = calc_psnr(gt, result)
                ssim_val = calc_ssim(gt, result, channel_axis=2, data_range=255)
                
                self.eval_results[name] = {
                    'result': result,
                    'time': elapsed,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                }
                
                # Show final result for this method
                self.eval_queue.put(('eval_preview', {
                    'image': result.copy(),
                    'method': name,
                    'iteration': -1,  # -1 = final
                    'remaining': 0,
                    'percent': 100.0
                }))
                
                self.eval_queue.put(('result', (name, elapsed, psnr_val, ssim_val)))
                self.eval_queue.put(('method_complete', name))
            
            self.eval_queue.put(('progress', 100))
            self.eval_queue.put(('complete', None))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.eval_queue.put(('error', str(e)))
    
    def _check_eval_queue(self):
        try:
            while not self.eval_queue.empty():
                t, d = self.eval_queue.get_nowait()
                
                if t == 'status':
                    self.tab2_status.config(text=d)
                
                elif t == 'progress':
                    self.tab2_progress['value'] = d
                
                elif t == 'method_start':
                    # Highlight which method is running
                    self.tab2_status.config(text=f"🎨 Visualizing: {d}...")
                
                elif t == 'method_complete':
                    self.tab2_status.config(text=f"✓ {d} complete!")
                
                elif t == 'eval_preview':
                    # Live visualization update
                    img = d['image']
                    method = d['method']
                    iteration = d['iteration']
                    remaining = d['remaining']
                    percent = d['percent']
                    
                    # Update canvas with preview
                    self._update_eval_canvas(img)
                    
                    # Update status with progress info
                    if iteration == -1:
                        self.tab2_status.config(text=f"✓ {method} - Complete!")
                    else:
                        self.tab2_status.config(
                            text=f"🎨 {method} | It:{iteration} | Rem:{remaining}px | {percent:.1f}%"
                        )
                
                elif t == 'result':
                    name, elapsed, psnr_val, ssim_val = d
                    self.tab2_tree.insert('', 'end', values=(
                        name, f"{elapsed:.2f}", f"{psnr_val:.2f}", f"{ssim_val:.4f}"
                    ))
                
                elif t == 'complete':
                    self.eval_is_running = False
                    self.tab2_run_btn.config(state='normal', bg='#3498db')
                    self.tab2_export_btn.config(state='normal')
                    self.tab2_show_btn.config(state='normal')
                    self.tab2_status.config(text="✓ Benchmark selesai!")
                    self._print_markdown()
                    # Restore original input display
                    self._update_eval_canvas(self.eval_input_image)
                
                elif t == 'error':
                    self.eval_is_running = False
                    self.tab2_run_btn.config(state='normal', bg='#3498db')
                    self.tab2_status.config(text=f"Error: {d}")
                    messagebox.showerror("Error", str(d))
        except:
            pass
        
        if self.eval_is_running:
            self.root.after(50, self._check_eval_queue)  # Faster refresh for smoother animation
    
    def _update_eval_canvas(self, image: np.ndarray):
        if image is None:
            return
        
        # Use controller's scale factor if available
        ctrl = self.tab2_controller
        h, w = image.shape[:2]
        
        # Use same scaling as controller
        if ctrl and ctrl.scale_factor:
            scale = ctrl.scale_factor
        else:
            max_w, max_h = 700, 550
            scale = min(max_w / w, max_h / h, 1.0)
        
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize for display
        display = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply zoom if set
        if ctrl and ctrl.zoom_level != 1.0:
            zh, zw = int(new_h * ctrl.zoom_level), int(new_w * ctrl.zoom_level)
            display = cv2.resize(display, (zw, zh), interpolation=cv2.INTER_LINEAR)
        
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(display_rgb)
        self._eval_preview_tk = ImageTk.PhotoImage(pil_img)
        
        # Get pan offset from controller
        px = ctrl.pan_offset_x if ctrl else 0
        py = ctrl.pan_offset_y if ctrl else 0
        
        # Update canvas
        if hasattr(self, '_eval_canvas_img_id') and self._eval_canvas_img_id:
            self.tab2_canvas.itemconfig(self._eval_canvas_img_id, image=self._eval_preview_tk)
            self.tab2_canvas.coords(self._eval_canvas_img_id, px, py)
        else:
            self._eval_canvas_img_id = self.tab2_canvas.create_image(
                px, py, anchor='nw', image=self._eval_preview_tk
            )
    
    def _print_markdown(self):
        print("\n" + "="*70)
        print("BENCHMARK RESULTS (Markdown - Copy for Thesis)")
        print("="*70)
        print("\n| Metode | Waktu (s) | PSNR (dB) | SSIM |")
        print("|--------|-----------|-----------|------|")
        for n, d in self.eval_results.items():
            print(f"| {n} | {d['time']:.2f} | {d['psnr']:.2f} | {d['ssim']:.4f} |")
        print("\n" + "="*70)
    
    def _tab2_show_comparison(self):
        if not self.eval_results:
            return
        
        # Create comparison image
        h, w = self.eval_input_image.shape[:2]
        
        # Calculate thumbnail size to fit max 1600px total width
        MAX_TOTAL_WIDTH = 1600
        num_images = 6  # Input, Telea, NS, Criminisi, Hybrid, GT
        gap = 4
        label_h = 28
        
        # Calculate max thumb width
        available_width = MAX_TOTAL_WIDTH - (num_images - 1) * gap
        thumb_w = min(available_width // num_images, w)
        scale = thumb_w / w
        thumb_h = int(h * scale)
        
        def resize(img):
            return cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        
        images = [
            ("Input", resize(self.eval_input_image)),
            ("Telea", resize(self.eval_results.get("Telea", {}).get('result', self.eval_input_image))),
            ("Criminisi", resize(self.eval_results.get("Criminisi Std", {}).get('result', self.eval_input_image))),
            ("Hybrid", resize(self.eval_results.get("Hybrid (Ours)", {}).get('result', self.eval_input_image))),
            ("Ground Truth", resize(self.eval_ground_truth)),
        ]
        
        # Build grid
        grid_w = len(images) * thumb_w + (len(images) - 1) * gap
        grid_h = thumb_h + label_h
        
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40
        
        x = 0
        for name, img in images:
            grid[0:thumb_h, x:x+thumb_w] = img
            # Label background
            cv2.rectangle(grid, (x, thumb_h), (x + thumb_w, grid_h), (60, 60, 60), -1)
            # Label text (smaller font for compact display)
            font_scale = 0.45 if len(name) > 8 else 0.5
            ts = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = x + (thumb_w - ts[0]) // 2
            ty = thumb_h + (label_h + ts[1]) // 2
            cv2.putText(grid, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            x += thumb_w + gap
        
        # Always resize to fit max width (ensures it fits on screen)
        if grid_w > MAX_TOTAL_WIDTH:
            final_scale = MAX_TOTAL_WIDTH / grid_w
            new_grid_w = int(grid_w * final_scale)
            new_grid_h = int(grid_h * final_scale)
            grid = cv2.resize(grid, (new_grid_w, new_grid_h), interpolation=cv2.INTER_AREA)
        
        # Show with OpenCV
        cv2.imshow("Benchmark Comparison (Press any key to close)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _tab2_export(self):
        if not self.eval_results:
            return
        
        od = filedialog.askdirectory(title="Pilih Folder Output")
        if not od:
            return
        
        import os
        cv2.imwrite(os.path.join(od, "ground_truth.png"), self.eval_ground_truth)
        cv2.imwrite(os.path.join(od, "input_image.png"), self.eval_input_image)
        cv2.imwrite(os.path.join(od, "mask.png"), self.tab2_controller.mask)
        
        for name, data in self.eval_results.items():
            sn = name.replace(" ", "_").replace("(", "").replace(")", "")
            cv2.imwrite(os.path.join(od, f"result_{sn}.png"), data['result'])
        
        # CSV
        with open(os.path.join(od, "results.csv"), 'w') as f:
            f.write("Method,Time,PSNR,SSIM\n")
            for n, d in self.eval_results.items():
                f.write(f"{n},{d['time']:.4f},{d['psnr']:.4f},{d['ssim']:.6f}\n")
        
        messagebox.showinfo("Berhasil", f"Hasil diekspor ke:\n{od}")
    
    # QUEUE CHECKER
    def _check_queues(self):
        # Tab 1 queue
        try:
            while not self.progress_queue.empty():
                t, d = self.progress_queue.get_nowait()
                
                if t == 'tab1_preview':
                    self.tab1_controller.result_image = d['image']
                    self.tab1_controller.show_result = True
                    self.tab1_controller.update_display()
                    self.tab1_status.config(text=f"It:{d['it']} | Sisa:{d['rem']}px | {d['pct']:.1f}%")
                
                elif t == 'tab1_complete':
                    self.tab1_controller.result_image = d
                    self.tab1_controller.show_result = True
                    self.tab1_controller.update_display()
                    self.is_processing = False
                    self.tab1_progress.stop()
                    self.tab1_run_btn.config(state='normal', bg='#3498db')
                    self.tab1_save_btn.config(state='normal')
                    self.tab1_toggle_btn.config(state='normal')
                    self.tab1_status.config(text="✓ Selesai!")
                
                elif t == 'tab1_error':
                    self.is_processing = False
                    self.tab1_progress.stop()
                    self.tab1_run_btn.config(state='normal', bg='#3498db')
                    self.tab1_status.config(text="Error!")
                    messagebox.showerror("Error", d)
        except:
            pass
        
        self.root.after(100, self._check_queues)
    
    def run(self):
        self.root.mainloop()


def main():
    root = tk.Tk()
    app = MangaCleanerApp(root)
    app.run()


if __name__ == "__main__":
    main()
