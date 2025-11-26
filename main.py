import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import threading
import queue
from typing import Optional, Tuple
import time

from inpainter import HybridInpainter


class MangaCleanerApp:
    """
    Professional Manga Text Inpainting Application with Tkinter GUI.
    
    Features:
    - Drag-and-drop style image upload
    - Interactive brush tool for masking
    - Real-time inpainting preview
    - Progress tracking with status updates
    - Professional UI with ttk styling
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎨 Manga Text Inpainting Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Image data
        self.original_image: Optional[np.ndarray] = None  # Full resolution (BGR)
        self.display_image: Optional[np.ndarray] = None   # Scaled for display (BGR)
        self.mask: Optional[np.ndarray] = None            # Full resolution mask
        self.display_mask: Optional[np.ndarray] = None    # Scaled mask
        self.result_image: Optional[np.ndarray] = None    # Inpainting result
        
        # Display scaling
        self.scale_factor = 1.0  # display_size / original_size
        self.max_display_width = 800
        self.max_display_height = 650
        
        # Drawing state
        self.is_drawing = False
        self.brush_size = 15
        self.last_x = None
        self.last_y = None
        
        # Auto-snap brush state
        self.auto_snap_enabled = True  # Smart segmentation on by default
        self.stroke_bbox = None  # Track bounding box of current stroke (x1, y1, x2, y2)
        
        # Undo/Redo history
        self.mask_history = []  # List of mask states
        self.history_index = -1  # Current position in history
        self.max_history = 50  # Maximum undo steps
        
        # View state
        self.show_result = False  # False = show original, True = show result
        
        # Zoom state
        self.zoom_level = 1.0  # 1.0 = 100%, 2.0 = 200%, etc.
        self.zoom_min = 0.5
        self.zoom_max = 4.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Inpainting state
        self.inpainter = HybridInpainter(target_proc_size=350, patch_size=9)
        self.is_processing = False
        self.progress_queue = queue.Queue()
        
        # Canvas references
        self.canvas_image_id = None
        self.tk_image = None  # Keep reference to prevent garbage collection
        
        # Build UI
        self._create_startup_screen()
        
        # Start queue checker for thread-safe updates
        self._check_progress_queue()
    
    def _create_startup_screen(self):
        """Create the initial upload screen."""
        self.startup_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.startup_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = tk.Label(
            self.startup_frame,
            text="🎨 Manga Text Inpainting Tool",
            font=('Arial', 28, 'bold'),
            bg='#f0f0f0',
            fg='#333333'
        )
        title_label.pack(pady=(100, 20))
        
        # Subtitle
        subtitle_label = tk.Label(
            self.startup_frame,
            text="Remove text from manga images with AI-powered inpainting",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#666666'
        )
        subtitle_label.pack(pady=(0, 50))
        
        # Upload button
        upload_btn = tk.Button(
            self.startup_frame,
            text="📁 Upload Image",
            font=('Arial', 16, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            padx=40,
            pady=20,
            cursor='hand2',
            command=self._upload_image
        )
        upload_btn.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(
            self.startup_frame,
            text="Supported formats: JPG, PNG\nMax size: 4000x4000 pixels",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#999999',
            justify='center'
        )
        instructions.pack(pady=(10, 0))
    
    def _upload_image(self):
        """Handle image upload via file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Manga Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image with OpenCV (BGR format)
                self.original_image = cv2.imread(file_path)
                
                if self.original_image is None:
                    raise ValueError("Could not load image")
                
                # Initialize masks
                h, w = self.original_image.shape[:2]
                self.mask = np.zeros((h, w), dtype=np.uint8)
                self.result_image = None
                
                # Initialize history
                self.mask_history = [self.mask.copy()]
                self.history_index = 0
                self.show_result = False
                self.zoom_level = 1.0
                self.pan_offset_x = 0
                self.pan_offset_y = 0
                
                # Calculate display scaling
                self._calculate_display_scale()
                
                # Create scaled display image
                display_h = int(h * self.scale_factor)
                display_w = int(w * self.scale_factor)
                self.display_image = cv2.resize(
                    self.original_image,
                    (display_w, display_h),
                    interpolation=cv2.INTER_AREA
                )
                self.display_mask = np.zeros((display_h, display_w), dtype=np.uint8)
                
                # Switch to editor
                self._create_editor_screen()
                
                print(f"[APP] Image loaded: {w}×{h}, scale: {self.scale_factor:.3f}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def _calculate_display_scale(self):
        """Calculate scale factor to fit image in display area."""
        h, w = self.original_image.shape[:2]
        
        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h
        
        self.scale_factor = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    def _create_editor_screen(self):
        """Create the main editor interface."""
        # Remove startup screen
        self.startup_frame.destroy()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side: Canvas
        left_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', bd=1)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.canvas_label = tk.Label(
            left_frame,
            text="Draw mask over text to remove",
            font=('Arial', 10),
            bg='#ffffff',
            fg='#666666'
        )
        self.canvas_label.pack(pady=5)
        
        self.canvas = tk.Canvas(
            left_frame,
            bg='#e0e0e0',
            highlightthickness=0,
            cursor='crosshair'
        )
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Bind mouse events
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)  # Zoom with mouse wheel
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
        self.root.bind('<Control-Z>', lambda e: self._redo())  # Ctrl+Shift+Z
        
        # Display initial image
        self._update_canvas_display()
        
        # Right side: Control Panel
        right_frame = tk.Frame(main_frame, bg='#ffffff', relief='solid', bd=1, width=300)
        right_frame.pack(side='right', fill='y', padx=(0, 0))
        right_frame.pack_propagate(False)
        
        # Tools label
        tools_label = tk.Label(
            right_frame,
            text="🛠️ Tools",
            font=('Arial', 16, 'bold'),
            bg='#ffffff',
            fg='#333333'
        )
        tools_label.pack(pady=(20, 10))
        
        # Brush size slider
        brush_frame = tk.Frame(right_frame, bg='#ffffff')
        brush_frame.pack(pady=10, padx=20, fill='x')
        
        brush_label = tk.Label(
            brush_frame,
            text="Brush Size:",
            font=('Arial', 11),
            bg='#ffffff'
        )
        brush_label.pack(anchor='w')
        
        self.brush_size_var = tk.IntVar(value=15)
        self.brush_slider = ttk.Scale(
            brush_frame,
            from_=5,
            to=50,
            orient='horizontal',
            variable=self.brush_size_var,
            command=self._on_brush_size_change
        )
        self.brush_slider.pack(fill='x', pady=5)
        
        self.brush_size_label = tk.Label(
            brush_frame,
            text="15 px",
            font=('Arial', 9),
            bg='#ffffff',
            fg='#666666'
        )
        self.brush_size_label.pack(anchor='e')
        
        # Auto-Snap Checkbox
        self.auto_snap_var = tk.BooleanVar(value=True)
        self.auto_snap_check = tk.Checkbutton(
            brush_frame,
            text="🎯 Auto-Snap Mask (Smart Segmentation)",
            variable=self.auto_snap_var,
            font=('Arial', 10),
            bg='#ffffff',
            activebackground='#ffffff',
            command=self._on_auto_snap_toggle
        )
        self.auto_snap_check.pack(anchor='w', pady=(10, 0))
        
        auto_snap_hint = tk.Label(
            brush_frame,
            text="Automatically snaps to black text edges",
            font=('Arial', 8, 'italic'),
            bg='#ffffff',
            fg='#888888'
        )
        auto_snap_hint.pack(anchor='w', pady=(2, 0))
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15, padx=20)
        
        # Undo/Redo buttons
        history_frame = tk.Frame(right_frame, bg='#ffffff')
        history_frame.pack(pady=10, padx=20, fill='x')
        
        self.undo_btn = tk.Button(
            history_frame,
            text="↶ Undo",
            font=('Arial', 10),
            bg='#9E9E9E',
            fg='white',
            activebackground='#757575',
            activeforeground='white',
            padx=10,
            pady=5,
            cursor='hand2',
            command=self._undo,
            state='disabled'
        )
        self.undo_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.redo_btn = tk.Button(
            history_frame,
            text="↷ Redo",
            font=('Arial', 10),
            bg='#9E9E9E',
            fg='white',
            activebackground='#757575',
            activeforeground='white',
            padx=10,
            pady=5,
            cursor='hand2',
            command=self._redo,
            state='disabled'
        )
        self.redo_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15, padx=20)
        
        # Zoom controls
        zoom_label = tk.Label(
            right_frame,
            text="🔍 Zoom:",
            font=('Arial', 11, 'bold'),
            bg='#ffffff'
        )
        zoom_label.pack(pady=(0, 5), padx=20, anchor='w')
        
        zoom_frame = tk.Frame(right_frame, bg='#ffffff')
        zoom_frame.pack(pady=5, padx=20, fill='x')
        
        zoom_out_btn = tk.Button(
            zoom_frame,
            text="−",
            font=('Arial', 14, 'bold'),
            bg='#607D8B',
            fg='white',
            activebackground='#455A64',
            activeforeground='white',
            width=3,
            cursor='hand2',
            command=self._zoom_out
        )
        zoom_out_btn.pack(side='left', padx=(0, 5))
        
        self.zoom_label = tk.Label(
            zoom_frame,
            text="100%",
            font=('Arial', 11),
            bg='#ffffff',
            width=8
        )
        self.zoom_label.pack(side='left', padx=5)
        
        zoom_in_btn = tk.Button(
            zoom_frame,
            text="+",
            font=('Arial', 14, 'bold'),
            bg='#607D8B',
            fg='white',
            activebackground='#455A64',
            activeforeground='white',
            width=3,
            cursor='hand2',
            command=self._zoom_in
        )
        zoom_in_btn.pack(side='left', padx=(5, 0))
        
        zoom_reset_btn = tk.Button(
            zoom_frame,
            text="Reset",
            font=('Arial', 9),
            bg='#78909C',
            fg='white',
            activebackground='#546E7A',
            activeforeground='white',
            cursor='hand2',
            command=self._zoom_reset
        )
        zoom_reset_btn.pack(side='right')
        
        zoom_info = tk.Label(
            right_frame,
            text="Tip: Hold Shift + Drag to pan",
            font=('Arial', 8),
            bg='#ffffff',
            fg='#999999'
        )
        zoom_info.pack(pady=(2, 0), padx=20, anchor='w')
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15, padx=20)
        
        # Run Inpainting button
        self.run_btn = tk.Button(
            right_frame,
            text="▶ Run Inpainting",
            font=('Arial', 12, 'bold'),
            bg='#2196F3',
            fg='white',
            activebackground='#1976D2',
            activeforeground='white',
            padx=20,
            pady=10,
            cursor='hand2',
            command=self._run_inpainting
        )
        self.run_btn.pack(pady=10, padx=20, fill='x')
        
        # Clear Mask button
        clear_btn = tk.Button(
            right_frame,
            text="🗑️ Clear Mask",
            font=('Arial', 11),
            bg='#FF9800',
            fg='white',
            activebackground='#F57C00',
            activeforeground='white',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self._clear_mask
        )
        clear_btn.pack(pady=5, padx=20, fill='x')
        
        # Toggle Before/After button
        self.toggle_btn = tk.Button(
            right_frame,
            text="👁️ View Result",
            font=('Arial', 11),
            bg='#9C27B0',
            fg='white',
            activebackground='#7B1FA2',
            activeforeground='white',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self._toggle_view,
            state='disabled'
        )
        self.toggle_btn.pack(pady=5, padx=20, fill='x')
        
        # Save Image button
        self.save_btn = tk.Button(
            right_frame,
            text="💾 Save Result",
            font=('Arial', 11),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self._save_image,
            state='disabled'
        )
        self.save_btn.pack(pady=5, padx=20, fill='x')
        
        # New Image button
        new_btn = tk.Button(
            right_frame,
            text="📁 New Image",
            font=('Arial', 11),
            bg='#9E9E9E',
            fg='white',
            activebackground='#757575',
            activeforeground='white',
            padx=20,
            pady=8,
            cursor='hand2',
            command=self._new_image
        )
        new_btn.pack(pady=5, padx=20, fill='x')
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=15, padx=20)
        
        # Info
        info_label = tk.Label(
            right_frame,
            text="ℹ️ Instructions:\n\n"
                 "1. Draw red mask over text\n"
                 "2. Click 'Run Inpainting'\n"
                 "3. Wait for processing\n"
                 "4. Save the result!",
            font=('Arial', 9),
            bg='#ffffff',
            fg='#666666',
            justify='left'
        )
        info_label.pack(pady=20, padx=20)
        
        # Bottom: Progress bar and status
        bottom_frame = tk.Frame(self.root, bg='#f0f0f0')
        bottom_frame.pack(side='bottom', fill='x', padx=10, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(
            bottom_frame,
            mode='indeterminate',
            length=300
        )
        self.progress_bar.pack(side='left', padx=(0, 10))
        
        self.status_label = tk.Label(
            bottom_frame,
            text="Ready. Draw mask to begin.",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#333333'
        )
        self.status_label.pack(side='left')
    
    def _on_brush_size_change(self, value):
        """Handle brush size slider change."""
        self.brush_size = int(float(value))
        self.brush_size_label.config(text=f"{self.brush_size} px")
    
    def _on_auto_snap_toggle(self):
        """Handle auto-snap checkbox toggle."""
        self.auto_snap_enabled = self.auto_snap_var.get()
        status = "ON" if self.auto_snap_enabled else "OFF"
        print(f"[AUTO-SNAP] Smart segmentation: {status}")
    
    def _on_mouse_down(self, event):
        """Handle mouse button press."""
        # Check if Shift is held for panning
        if event.state & 0x0001:  # Shift key
            self.is_panning = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.canvas.config(cursor='fleur')  # Hand cursor
        else:
            # Start drawing
            self.is_drawing = True
            self.last_x = event.x
            self.last_y = event.y
            
            # Initialize stroke bounding box for auto-snap
            # Convert to original image coordinates
            orig_x = int((event.x - self.pan_offset_x) / (self.zoom_level * self.scale_factor))
            orig_y = int((event.y - self.pan_offset_y) / (self.zoom_level * self.scale_factor))
            self.stroke_bbox = [orig_x, orig_y, orig_x, orig_y]  # [x1, y1, x2, y2]
            
            # Save current state before drawing (for undo)
            self._save_mask_state()
            
            self._draw_point(event.x, event.y)
    
    def _on_mouse_drag(self, event):
        """Handle mouse drag."""
        if self.is_panning:
            # Pan the canvas
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self._update_canvas_display()
        elif self.is_drawing:
            if self.last_x is not None and self.last_y is not None:
                self._draw_line(self.last_x, self.last_y, event.x, event.y)
            self.last_x = event.x
            self.last_y = event.y
    
    def _on_mouse_up(self, event):
        """Handle mouse button release."""
        if self.is_panning:
            self.is_panning = False
            self.canvas.config(cursor='crosshair')
        else:
            # Trigger auto-snap if enabled and we have a valid stroke
            if self.is_drawing and self.auto_snap_enabled and self.stroke_bbox is not None:
                self._refine_mask_auto_snap()
            
            self.is_drawing = False
            self.last_x = None
            self.last_y = None
            self.stroke_bbox = None
    
    def _draw_point(self, x: int, y: int):
        """Draw a single point on the mask (accounting for zoom and pan)."""
        # Convert canvas coordinates to image coordinates (accounting for zoom and pan)
        img_x = int((x - self.pan_offset_x) / self.zoom_level)
        img_y = int((y - self.pan_offset_y) / self.zoom_level)
        
        # Check bounds
        h, w = self.display_mask.shape
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return
        
        # Draw on display mask
        cv2.circle(
            self.display_mask,
            (img_x, img_y),
            self.brush_size // 2,
            255,
            -1
        )
        
        # Map to original coordinates and draw on full mask
        orig_x = int(img_x / self.scale_factor)
        orig_y = int(img_y / self.scale_factor)
        orig_brush_size = max(1, int(self.brush_size / self.scale_factor))
        
        # Update stroke bounding box for auto-snap
        if self.stroke_bbox is not None:
            radius = orig_brush_size // 2
            self.stroke_bbox[0] = min(self.stroke_bbox[0], orig_x - radius)
            self.stroke_bbox[1] = min(self.stroke_bbox[1], orig_y - radius)
            self.stroke_bbox[2] = max(self.stroke_bbox[2], orig_x + radius)
            self.stroke_bbox[3] = max(self.stroke_bbox[3], orig_y + radius)
        
        # Check bounds for original mask
        h_orig, w_orig = self.mask.shape
        if orig_x >= 0 and orig_x < w_orig and orig_y >= 0 and orig_y < h_orig:
            cv2.circle(
                self.mask,
                (orig_x, orig_y),
                orig_brush_size // 2,
                255,
                -1
            )
        
        self._update_canvas_display()
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line on the mask (accounting for zoom and pan)."""
        # Convert canvas coordinates to image coordinates (accounting for zoom and pan)
        img_x1 = int((x1 - self.pan_offset_x) / self.zoom_level)
        img_y1 = int((y1 - self.pan_offset_y) / self.zoom_level)
        img_x2 = int((x2 - self.pan_offset_x) / self.zoom_level)
        img_y2 = int((y2 - self.pan_offset_y) / self.zoom_level)
        
        # Draw on display mask
        cv2.line(
            self.display_mask,
            (img_x1, img_y1),
            (img_x2, img_y2),
            255,
            self.brush_size
        )
        
        # Map to original coordinates and draw on full mask
        orig_x1 = int(img_x1 / self.scale_factor)
        orig_y1 = int(img_y1 / self.scale_factor)
        orig_x2 = int(img_x2 / self.scale_factor)
        orig_y2 = int(img_y2 / self.scale_factor)
        orig_brush_size = max(1, int(self.brush_size / self.scale_factor))
        
        # Update stroke bounding box for auto-snap
        if self.stroke_bbox is not None:
            radius = orig_brush_size // 2
            self.stroke_bbox[0] = min(self.stroke_bbox[0], min(orig_x1, orig_x2) - radius)
            self.stroke_bbox[1] = min(self.stroke_bbox[1], min(orig_y1, orig_y2) - radius)
            self.stroke_bbox[2] = max(self.stroke_bbox[2], max(orig_x1, orig_x2) + radius)
            self.stroke_bbox[3] = max(self.stroke_bbox[3], max(orig_y1, orig_y2) + radius)
        
        cv2.line(
            self.mask,
            (orig_x1, orig_y1),
            (orig_x2, orig_y2),
            255,
            orig_brush_size
        )
        
        self._update_canvas_display()
    
    def _refine_mask_auto_snap(self):
        """
        Refine the mask using Otsu thresholding for automatic text segmentation.
        This "snaps" the rough brush stroke to precise text boundaries.
        """
        if self.stroke_bbox is None or self.original_image is None:
            return
        
        try:
            x1, y1, x2, y2 = self.stroke_bbox
            h_orig, w_orig = self.mask.shape
            
            # Add padding and ensure bounds
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_orig, x2 + padding)
            y2 = min(h_orig, y2 + padding)
            
            # Skip if ROI is too small
            if x2 - x1 < 5 or y2 - y1 < 5:
                return
            
            print(f"[AUTO-SNAP] Refining mask in ROI: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Extract ROI from original image (BGR)
            roi_image = self.original_image[y1:y2, x1:x2].copy()
            roi_mask = self.mask[y1:y2, x1:x2].copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding to separate black text from white background
            # THRESH_BINARY_INV: Makes text (black) become white, background (white) become black
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Only keep pixels that were in the original rough mask
            # This prevents "leaking" to nearby text that wasn't selected
            refined_mask = cv2.bitwise_and(binary, roi_mask)
            
            # AGGRESSIVE DILATION to cover anti-aliased edges (grey fuzzy pixels)
            # Manga text has 2-4px soft edges that Otsu misses
            kernel = np.ones((5, 5), np.uint8)  # Larger kernel for stronger expansion
            refined_mask = cv2.dilate(refined_mask, kernel, iterations=3)
            
            # Ensure the refined mask doesn't extend beyond original ROI bounds
            refined_mask = cv2.bitwise_and(refined_mask, roi_mask)
            
            # Calculate refinement stats
            original_pixels = np.sum(roi_mask > 0)
            refined_pixels = np.sum(refined_mask > 0)
            
            # Only apply if refinement is reasonable (not too different)
            if refined_pixels > 0 and refined_pixels < original_pixels * 2:
                # Update the mask with refined result
                self.mask[y1:y2, x1:x2] = refined_mask
                
                # Update display mask (scaled version)
                display_x1 = int(x1 * self.scale_factor)
                display_y1 = int(y1 * self.scale_factor)
                display_x2 = int(x2 * self.scale_factor)
                display_y2 = int(y2 * self.scale_factor)
                
                refined_display = cv2.resize(
                    refined_mask, 
                    (display_x2 - display_x1, display_y2 - display_y1),
                    interpolation=cv2.INTER_NEAREST
                )
                self.display_mask[display_y1:display_y2, display_x1:display_x2] = refined_display
                
                # Update canvas
                self._update_canvas_display()
                
                print(f"[AUTO-SNAP] ✅ Refined: {original_pixels} → {refined_pixels} pixels")
            else:
                print(f"[AUTO-SNAP] ⚠️ Refinement skipped (too different: {original_pixels} → {refined_pixels})")
                
        except Exception as e:
            print(f"[AUTO-SNAP] ⚠️ Refinement failed: {e}")
            # Keep original mask on error
    
    def _update_canvas_display(self):
        """Update the canvas with current image and mask overlay (with zoom support)."""
        if self.display_image is None:
            return
        
        # PRIORITY 1: If processing, show live preview
        if self.is_processing and self.result_image is not None:
            # Show live inpainting preview
            display = self.result_image.copy()
            # Scale result to display size
            h, w = self.display_image.shape[:2]
            display = cv2.resize(display, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # PRIORITY 2: If toggle is ON and we have result, show it
        elif self.show_result and self.result_image is not None:
            # Show result (no mask overlay)
            display = self.result_image.copy()
            # Scale result to display size
            h, w = self.display_image.shape[:2]
            display = cv2.resize(display, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # PRIORITY 3: Show original with mask overlay
        else:
            # Show original with mask overlay
            display = self.display_image.copy()
            
            # Overlay mask in red (only if not viewing result or processing)
            if not self.show_result and not self.is_processing:
                mask_overlay = self.display_mask > 0
                if np.any(mask_overlay):
                    display[mask_overlay] = display[mask_overlay] * 0.5 + np.array([0, 0, 255]) * 0.5
        
        # Apply zoom
        if self.zoom_level != 1.0:
            h, w = display.shape[:2]
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB for PIL
        display_rgb = cv2.cvtColor(display.astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(display_rgb)
        
        # Apply pan offset (crop the image)
        if self.pan_offset_x != 0 or self.pan_offset_y != 0 or self.zoom_level != 1.0:
            # Calculate visible region
            canvas_w = self.canvas.winfo_width() or 800
            canvas_h = self.canvas.winfo_height() or 600
            
            img_w, img_h = pil_image.size
            
            # Calculate crop box with pan offset
            left = max(0, -self.pan_offset_x)
            top = max(0, -self.pan_offset_y)
            right = min(img_w, canvas_w - self.pan_offset_x)
            bottom = min(img_h, canvas_h - self.pan_offset_y)
            
            # Ensure valid crop box
            if right > left and bottom > top:
                pil_image = pil_image.crop((left, top, right, bottom))
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        x_pos = max(0, self.pan_offset_x)
        y_pos = max(0, self.pan_offset_y)
        
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(
                x_pos, y_pos,
                anchor='nw',
                image=self.tk_image
            )
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)
            self.canvas.coords(self.canvas_image_id, x_pos, y_pos)
    
    def _clear_mask(self):
        """Clear the current mask."""
        if self.mask is not None:
            # Save state for undo
            self._save_mask_state()
            
            self.mask.fill(0)
            self.display_mask.fill(0)
            self.result_image = None
            self.show_result = False
            self.save_btn.config(state='disabled')
            self.toggle_btn.config(state='disabled', text="👁️ View Result")
            self._update_canvas_display()
            self.status_label.config(text="Mask cleared. Ready to draw.")
            self.canvas_label.config(
                text="Draw mask over text to remove",
                fg='#666666',
                font=('Arial', 10)
            )
    
    def _run_inpainting(self):
        """Start the inpainting process in a background thread."""
        if self.is_processing:
            return
        
        # Check if mask is drawn
        if np.count_nonzero(self.mask) == 0:
            messagebox.showwarning("No Mask", "Please draw a mask over the text first!")
            return
        
        # Disable button and start progress
        self.is_processing = True
        self.run_btn.config(state='disabled', bg='#BDBDBD')
        self.progress_bar.start(10)
        self.status_label.config(text="⚡ Initializing inpainting algorithm...")
        self.canvas_label.config(
            text="⏳ Starting inpainting process...",
            fg='#FF9800',
            font=('Arial', 10, 'bold')
        )
        
        # Start inpainting in background thread
        thread = threading.Thread(target=self._inpainting_worker, daemon=True)
        thread.start()
    
    def _inpainting_worker(self):
        """Background worker for inpainting (runs in separate thread)."""
        try:
            print("[APP] Starting inpainting...")
            
            # Run inpainting with progress callback
            result = self.inpainter.inpaint(
                self.original_image.copy(),
                self.mask.copy(),
                padding=20,
                progress_callback=self._progress_callback
            )
            
            # Put result in queue for main thread
            self.progress_queue.put(('complete', result))
            
        except Exception as e:
            print(f"[APP] Error during inpainting: {e}")
            self.progress_queue.put(('error', str(e)))
    
    def _progress_callback(self, preview_image, iteration, remaining, percent):
        """Callback from inpainter (called from background thread)."""
        # Put preview update in queue with progress info
        self.progress_queue.put(('preview', {
            'image': preview_image.copy(),
            'iteration': iteration,
            'remaining': remaining,
            'percent': percent
        }))
    
    def _check_progress_queue(self):
        """Check for updates from background thread (runs in main thread)."""
        try:
            while not self.progress_queue.empty():
                msg_type, data = self.progress_queue.get_nowait()
                
                if msg_type == 'preview':
                    # Update live preview with progress info
                    self.result_image = data['image']
                    
                    # Update status with progress info
                    iteration = data['iteration']
                    remaining = data['remaining']
                    percent = data['percent']
                    self.status_label.config(
                        text=f"⚡ Processing... Iter: {iteration} | Remaining: {remaining} px | {percent:.1f}% done"
                    )
                    
                    # Update canvas label
                    self.canvas_label.config(
                        text="🔴 LIVE PREVIEW - Inpainting in progress...",
                        fg='#FF5722',
                        font=('Arial', 10, 'bold')
                    )
                    
                    # Update canvas to show live preview
                    self._update_canvas_display()
                
                elif msg_type == 'complete':
                    # Inpainting finished
                    self.result_image = data
                    self._update_canvas_display()
                    
                    self.is_processing = False
                    self.progress_bar.stop()
                    self.run_btn.config(state='normal', bg='#2196F3')
                    self.save_btn.config(state='normal')
                    self.toggle_btn.config(state='normal', bg='#9C27B0')
                    self.status_label.config(text="✅ Inpainting complete! Toggle to view result or save.")
                    
                    # Reset canvas label
                    self.canvas_label.config(
                        text="✅ Inpainting complete - Toggle to compare",
                        fg='#4CAF50',
                        font=('Arial', 10, 'bold')
                    )
                    
                    print("[APP] Inpainting complete!")
                
                elif msg_type == 'error':
                    # Error occurred
                    self.is_processing = False
                    self.progress_bar.stop()
                    self.run_btn.config(state='normal', bg='#2196F3')
                    self.status_label.config(text="❌ Error occurred.")
                    
                    messagebox.showerror("Error", f"Inpainting failed:\n{data}")
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_progress_queue)
    
    def _save_image(self):
        """Save the inpainted result."""
        if self.result_image is None:
            messagebox.showwarning("No Result", "Please run inpainting first!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Inpainted Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.result_image)
                messagebox.showinfo("Success", f"Image saved successfully!\n{file_path}")
                self.status_label.config(text=f"✅ Image saved: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
    
    def _save_mask_state(self):
        """Save current mask state to history for undo/redo."""
        if self.mask is None:
            return
        
        # Remove any states after current index (for branching undo)
        self.mask_history = self.mask_history[:self.history_index + 1]
        
        # Add current state
        self.mask_history.append(self.mask.copy())
        self.history_index += 1
        
        # Limit history size
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)
            self.history_index -= 1
        
        # Update button states
        self._update_history_buttons()
    
    def _undo(self):
        """Undo last drawing action."""
        if self.history_index > 0:
            self.history_index -= 1
            self.mask = self.mask_history[self.history_index].copy()
            
            # Update display mask
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(
                self.mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            
            self._update_canvas_display()
            self._update_history_buttons()
            self.status_label.config(text="Undo applied.")
    
    def _redo(self):
        """Redo previously undone action."""
        if self.history_index < len(self.mask_history) - 1:
            self.history_index += 1
            self.mask = self.mask_history[self.history_index].copy()
            
            # Update display mask
            h, w = self.display_mask.shape
            self.display_mask = cv2.resize(
                self.mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            
            self._update_canvas_display()
            self._update_history_buttons()
            self.status_label.config(text="Redo applied.")
    
    def _update_history_buttons(self):
        """Update undo/redo button states."""
        # Undo button
        if self.history_index > 0:
            self.undo_btn.config(state='normal', bg='#607D8B')
        else:
            self.undo_btn.config(state='disabled', bg='#BDBDBD')
        
        # Redo button
        if self.history_index < len(self.mask_history) - 1:
            self.redo_btn.config(state='normal', bg='#607D8B')
        else:
            self.redo_btn.config(state='disabled', bg='#BDBDBD')
    
    def _zoom_in(self):
        """Zoom in (increase zoom level)."""
        if self.zoom_level < self.zoom_max:
            self.zoom_level = min(self.zoom_level + 0.25, self.zoom_max)
            self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
            self._update_canvas_display()
            self.status_label.config(text=f"Zoomed to {int(self.zoom_level * 100)}%")
    
    def _zoom_out(self):
        """Zoom out (decrease zoom level)."""
        if self.zoom_level > self.zoom_min:
            self.zoom_level = max(self.zoom_level - 0.25, self.zoom_min)
            self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
            self._update_canvas_display()
            self.status_label.config(text=f"Zoomed to {int(self.zoom_level * 100)}%")
    
    def _zoom_reset(self):
        """Reset zoom to 100%."""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.zoom_label.config(text="100%")
        self._update_canvas_display()
        self.status_label.config(text="Zoom reset to 100%")
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()
    
    def _toggle_view(self):
        """Toggle between original and result view."""
        if self.result_image is None:
            return
        
        self.show_result = not self.show_result
        
        if self.show_result:
            self.toggle_btn.config(text="👁️ View Original")
            self.status_label.config(text="Viewing result (after inpainting)")
            self.canvas_label.config(
                text="📸 Viewing Result (Cleaned)",
                fg='#2196F3',
                font=('Arial', 10, 'bold')
            )
        else:
            self.toggle_btn.config(text="👁️ View Result")
            self.status_label.config(text="Viewing original (with mask overlay)")
            self.canvas_label.config(
                text="📸 Viewing Original (with Mask)",
                fg='#666666',
                font=('Arial', 10)
            )
        
        self._update_canvas_display()
    
    def _new_image(self):
        """Load a new image."""
        if self.is_processing:
            messagebox.showwarning("Processing", "Please wait for current operation to complete.")
            return
        
        # Reset state
        self.original_image = None
        self.display_image = None
        self.mask = None
        self.display_mask = None
        self.result_image = None
        self.mask_history = []
        self.history_index = -1
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.show_result = False
        
        # Go back to upload screen
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self._create_startup_screen()
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()


def main():
    """Application entry point."""
    root = tk.Tk()
    app = MangaCleanerApp(root)
    app.run()


if __name__ == "__main__":
    main()
