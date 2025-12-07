"""
Inpainting untuk Manga - Metode Criminisi dengan Piramida Adaptif

Fitur utama:
- Proses piramida (450px) untuk menjaga detail SFX
- Ukuran patch adaptif berdasarkan tekstur lokal
- Patch kecil (7-9px) untuk area ramai, besar (11px) untuk area polos
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
import time


class HybridInpainter:
    """Kelas utama inpainting manga dengan patch adaptif."""
    
    def __init__(self, target_proc_size: int = 450, patch_size: int = 9):
        self.target_proc_size = target_proc_size
        self.base_patch_size = patch_size
        self.patch_size = patch_size
        
        # variabel tracking progress
        self.progress_callback: Optional[Callable] = None
        self.is_running = False
        self.total_pixels = 0
        self.filled_pixels = 0
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, 
                padding: int = 20, progress_callback: Optional[Callable] = None,
                method: str = "adaptive") -> np.ndarray:
        """Fungsi utama inpainting. method: telea/criminisi_standard/adaptive"""
        print(f"\n[INPAINTER] 🎨 INPAINTING (method={method})")
        print(f"[INPAINTER] Input image shape: {image.shape}")
        print(f"[INPAINTER] Input mask shape: {mask.shape}")
        print(f"[INPAINTER] Mask pixel count: {np.count_nonzero(mask)}")
        
        self.progress_callback = progress_callback
        self.is_running = True
        
        # cek mask kosong
        if np.count_nonzero(mask) == 0:
            print("[INPAINTER] WARNING: Mask is empty! Returning original.")
            self.is_running = False
            return image.copy()
        
        # METODE TELEA (baseline cepat)
        if method == "telea":
            print("[INPAINTER] Using OpenCV Telea (cv2.INPAINT_TELEA)")
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            result = cv2.inpaint(image, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("[INPAINTER] ✅ Telea inpainting complete!")
            self.is_running = False
            return result
        
        # METODE CRIMINISI STANDAR (lambat, tanpa optimasi)
        if method == "criminisi_standard":
            print("[INPAINTER] Using PURE Standard Criminisi 2004")

            # proses full image tanpa ROI
            result = self._inpaint_standard_criminisi(image.copy(), mask.copy())
            
            print("[INPAINTER] ✅ Standard Criminisi 2004 complete!")
            self.is_running = False
            return result
        
        # METODE ADAPTIF PIRAMIDA (metode yang diusulkan)
        print("[INPAINTER] Using Adaptive Hybrid Pyramid (proposed method)")
        
        # ambil ROI dari gambar
        roi_data = self._extract_roi(image, mask, padding)
        if roi_data is None:
            print("[INPAINTER] WARNING: Could not extract ROI!")
            self.is_running = False
            return image.copy()
        
        roi_img, roi_mask, bbox = roi_data
        y1, y2, x1, x2 = bbox
        
        print(f"[INPAINTER] ROI extracted: {roi_img.shape[1]}×{roi_img.shape[0]}")
        print(f"[INPAINTER] ROI mask pixels: {np.count_nonzero(roi_mask)}")
        
        orig_h, orig_w = roi_img.shape[:2]
        
        # cek perlu downscale atau tidak
        max_dim = max(orig_h, orig_w)
        
        if max_dim > self.target_proc_size * 1.3:
            # MODE PIRAMIDA: kecilkan dulu, proses, besarkan lagi
            print(f"\n[PYRAMID MODE] ⚡")
            print(f"[PYRAMID] Original: {orig_w}×{orig_h}")
            
            # hitung faktor skala
            if orig_h > orig_w:
                scale_factor = self.target_proc_size / orig_h
            else:
                scale_factor = self.target_proc_size / orig_w
            
            new_w = max(50, int(orig_w * scale_factor))
            new_h = max(50, int(orig_h * scale_factor))
            
            print(f"[PYRAMID] Downscaling to: {new_w}×{new_h} (scale: {scale_factor:.3f})")
            print(f"[PYRAMID] Speed boost: ~{1/(scale_factor**2):.1f}x faster!")
            
            # perkecil gambar
            small_img = cv2.resize(roi_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            small_mask = (small_mask > 127).astype(np.uint8) * 255
            
            print(f"[PYRAMID] Processing {np.count_nonzero(small_mask) // 255} pixels at small scale...")
            
            # proses di skala kecil
            result_small = self._inpaint_small(small_img, small_mask, bbox, image, 
                                              original_roi_shape=(orig_h, orig_w),
                                              use_adaptive=True)
            
            # perbesar hasil ke ukuran asli
            print(f"[PYRAMID] Upscaling back to {orig_w}×{orig_h} (CUBIC interpolation)...")
            result_roi = cv2.resize(result_small, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            
            print(f"[PYRAMID] ✅ Pyramid processing complete!")
        else:
            # langsung proses tanpa piramida
            print(f"[INPAINTER] ROI small enough, processing directly...")
            result_roi = self._inpaint_small(roi_img, roi_mask, bbox, image, 
                                            original_roi_shape=None,
                                            use_adaptive=True)
        
        print(f"[INPAINTER] Pasting result back to original image...")
        
        # tempel hasil ke gambar asli
        result = image.copy()
        result[y1:y2, x1:x2] = result_roi
        
        print(f"[INPAINTER] ✅ Inpainting complete!")
        self.is_running = False
        
        return result
    
    def _extract_roi(self, image: np.ndarray, mask: np.ndarray, 
                     padding: int) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple]]:
        """Ambil area ROI di sekitar mask."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        h, w = image.shape[:2]
        y1 = max(0, y_min - padding)
        y2 = min(h, y_max + padding + 1)
        x1 = max(0, x_min - padding)
        x2 = min(w, x_max + padding + 1)
        
        roi_img = image[y1:y2, x1:x2].copy()
        roi_mask = mask[y1:y2, x1:x2].copy()
        
        return roi_img, roi_mask, (y1, y2, x1, x2)
    
    # CRIMINISI STANDAR 2004 - baseline murni tanpa optimasi
    def _inpaint_standard_criminisi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Implementasi Criminisi 2004 murni. Lambat tapi jujur sebagai baseline."""
        PATCH_SIZE = 9
        half = PATCH_SIZE // 2
        
        print(f"[STD-CRIMINISI] ══════════════════════════════════════")
        print(f"[STD-CRIMINISI] PURE Criminisi 2004 (No Optimizations)")
        print(f"[STD-CRIMINISI] Patch: {PATCH_SIZE}x{PATCH_SIZE} FIXED")
        print(f"[STD-CRIMINISI] Search: FULL IMAGE (slow)")
        print(f"[STD-CRIMINISI] Priority: Check ALL boundary pixels")
        print(f"[STD-CRIMINISI] ══════════════════════════════════════")
        
        # copy untuk kerja
        working_img = image.astype(np.float32)
        working_mask = (mask > 0).astype(np.uint8)
        
        if working_mask.sum() == 0:
            return image.copy()
        
        # inisialisasi confidence
        confidence = (1.0 - working_mask).astype(np.float32)
        
        # hitung gradien
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        h, w = image.shape[:2]
        initial_pixels = working_mask.sum()
        iteration = 0
        max_iterations = 10000
        
        print(f"[STD-CRIMINISI] Pixels to fill: {initial_pixels}")
        print(f"[STD-CRIMINISI] Image size: {w}x{h} = {w*h} pixels to search")
        
        last_update_time = time.time()
        start_time = time.time()
        
        while working_mask.sum() > 0 and iteration < max_iterations:
            # LANGKAH 1: cari piksel tepi
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(working_mask, kernel, iterations=1)
            fill_front = np.argwhere((dilated > 0) & (working_mask > 0))
            
            if len(fill_front) == 0:
                fill_front = np.argwhere(working_mask > 0)
                if len(fill_front) == 0:
                    break
            
            # LANGKAH 2: hitung prioritas semua piksel tepi
            best_priority = -1
            best_pixel = None
            for pixel in fill_front:
                py, px = pixel
                
                if py - half < 0 or py + half + 1 > h or px - half < 0 or px + half + 1 > w:
                    continue
                
                # hitung confidence C(p)
                patch_conf = confidence[py-half:py+half+1, px-half:px+half+1]
                C_p = np.mean(patch_conf)
                
                # hitung data term D(p)
                gx = grad_x[py, px]
                gy = grad_y[py, px]
                D_p = np.sqrt(gx**2 + gy**2) / 255.0
                
                # prioritas = C * D
                priority = C_p * (D_p + 0.001)
                
                if priority > best_priority:
                    best_priority = priority
                    best_pixel = (py, px)
            
            if best_pixel is None:
                idx = np.random.randint(0, len(fill_front))
                best_pixel = tuple(fill_front[idx])
            
            ty, tx = best_pixel
            
            # LANGKAH 3: cari patch terbaik di seluruh gambar
            if ty - half < 0 or ty + half + 1 > h or tx - half < 0 or tx + half + 1 > w:
                working_mask[ty, tx] = 0
                iteration += 1
                continue
            
            target_patch = working_img[ty-half:ty+half+1, tx-half:tx+half+1].copy()
            target_patch_mask = working_mask[ty-half:ty+half+1, tx-half:tx+half+1]
            known_in_target = (target_patch_mask == 0)
            
            if known_in_target.sum() < PATCH_SIZE:
                mean_val = np.mean(working_img[working_mask == 0], axis=0) if (working_mask == 0).sum() > 0 else [128, 128, 128]
                working_img[ty, tx] = mean_val
                working_mask[ty, tx] = 0
                confidence[ty, tx] = 0.1
                iteration += 1
                continue
            
            # isi bagian kosong dengan rata-rata untuk matching
            if np.any(target_patch_mask > 0):
                mean_color = np.mean(target_patch[known_in_target], axis=0)
                for c in range(3):
                    target_patch[:, :, c] = np.where(target_patch_mask > 0, mean_color[c], target_patch[:, :, c])
            
            # cari di seluruh gambar
            best_match = None
            
            try:
                result = cv2.matchTemplate(working_img.astype(np.float32),
                                          target_patch.astype(np.float32),
                                          cv2.TM_SQDIFF)
                
                result_h, result_w = result.shape
                
                # tandai posisi yang overlap dengan mask
                for ry in range(result_h):
                    for rx in range(result_w):
                        patch_mask = working_mask[ry:ry+PATCH_SIZE, rx:rx+PATCH_SIZE]
                        if np.any(patch_mask > 0):
                            result[ry, rx] = float('inf')
                        # hindari self-match
                        if abs(ry + half - ty) < PATCH_SIZE and abs(rx + half - tx) < PATCH_SIZE:
                            result[ry, rx] = float('inf')
                
                if np.min(result) < float('inf'):
                    min_loc = np.unravel_index(np.argmin(result), result.shape)
                    best_match = (min_loc[0] + half, min_loc[1] + half)
            except Exception as e:
                pass
            
            # LANGKAH 4: salin patch ke target
            if best_match is not None:
                sy, sx = best_match
                if (sy - half >= 0 and sy + half + 1 <= h and 
                    sx - half >= 0 and sx + half + 1 <= w):
                    
                    for py_off in range(-half, half + 1):
                        for px_off in range(-half, half + 1):
                            tpy, tpx = ty + py_off, tx + px_off
                            spy, spx = sy + py_off, sx + px_off
                            
                            if 0 <= tpy < h and 0 <= tpx < w and 0 <= spy < h and 0 <= spx < w:
                                if working_mask[tpy, tpx] > 0:
                                    working_img[tpy, tpx] = working_img[spy, spx]
                                    working_mask[tpy, tpx] = 0
                                    confidence[tpy, tpx] = confidence[spy, spx] * confidence[ty, tx]
            else:
                # fallback: isi dengan rata-rata lokal
                local_region = working_img[max(0,ty-20):min(h,ty+20), max(0,tx-20):min(w,tx+20)]
                local_mask = working_mask[max(0,ty-20):min(h,ty+20), max(0,tx-20):min(w,tx+20)]
                known_local = local_region[local_mask == 0]
                if len(known_local) > 0:
                    working_img[ty, tx] = np.mean(known_local, axis=0)
                working_mask[ty, tx] = 0
                confidence[ty, tx] = 0.1
            
            iteration += 1
            
            # update progress tiap 500ms
            current_time = time.time()
            if current_time - last_update_time >= 0.5:
                remaining = working_mask.sum()
                percent = ((initial_pixels - remaining) / initial_pixels) * 100
                elapsed = current_time - start_time
                print(f"[STD-CRIMINISI] Iter {iteration}: {remaining} left ({percent:.1f}%) - {elapsed:.1f}s elapsed")
                
                if self.progress_callback:
                    self.progress_callback(np.clip(working_img, 0, 255).astype(np.uint8), 
                                          iteration, remaining, percent)
                last_update_time = current_time
        
        total_time = time.time() - start_time
        print(f"[STD-CRIMINISI] ✅ Complete in {total_time:.1f}s ({iteration} iterations)")
        
        return np.clip(working_img, 0, 255).astype(np.uint8)
    
    def _analyze_boundary_texture(self, roi_img: np.ndarray, roi_mask: np.ndarray) -> Tuple[float, float]:
        """Analisis tekstur di tepi mask untuk menentukan kompleksitas."""
        # konversi ke grayscale
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img.copy()
        
        # buat mask tepi dengan dilasi
        kernel = np.ones((5, 5), np.uint8)
        mask_binary = (roi_mask > 127).astype(np.uint8)
        
        dilated = cv2.dilate(mask_binary, kernel, iterations=2)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        
        boundary = dilated - mask_binary
        boundary_pixels = gray[boundary > 0]
        
        if len(boundary_pixels) < 10:
            # fallback: analisis semua piksel yang diketahui
            known_pixels = gray[mask_binary == 0]
            if len(known_pixels) < 10:
                return 128.0, 100.0
            boundary_pixels = known_pixels
        
        mean_val = float(np.mean(boundary_pixels))
        std_val = float(np.std(boundary_pixels))
        
        return mean_val, std_val
    
    def _calculate_local_variance(self, image: np.ndarray, mask: np.ndarray, 
                                  target_pixel: Tuple[int, int]) -> float:
        """Hitung variance lokal untuk menentukan ukuran patch."""
        ty, tx = target_pixel
        
        # ambil area sekitar piksel target
        window_size = 11
        half = window_size // 2
        
        y1 = max(0, ty - half)
        y2 = min(image.shape[0], ty + half + 1)
        x1 = max(0, tx - half)
        x2 = min(image.shape[1], tx + half + 1)
        
        neighborhood = image[y1:y2, x1:x2]
        neighborhood_mask = mask[y1:y2, x1:x2]
        
        # ambil piksel yang sudah diketahui
        known_mask = (neighborhood_mask == 0)
        
        if known_mask.sum() < 5:
            return 25.0
        
        known_pixels = neighborhood[known_mask]
        
        # hitung variance dari intensitas piksel
        pixel_intensities = np.mean(known_pixels, axis=1)
        variance = np.std(pixel_intensities)
        
        return variance
    
    def _inpaint_small(self, image: np.ndarray, mask: np.ndarray,
                      bbox: Tuple, full_image: np.ndarray,
                      original_roi_shape: Optional[Tuple[int, int]] = None,
                      use_adaptive: bool = True) -> np.ndarray:
        """Proses inpainting utama dengan Criminisi adaptif."""
        target_region = (mask > 0).astype(np.uint8)
        
        if target_region.sum() == 0:
            return image.copy()
        
        working_img = image.astype(np.float32)
        working_mask = target_region.copy()
        confidence = (1.0 - target_region).astype(np.float32)
        
        # hitung gradien untuk prioritas
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        max_iterations = 3000
        iteration = 0
        initial_mask_sum = working_mask.sum()
        
        print(f"[INPAINTER] Starting iterations... Target pixels: {initial_mask_sum}")
        
        if use_adaptive:
            print(f"[INPAINTER] Using ADAPTIVE patch sizing (7x7 to 11x11, prioritizing detail)")
        else:
            print(f"[INPAINTER] Using FIXED patch sizing (9x9) - Standard Criminisi Mode")
            self.patch_size = 9  # Lock to fixed size
        
        last_update_time = time.time()
        update_interval = 0.25
        
        # watchdog untuk deteksi stagnasi
        last_pixels_remaining = initial_mask_sum
        stagnation_counter = 0
        max_stagnation = 10
        
        while working_mask.sum() > 0 and iteration < max_iterations:
            current_pixels_remaining = working_mask.sum()
            fill_front = self._get_fill_front(working_mask)
            
            if len(fill_front) == 0:
                print(f"[INPAINTER] Fill front empty at iteration {iteration}")
                break
            
            priorities = self._compute_priorities(fill_front, confidence, grad_x, grad_y, working_mask)
            
            # pilih piksel dengan prioritas tertinggi
            if np.max(priorities) > 1e-6:
                target_pixel = fill_front[np.argmax(priorities)]
            else:
                target_pixel = fill_front[np.random.randint(0, len(fill_front))]
            
            # tentukan ukuran patch berdasarkan variance lokal
            if use_adaptive:
                local_variance = self._calculate_local_variance(working_img, working_mask, target_pixel)
                
                if local_variance < 10:
                    # area polos (speech bubble putih)
                    self.patch_size = 11
                elif local_variance > 30:
                    # area ramai (SFX, screentone)
                    self.patch_size = 7
                else:
                    # area sedang
                    self.patch_size = 9
            
            best_match = self._find_best_match(working_img, working_mask, target_pixel)
            
            if best_match is None:
                self._fill_simple(working_img, working_mask, confidence, target_pixel)
            else:
                pixels_filled = self._copy_patch(working_img, working_mask, confidence, 
                                                target_pixel, best_match)
            
            iteration += 1
            
            # cek stagnasi
            if current_pixels_remaining >= last_pixels_remaining:
                stagnation_counter += 1
                
                if stagnation_counter >= max_stagnation:
                    print(f"\nSTAGNATION DETECTED")
                    print(f"[WATCHDOG] No progress for {stagnation_counter} iterations")
                    print(f"[WATCHDOG] Remaining: {current_pixels_remaining} pixels")
                    print(f"[WATCHDOG] Forcing random fill to break deadlock...")
                    
                    self._force_fill_random(working_img, working_mask, confidence)
                    stagnation_counter = 0
            else:
                stagnation_counter = 0
            
            last_pixels_remaining = current_pixels_remaining
            
            # update progress
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                remaining = working_mask.sum()
                percent = ((initial_mask_sum - remaining) / initial_mask_sum) * 100
                
                if self.progress_callback and bbox and full_image is not None:
                    preview = full_image.copy()
                    y1, y2, x1, x2 = bbox
                    current_result = np.clip(working_img, 0, 255).astype(np.uint8)
                    
                    # upscale jika dalam mode piramida
                    if original_roi_shape is not None:
                        orig_h, orig_w = original_roi_shape
                        current_h, current_w = current_result.shape[:2]
                        
                        if (orig_h, orig_w) != (current_h, current_w):
                            current_result = cv2.resize(current_result, (orig_w, orig_h), 
                                                       interpolation=cv2.INTER_LINEAR)
                    
                    preview[y1:y2, x1:x2] = current_result
                    self.progress_callback(preview, iteration, remaining, percent)
                
                last_update_time = current_time
            
            if iteration % 50 == 0:
                remaining = working_mask.sum()
                percent = ((initial_mask_sum - remaining) / initial_mask_sum) * 100
                print(f"Iteration {iteration}: {remaining} pixels remaining ({percent:.1f}% complete)")
        
        if iteration >= max_iterations:
            print(f"[INPAINTER] ⚠️  Reached max iterations ({max_iterations})")
        
        return np.clip(working_img, 0, 255).astype(np.uint8)
    
    def _get_fill_front(self, mask: np.ndarray) -> np.ndarray:
        """Ambil piksel tepi dari area target."""
        if mask.sum() == 0:
            return np.array([])
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        boundary = dilated - mask
        
        front_coords = np.argwhere((boundary > 0) & (mask > 0))
        
        if len(front_coords) == 0:
            front_coords = np.argwhere(mask > 0)
        
        return front_coords
    
    def _compute_priorities(self, fill_front: np.ndarray, confidence: np.ndarray,
                           grad_x: np.ndarray, grad_y: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
        """Hitung prioritas untuk setiap piksel tepi."""
        priorities = np.zeros(len(fill_front))
        
        for i, (y, x) in enumerate(fill_front):
            half = 4
            y1, y2 = max(0, y - half), min(confidence.shape[0], y + half + 1)
            x1, x2 = max(0, x - half), min(confidence.shape[1], x + half + 1)
            C_p = np.mean(confidence[y1:y2, x1:x2])
            
            if 0 <= y < grad_x.shape[0] and 0 <= x < grad_x.shape[1]:
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                D_p = np.sqrt(gx**2 + gy**2) / 255.0
            else:
                D_p = 0.0
            
            priorities[i] = C_p * (D_p + 0.001)
        
        return priorities
    
    def _find_best_match(self, image: np.ndarray, mask: np.ndarray,
                        target_pixel: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Cari patch terbaik dengan SSD + penalty jarak + penalty variance."""
        ty, tx = target_pixel
        half = self.patch_size // 2
        
        ty1, ty2 = ty - half, ty + half + 1
        tx1, tx2 = tx - half, tx + half + 1
        
        if ty1 < 0 or tx1 < 0 or ty2 > image.shape[0] or tx2 > image.shape[1]:
            return None
        
        target_patch = image[ty1:ty2, tx1:tx2].copy()
        target_mask = mask[ty1:ty2, tx1:tx2]
        
        known_mask = (target_mask == 0)
        if known_mask.sum() < self.patch_size:
            return None
        
        # hitung variance patch target
        target_known_pixels = target_patch[known_mask]
        target_gray = np.mean(target_known_pixels, axis=1)
        target_variance = np.std(target_gray)
        
        # isi area mask dengan rata-rata untuk matching
        if np.any(target_mask > 0):
            mean_color = np.mean(target_patch[known_mask], axis=0)
            for c in range(3):
                target_patch[:, :, c] = np.where(target_mask > 0, 
                                                 mean_color[c], 
                                                 target_patch[:, :, c])
        
        # buat area pencarian yang valid
        h, w = image.shape[:2]
        kernel = np.ones((self.patch_size, self.patch_size), np.uint8)
        invalid = cv2.dilate(mask, kernel, iterations=1)
        valid_search = (invalid == 0).astype(np.uint8)
        
        # batasi area pencarian untuk kecepatan
        search_margin = min(max(30, self.patch_size * 3), min(h, w) // 2)
        sy1 = max(0, ty - search_margin)
        sy2 = min(h, ty + search_margin)
        sx1 = max(0, tx - search_margin)
        sx2 = min(w, tx + search_margin)
        
        search_img = image[sy1:sy2, sx1:sx2]
        search_valid = valid_search[sy1:sy2, sx1:sx2]
        
        if search_img.shape[0] < self.patch_size or search_img.shape[1] < self.patch_size:
            return None
        
        try:
            result = cv2.matchTemplate(search_img.astype(np.float32),
                                      target_patch.astype(np.float32),
                                      cv2.TM_SQDIFF)
            
            # penalty jarak (prioritaskan patch terdekat)
            result_h, result_w = result.shape
            yy, xx = np.meshgrid(range(result_h), range(result_w), indexing='ij')
            target_local_y = ty - sy1
            target_local_x = tx - sx1
            
            distances = np.sqrt((yy - target_local_y)**2 + (xx - target_local_x)**2)
            max_distance = np.sqrt(result_h**2 + result_w**2)
            if max_distance > 0:
                normalized_distances = distances / max_distance
            else:
                normalized_distances = np.zeros_like(distances)
            
            # hitung faktor penalty dari range hasil
            valid_results = result[result < float('inf')]
            if len(valid_results) > 0:
                result_range = np.max(valid_results) - np.min(valid_results)
                distance_penalty_factor = 0.20 * result_range
            else:
                distance_penalty_factor = 100.0
            
            distance_penalty = normalized_distances * distance_penalty_factor
            
            # penalty variance (konsistensi tekstur)
            VARIANCE_WEIGHT = 100.0
            final_error = result + distance_penalty
            
            # tandai area invalid
            valid_result = cv2.matchTemplate(search_valid.astype(np.float32),
                                            np.ones((self.patch_size, self.patch_size), np.float32),
                                            cv2.TM_SQDIFF)
            
            final_error[valid_result > 0.1] = float('inf')
            
            # hindari self-match
            self_margin = half + 3
            self_y1 = max(0, target_local_y - self_margin)
            self_y2 = min(result_h, target_local_y + self_margin)
            self_x1 = max(0, target_local_x - self_margin)
            self_x2 = min(result_w, target_local_x + self_margin)
            
            if self_y2 > self_y1 and self_x2 > self_x1:
                final_error[self_y1:self_y2, self_x1:self_x2] = float('inf')
            
            # hitung variance penalty untuk setiap posisi valid
            for res_y in range(result_h):
                for res_x in range(result_w):
                    if final_error[res_y, res_x] == float('inf'):
                        continue
                    src_y1 = sy1 + res_y
                    src_y2 = sy1 + res_y + self.patch_size
                    src_x1 = sx1 + res_x
                    src_x2 = sx1 + res_x + self.patch_size
                    
                    if src_y2 <= h and src_x2 <= w:
                        patch_region_mask = mask[src_y1:src_y2, src_x1:src_x2]
                        if np.any(patch_region_mask > 0):
                            final_error[res_y, res_x] = float('inf')
                            continue
                        
                        # tambahkan penalty variance
                        src_patch = image[src_y1:src_y2, src_x1:src_x2]
                        src_gray = np.mean(src_patch.reshape(-1, 3), axis=1)
                        src_variance = np.std(src_gray)
                        variance_diff = abs(target_variance - src_variance)
                        final_error[res_y, res_x] += variance_diff * VARIANCE_WEIGHT
                    else:
                        final_error[res_y, res_x] = float('inf')
            
            # ambil posisi terbaik
            if np.min(final_error) == float('inf'):
                return None
            
            min_loc = np.unravel_index(np.argmin(final_error), final_error.shape)
            best_y, best_x = min_loc
            global_y = sy1 + best_y + half
            global_x = sx1 + best_x + half
            
            if (global_y - half >= 0 and global_y + half + 1 <= h and
                global_x - half >= 0 and global_x + half + 1 <= w):
                return (global_y, global_x)
            
            return None
            
        except Exception as e:
            print(f"[INPAINTER] Match error: {e}")
            return None
    
    def _copy_patch(self, image: np.ndarray, mask: np.ndarray,
                   confidence: np.ndarray, target: Tuple[int, int],
                   source: Tuple[int, int]) -> int:
        """Salin patch dari source ke target."""
        ty, tx = target
        sy, sx = source
        half = self.patch_size // 2
        
        ty1, ty2 = ty - half, ty + half + 1
        tx1, tx2 = tx - half, tx + half + 1
        sy1, sy2 = sy - half, sy + half + 1
        sx1, sx2 = sx - half, sx + half + 1
        
        if (ty1 < 0 or tx1 < 0 or ty2 > image.shape[0] or tx2 > image.shape[1] or
            sy1 < 0 or sx1 < 0 or sy2 > image.shape[0] or sx2 > image.shape[1]):
            return
        
        target_mask_patch = mask[ty1:ty2, tx1:tx2]
        source_patch = image[sy1:sy2, sx1:sx2]
        
        if source_patch.shape != target_mask_patch.shape[:2] + (3,):
            return
        
        conf_value = np.mean(confidence[ty1:ty2, tx1:tx2][target_mask_patch == 0])
        if np.isnan(conf_value):
            conf_value = 0.5
        
        copy_mask = target_mask_patch > 0
        pixels_filled = np.count_nonzero(copy_mask)
        
        if pixels_filled > 0:
            image[ty1:ty2, tx1:tx2][copy_mask] = source_patch[copy_mask]
            mask[ty1:ty2, tx1:tx2][copy_mask] = 0
            confidence[ty1:ty2, tx1:tx2][copy_mask] = conf_value
        
        return pixels_filled
    
    def _fill_simple(self, image: np.ndarray, mask: np.ndarray,
                    confidence: np.ndarray, pixel: Tuple[int, int]):
        """Isi sederhana dengan rata-rata tetangga."""
        y, x = pixel
        
        y1, y2 = max(0, y - 1), min(image.shape[0], y + 2)
        x1, x2 = max(0, x - 1), min(image.shape[1], x + 2)
        
        neighborhood = image[y1:y2, x1:x2]
        neighborhood_mask = mask[y1:y2, x1:x2]
        
        known = neighborhood[neighborhood_mask == 0]
        if len(known) > 0:
            image[y, x] = np.mean(known, axis=0)
            mask[y, x] = 0
            confidence[y, x] = 0.5
    
    def _force_fill_random(self, image: np.ndarray, mask: np.ndarray,
                          confidence: np.ndarray, num_pixels: int = 50):
        """Paksa isi piksel random untuk mengatasi stagnasi."""
        masked_coords = np.argwhere(mask > 0)
        
        if len(masked_coords) == 0:
            return
        
        num_to_fill = min(num_pixels, len(masked_coords))
        random_indices = np.random.choice(len(masked_coords), num_to_fill, replace=False)
        
        for idx in random_indices:
            y, x = masked_coords[idx]
            
            search_radius = 5
            y1 = max(0, y - search_radius)
            y2 = min(image.shape[0], y + search_radius + 1)
            x1 = max(0, x - search_radius)
            x2 = min(image.shape[1], x + search_radius + 1)
            
            neighborhood = image[y1:y2, x1:x2]
            neighborhood_mask = mask[y1:y2, x1:x2]
            
            known = neighborhood[neighborhood_mask == 0]
            if len(known) > 0:
                image[y, x] = np.mean(known, axis=0)
                mask[y, x] = 0
                confidence[y, x] = 0.3  # Low confidence for forced fill
        
        print(f"[WATCHDOG] Force-filled {num_to_fill} pixels")
