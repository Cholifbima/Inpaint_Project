"""
HybridInpainter: Adaptive Pyramid-Based Criminisi Inpainting
- Pyramid processing (450px) for detail preservation on complex SFX
- Texture-aware adaptive patch sizing for quality
- Small patches (7x7-9x9) prioritized for complex textures/speed lines
- Moderate patches (11x11) only for very flat areas
- Direct paste (no blending filters) for sharp, clean results
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
import time


class HybridInpainter:
    """
    Adaptive Pyramid-Based Inpainting with Texture-Aware Patch Sizing.
    
    Key Features:
    - Pyramid processing (450px): Downscale → Process → Upscale (preserves SFX details!)
    - Adaptive patch sizing based on local variance (TUNED FOR COMPLEX TEXTURES):
      * Very low variance (<10): Moderate patch (11x11) for clean fills
      * High variance (>30): Small patch (7x7) for maximum detail preservation
      * Medium (10-30): Small-medium patch (9x9) favoring detail
    - Direct paste (no blur filters) for sharp, artifact-free results
    - Distance-weighted matching to prevent texture bleeding
    
    Result: Sharp, detailed results on complex backgrounds like speed lines!
    """
    
    def __init__(self, target_proc_size: int = 450, patch_size: int = 9):
        """
        Initialize the adaptive pyramid-based inpainter.
        
        Args:
            target_proc_size: Target height/width for processing (default 450px for detail preservation)
            patch_size: Base patch size (will be adapted based on variance)
        """
        self.target_proc_size = target_proc_size
        self.base_patch_size = patch_size  # Base size, will be adapted
        self.patch_size = patch_size  # Current patch size (dynamic)
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.is_running = False
        self.total_pixels = 0
        self.filled_pixels = 0
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, 
                padding: int = 20, progress_callback: Optional[Callable] = None,
                method: str = "adaptive") -> np.ndarray:
        """
        Perform inpainting using the specified method.
        
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W), 255=inpaint, 0=keep
            padding: Padding around ROI bounding box
            progress_callback: Callback for progress updates
            method: Inpainting method to use:
                - "telea": OpenCV Telea (fast baseline)
                - "criminisi_standard": Standard Criminisi on full resolution (slow baseline)
                - "adaptive": Adaptive Hybrid Pyramid (proposed method, default)
            
        Returns:
            Inpainted image
        """
        print(f"\n[INPAINTER] 🎨 INPAINTING (method={method})")
        print(f"[INPAINTER] Input image shape: {image.shape}")
        print(f"[INPAINTER] Input mask shape: {mask.shape}")
        print(f"[INPAINTER] Mask pixel count: {np.count_nonzero(mask)}")
        
        self.progress_callback = progress_callback
        self.is_running = True
        
        # Verify mask
        if np.count_nonzero(mask) == 0:
            print("[INPAINTER] WARNING: Mask is empty! Returning original.")
            self.is_running = False
            return image.copy()
        
        # ===========================================
        # METHOD A: OpenCV Telea (Baseline 1)
        # ===========================================
        if method == "telea":
            print("[INPAINTER] Using OpenCV Telea (cv2.INPAINT_TELEA)")
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            result = cv2.inpaint(image, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            print("[INPAINTER] ✅ Telea inpainting complete!")
            self.is_running = False
            return result
        
        # ===========================================
        # METHOD A2: OpenCV Navier-Stokes (Baseline 2)
        # ===========================================
        if method == "ns":
            print("[INPAINTER] Using OpenCV Navier-Stokes (cv2.INPAINT_NS)")
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            result = cv2.inpaint(image, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_NS)
            print("[INPAINTER] ✅ Navier-Stokes inpainting complete!")
            self.is_running = False
            return result
        
        # ===========================================
        # METHOD B: Standard Criminisi (Baseline 3)
        # Full resolution, no pyramid, no smart switch
        # ===========================================
        if method == "criminisi_standard":
            print("[INPAINTER] Using Standard Criminisi (full resolution, no pyramid)")
            print("[INPAINTER] ⚠️  WARNING: This will be SLOW on large masks!")
            
            # Extract ROI but process at FULL resolution
            roi_data = self._extract_roi(image, mask, padding)
            if roi_data is None:
                print("[INPAINTER] WARNING: Could not extract ROI!")
                self.is_running = False
                return image.copy()
            
            roi_img, roi_mask, bbox = roi_data
            y1, y2, x1, x2 = bbox
            
            print(f"[INPAINTER] ROI: {roi_img.shape[1]}×{roi_img.shape[0]} (FULL RESOLUTION)")
            print(f"[INPAINTER] Processing {np.count_nonzero(roi_mask)} pixels...")
            
            # Process at FULL resolution (no downscaling)
            result_roi = self._inpaint_small(roi_img, roi_mask, bbox, image, 
                                            original_roi_shape=None)
            
            # Paste back
            result = image.copy()
            result[y1:y2, x1:x2] = result_roi
            
            print("[INPAINTER] ✅ Standard Criminisi complete!")
            self.is_running = False
            return result
        
        # ===========================================
        # METHOD C: Adaptive Hybrid Pyramid (Proposed)
        # ===========================================
        print("[INPAINTER] Using Adaptive Hybrid Pyramid (proposed method)")
        
        # Extract ROI
        roi_data = self._extract_roi(image, mask, padding)
        if roi_data is None:
            print("[INPAINTER] WARNING: Could not extract ROI!")
            self.is_running = False
            return image.copy()
        
        roi_img, roi_mask, bbox = roi_data
        y1, y2, x1, x2 = bbox
        
        print(f"[INPAINTER] ROI extracted: {roi_img.shape[1]}×{roi_img.shape[0]}")
        print(f"[INPAINTER] ROI mask pixels: {np.count_nonzero(roi_mask)}")
        
        # Pure Adaptive Pyramid Criminisi (no smart switch)
        orig_h, orig_w = roi_img.shape[:2]
        
        # Determine if we need to downscale
        max_dim = max(orig_h, orig_w)
        
        if max_dim > self.target_proc_size * 1.3:  # More than 30% larger
            # PYRAMID MODE: Downscale → Process → Upscale
            print(f"\n[PYRAMID MODE] ⚡")
            print(f"[PYRAMID] Original: {orig_w}×{orig_h}")
            
            # Calculate scale factor (maintain aspect ratio)
            if orig_h > orig_w:
                scale_factor = self.target_proc_size / orig_h
            else:
                scale_factor = self.target_proc_size / orig_w
            
            new_w = max(50, int(orig_w * scale_factor))
            new_h = max(50, int(orig_h * scale_factor))
            
            print(f"[PYRAMID] Downscaling to: {new_w}×{new_h} (scale: {scale_factor:.3f})")
            print(f"[PYRAMID] Speed boost: ~{1/(scale_factor**2):.1f}x faster!")
            
            # Downscale
            small_img = cv2.resize(roi_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            small_mask = (small_mask > 127).astype(np.uint8) * 255  # Re-binarize
            
            print(f"[PYRAMID] Processing {np.count_nonzero(small_mask) // 255} pixels at small scale...")
            
            # Process at small scale (pass original shape for preview upscaling)
            result_small = self._inpaint_small(small_img, small_mask, bbox, image, 
                                              original_roi_shape=(orig_h, orig_w))
            
            # Upscale result (CUBIC for sharp, fast quality)
            print(f"[PYRAMID] Upscaling back to {orig_w}×{orig_h} (CUBIC interpolation)...")
            result_roi = cv2.resize(result_small, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
            
            print(f"[PYRAMID] ✅ Pyramid processing complete!")
        else:
            # Direct processing (ROI already small)
            print(f"[INPAINTER] ROI small enough, processing directly...")
            result_roi = self._inpaint_small(roi_img, roi_mask, bbox, image, 
                                            original_roi_shape=None)  # Not downscaled
        
        print(f"[INPAINTER] Pasting result back to original image...")
        
        # Direct paste (no seamless clone - preserves sharpness and detail)
        result = image.copy()
        result[y1:y2, x1:x2] = result_roi
        
        print(f"[INPAINTER] ✅ Inpainting complete!")
        self.is_running = False
        
        return result
    
    def _extract_roi(self, image: np.ndarray, mask: np.ndarray, 
                     padding: int) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple]]:
        """Extract Region of Interest around the mask."""
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
    
    def _analyze_boundary_texture(self, roi_img: np.ndarray, roi_mask: np.ndarray) -> Tuple[float, float]:
        """
        Analyze the texture at the boundary of the mask region.
        
        This helps the Smart Switch decide between Telea (simple) and Criminisi (complex).
        
        Args:
            roi_img: ROI image region (BGR)
            roi_mask: ROI mask (255=inpaint area, 0=known)
            
        Returns:
            (mean_brightness, std_deviation) of boundary pixels
        """
        # Convert to grayscale for analysis
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img.copy()
        
        # Create boundary mask by dilating and subtracting
        kernel = np.ones((5, 5), np.uint8)
        mask_binary = (roi_mask > 127).astype(np.uint8)
        
        # Dilate mask to get outer boundary
        dilated = cv2.dilate(mask_binary, kernel, iterations=2)
        # Erode mask to get inner boundary  
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        
        # Boundary = pixels just outside the mask (known pixels adjacent to unknown)
        boundary = dilated - mask_binary
        
        # Get pixel values at boundary
        boundary_pixels = gray[boundary > 0]
        
        if len(boundary_pixels) < 10:
            # Fallback: analyze the entire known region
            known_pixels = gray[mask_binary == 0]
            if len(known_pixels) < 10:
                return 128.0, 100.0  # Default to complex
            boundary_pixels = known_pixels
        
        mean_val = float(np.mean(boundary_pixels))
        std_val = float(np.std(boundary_pixels))
        
        return mean_val, std_val
    
    def _calculate_local_variance(self, image: np.ndarray, mask: np.ndarray, 
                                  target_pixel: Tuple[int, int]) -> float:
        """
        Calculate variance of known pixels surrounding the target pixel.
        
        This helps determine texture complexity:
        - Low variance (<15): Flat area (white bubble)
        - High variance (>40): Busy texture (screentones, hair)
        - Medium: Moderate detail
        
        Args:
            image: Working image
            mask: Working mask (1=unknown, 0=known)
            target_pixel: (y, x) position
            
        Returns:
            Standard deviation of surrounding known pixels
        """
        ty, tx = target_pixel
        
        # Extract neighborhood (e.g., 11x11 region around target)
        window_size = 11
        half = window_size // 2
        
        y1 = max(0, ty - half)
        y2 = min(image.shape[0], ty + half + 1)
        x1 = max(0, tx - half)
        x2 = min(image.shape[1], tx + half + 1)
        
        neighborhood = image[y1:y2, x1:x2]
        neighborhood_mask = mask[y1:y2, x1:x2]
        
        # Get known pixels in neighborhood
        known_mask = (neighborhood_mask == 0)
        
        if known_mask.sum() < 5:
            # Not enough known pixels, return medium variance
            return 25.0
        
        known_pixels = neighborhood[known_mask]
        
        # Calculate variance (use grayscale for simplicity)
        # Average across BGR channels
        pixel_intensities = np.mean(known_pixels, axis=1)
        variance = np.std(pixel_intensities)
        
        return variance
    
    def _inpaint_small(self, image: np.ndarray, mask: np.ndarray,
                      bbox: Tuple, full_image: np.ndarray,
                      original_roi_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Core inpainting on (possibly downscaled) image.
        Uses fixed patch size for simplicity and speed.
        
        Args:
            image: Image to inpaint (may be downscaled)
            mask: Mask for inpainting (same size as image)
            bbox: Bounding box in full image coordinates
            full_image: Full resolution original image
            original_roi_shape: Original ROI shape (h, w) before downscaling, None if not downscaled
        """
        # Convert to binary
        target_region = (mask > 0).astype(np.uint8)
        
        if target_region.sum() == 0:
            return image.copy()
        
        # Working copies
        working_img = image.astype(np.float32)
        working_mask = target_region.copy()
        
        # Initialize confidence
        confidence = (1.0 - target_region).astype(np.float32)
        
        # Compute gradients for priority
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Main loop
        max_iterations = 3000
        iteration = 0
        initial_mask_sum = working_mask.sum()
        
        print(f"[INPAINTER] Starting iterations... Target pixels: {initial_mask_sum}")
        print(f"[INPAINTER] Using ADAPTIVE patch sizing (7x7 to 11x11, prioritizing detail)")
        
        last_update_time = time.time()
        update_interval = 0.25  # Update every 250ms (every ~50 iterations)
        
        # Stagnation watchdog (detect infinite loops)
        last_pixels_remaining = initial_mask_sum
        stagnation_counter = 0
        max_stagnation = 10  # Allow 10 iterations without progress before intervention
        
        while working_mask.sum() > 0 and iteration < max_iterations:
            current_pixels_remaining = working_mask.sum()
            # Find fill front
            fill_front = self._get_fill_front(working_mask)
            
            if len(fill_front) == 0:
                print(f"[INPAINTER] Fill front empty at iteration {iteration}")
                break
            
            # Compute priorities
            priorities = self._compute_priorities(fill_front, confidence, grad_x, grad_y, working_mask)
            
            # Select best pixel
            if np.max(priorities) > 1e-6:
                target_pixel = fill_front[np.argmax(priorities)]
            else:
                # Random fallback
                target_pixel = fill_front[np.random.randint(0, len(fill_front))]
            
            # ==========================================
            # ADAPTIVE PATCH SIZING (Texture-Aware)
            # TUNED FOR COMPLEX SFX/SPEED LINES
            # ==========================================
            local_variance = self._calculate_local_variance(working_img, working_mask, target_pixel)
            
            # Decide patch size based on variance (AGGRESSIVE FOR DETAIL)
            if local_variance < 10:
                # VERY LOW VARIANCE: Flat/smooth area (pure white bubble)
                # Use MODERATE patch (not too large to avoid over-smoothing)
                self.patch_size = 11
            elif local_variance > 30:
                # HIGH VARIANCE: Busy texture (SFX, speed lines, screentones)
                # Use SMALL patch for maximum detail preservation
                self.patch_size = 7
            else:
                # MEDIUM VARIANCE: Balanced
                # Use SMALL-MEDIUM patch to favor detail
                self.patch_size = 9
            
            # Find best match with adaptive patch size
            best_match = self._find_best_match(working_img, working_mask, target_pixel)
            
            if best_match is None:
                # Simple fill fallback
                self._fill_simple(working_img, working_mask, confidence, target_pixel)
            else:
                # Copy patch
                pixels_filled = self._copy_patch(working_img, working_mask, confidence, 
                                                target_pixel, best_match)
            
            iteration += 1
            
            # STAGNATION WATCHDOG: Check if we're making progress
            if current_pixels_remaining >= last_pixels_remaining:
                stagnation_counter += 1
                
                if stagnation_counter >= max_stagnation:
                    print(f"\n[WATCHDOG] ⚠️  STAGNATION DETECTED!")
                    print(f"[WATCHDOG] No progress for {stagnation_counter} iterations")
                    print(f"[WATCHDOG] Remaining: {current_pixels_remaining} pixels")
                    print(f"[WATCHDOG] Forcing random fill to break deadlock...")
                    
                    # Force fill with random known neighbors
                    self._force_fill_random(working_img, working_mask, confidence)
                    stagnation_counter = 0  # Reset after intervention
            else:
                # Progress made, reset counter
                stagnation_counter = 0
            
            last_pixels_remaining = current_pixels_remaining
            
            # Progress updates
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                remaining = working_mask.sum()
                percent = ((initial_mask_sum - remaining) / initial_mask_sum) * 100
                
                if self.progress_callback and bbox and full_image is not None:
                    # Create preview
                    preview = full_image.copy()
                    y1, y2, x1, x2 = bbox
                    
                    # FIX: Handle downscaled vs native resolution
                    current_result = np.clip(working_img, 0, 255).astype(np.uint8)
                    
                    if original_roi_shape is not None:
                        # We're in downscaled mode - need to upscale for preview
                        orig_h, orig_w = original_roi_shape
                        current_h, current_w = current_result.shape[:2]
                        
                        if (orig_h, orig_w) != (current_h, current_w):
                            # Upscale working image to original ROI size for preview
                            current_result = cv2.resize(current_result, (orig_w, orig_h), 
                                                       interpolation=cv2.INTER_LINEAR)
                    
                    # Now dimensions should match
                    preview[y1:y2, x1:x2] = current_result
                    self.progress_callback(preview, iteration, remaining, percent)
                
                last_update_time = current_time
            
            # Console progress
            if iteration % 50 == 0:
                remaining = working_mask.sum()
                percent = ((initial_mask_sum - remaining) / initial_mask_sum) * 100
                print(f"Iteration {iteration}: {remaining} pixels remaining ({percent:.1f}% complete)")
        
        # Final check
        if iteration >= max_iterations:
            print(f"[INPAINTER] ⚠️  Reached max iterations ({max_iterations})")
        
        return np.clip(working_img, 0, 255).astype(np.uint8)
    
    def _get_fill_front(self, mask: np.ndarray) -> np.ndarray:
        """Get boundary pixels of target region."""
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
        """Compute priority for each fill front pixel."""
        priorities = np.zeros(len(fill_front))
        
        for i, (y, x) in enumerate(fill_front):
            # Confidence term
            half = 4
            y1, y2 = max(0, y - half), min(confidence.shape[0], y + half + 1)
            x1, x2 = max(0, x - half), min(confidence.shape[1], x + half + 1)
            C_p = np.mean(confidence[y1:y2, x1:x2])
            
            # Data term
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
        """
        Find best matching patch using cv2.matchTemplate.
        Uses standard SSD + distance penalty (clean and fast).
        """
        ty, tx = target_pixel
        half = self.patch_size // 2
        
        # Extract target patch
        ty1, ty2 = ty - half, ty + half + 1
        tx1, tx2 = tx - half, tx + half + 1
        
        if ty1 < 0 or tx1 < 0 or ty2 > image.shape[0] or tx2 > image.shape[1]:
            return None
        
        target_patch = image[ty1:ty2, tx1:tx2].copy()
        target_mask = mask[ty1:ty2, tx1:tx2]
        
        # Need some known pixels
        known_mask = (target_mask == 0)
        if known_mask.sum() < self.patch_size:
            return None
        
        # Fill masked region with mean for template matching
        if np.any(target_mask > 0):
            mean_color = np.mean(target_patch[known_mask], axis=0)
            for c in range(3):
                target_patch[:, :, c] = np.where(target_mask > 0, 
                                                 mean_color[c], 
                                                 target_patch[:, :, c])
        
        # Create valid search area (no masked regions)
        h, w = image.shape[:2]
        kernel = np.ones((self.patch_size, self.patch_size), np.uint8)
        invalid = cv2.dilate(mask, kernel, iterations=1)
        valid_search = (invalid == 0).astype(np.uint8)
        
        # Search area (limited for speed)
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
            # Template matching
            result = cv2.matchTemplate(search_img.astype(np.float32),
                                      target_patch.astype(np.float32),
                                      cv2.TM_SQDIFF)
            
            # DISTANCE-WEIGHTED MATCHING (prioritize nearby patches)
            result_h, result_w = result.shape
            
            # Create 2D grid of coordinates
            yy, xx = np.meshgrid(range(result_h), range(result_w), indexing='ij')
            
            # Target location in search area coordinates
            target_local_y = ty - sy1
            target_local_x = tx - sx1
            
            # Calculate Euclidean distance from target to each candidate position
            distances = np.sqrt((yy - target_local_y)**2 + (xx - target_local_x)**2)
            
            # Normalize distances (0 to 1 range)
            max_distance = np.sqrt(result_h**2 + result_w**2)
            if max_distance > 0:
                normalized_distances = distances / max_distance
            else:
                normalized_distances = np.zeros_like(distances)
            
            # Calculate penalty factor based on result range
            # Use the range of valid (non-inf) results
            valid_results = result[result < float('inf')]
            if len(valid_results) > 0:
                result_range = np.max(valid_results) - np.min(valid_results)
                # Stronger penalty (20% of result range) to prioritize locality
                distance_penalty_factor = 0.20 * result_range
            else:
                distance_penalty_factor = 100.0  # Fallback
            
            # Apply distance penalty (closer patches = lower penalty)
            distance_penalty = normalized_distances * distance_penalty_factor
            
            # Final weighted error = SSD error + distance penalty
            # This makes nearby patches more attractive
            final_error = result + distance_penalty
            
            # Mask invalid regions
            valid_result = cv2.matchTemplate(search_valid.astype(np.float32),
                                            np.ones((self.patch_size, self.patch_size), np.float32),
                                            cv2.TM_SQDIFF)
            
            final_error[valid_result > 0.1] = float('inf')
            
            # CRITICAL FIX: Mask self-match AND all masked regions more aggressively
            # 1. Mask the exact target location (prevent narcissist bug)
            self_margin = half + 3  # Larger margin
            self_y1 = max(0, target_local_y - self_margin)
            self_y2 = min(result_h, target_local_y + self_margin)
            self_x1 = max(0, target_local_x - self_margin)
            self_x2 = min(result_w, target_local_x + self_margin)
            
            if self_y2 > self_y1 and self_x2 > self_x1:
                final_error[self_y1:self_y2, self_x1:self_x2] = float('inf')
            
            # 2. Also mask any areas that overlap with ANY masked pixels
            # This prevents matching patches that contain holes
            for res_y in range(result_h):
                for res_x in range(result_w):
                    # Check if this location's patch overlaps mask
                    patch_y1 = sy1 + res_y
                    patch_y2 = sy1 + res_y + self.patch_size
                    patch_x1 = sx1 + res_x
                    patch_x2 = sx1 + res_x + self.patch_size
                    
                    if (patch_y2 <= h and patch_x2 <= w):
                        patch_region_mask = mask[patch_y1:patch_y2, patch_x1:patch_x2]
                        if np.any(patch_region_mask > 0):
                            final_error[res_y, res_x] = float('inf')
            
            # Find best
            if np.min(final_error) == float('inf'):
                return None
            
            min_loc = np.unravel_index(np.argmin(final_error), final_error.shape)
            best_y, best_x = min_loc
            
            # Convert to global coords
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
        """
        Copy patch from source to target.
        
        Returns:
            Number of pixels filled
        """
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
        """Simple fallback: fill with neighbor average."""
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
        """
        Emergency: Force fill random masked pixels to break stagnation.
        
        Args:
            num_pixels: Number of pixels to force-fill
        """
        # Find all masked pixels
        masked_coords = np.argwhere(mask > 0)
        
        if len(masked_coords) == 0:
            return
        
        # Randomly select up to num_pixels
        num_to_fill = min(num_pixels, len(masked_coords))
        random_indices = np.random.choice(len(masked_coords), num_to_fill, replace=False)
        
        for idx in random_indices:
            y, x = masked_coords[idx]
            
            # Fill with average of nearby known pixels
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
