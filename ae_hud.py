#!/usr/bin/env python3
"""
Auto Exposure and HUD Module for AlignVision

This module provides auto exposure control and image quality metrics calculation
for the AlignVision camera alignment system.
"""

import cv2
import numpy as np
import time
from pypylon import pylon


class AutoExposureController:
    """
    Auto exposure controller for Basler cameras using pypylon.
    Automatically adjusts camera exposure to maintain target brightness.
    """
    
    def __init__(self, target_brightness=128, adjustment_threshold=15, max_adjustment=500):
        """
        Initialize the auto exposure controller.
        
        Args:
            target_brightness (int): Target mean brightness value (0-255)
            adjustment_threshold (int): Brightness difference threshold to trigger adjustment
            max_adjustment (int): Maximum exposure adjustment per step (microseconds)
        """
        self.target_brightness = target_brightness
        self.adjustment_threshold = adjustment_threshold
        self.max_adjustment = max_adjustment
        self.last_adjustment_time = 0
        self.adjustment_interval = 0.5  # Minimum seconds between adjustments
        self.adjustment_history = []
        self.max_history = 5
        
    def calculate_brightness(self, frame):
        """Calculate mean brightness of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def adjust_exposure(self, frame, camera):
        """
        Adjust camera exposure based on frame brightness.
        
        Args:
            frame: Current camera frame
            camera: Basler camera instance
        """
        current_time = time.time()
        
        # Limit adjustment frequency
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return
            
        try:
            # Calculate current brightness
            brightness = self.calculate_brightness(frame)
            brightness_diff = brightness - self.target_brightness
            
            # Only adjust if difference exceeds threshold
            if abs(brightness_diff) < self.adjustment_threshold:
                return
                
            # Get current exposure time
            current_exposure = camera.ExposureTime.GetValue()
            
            # Calculate adjustment factor
            adjustment_factor = brightness_diff / self.target_brightness
            exposure_adjustment = int(current_exposure * adjustment_factor * 0.1)  # 10% adjustment
            
            # Limit adjustment magnitude
            exposure_adjustment = max(-self.max_adjustment, 
                                    min(self.max_adjustment, exposure_adjustment))
            
            # Calculate new exposure time
            new_exposure = current_exposure - exposure_adjustment
            
            # Ensure exposure is within camera limits
            min_exposure = camera.ExposureTime.GetMin()
            max_exposure = camera.ExposureTime.GetMax()
            new_exposure = max(min_exposure, min(max_exposure, new_exposure))
            
            # Apply new exposure if it's different enough
            if abs(new_exposure - current_exposure) > 10:  # Minimum 10 microsecond change
                camera.ExposureTime.SetValue(new_exposure)
                
                # Track adjustment history
                self.adjustment_history.append({
                    'time': current_time,
                    'brightness': brightness,
                    'old_exposure': current_exposure,
                    'new_exposure': new_exposure,
                    'adjustment': exposure_adjustment
                })
                
                # Limit history size
                if len(self.adjustment_history) > self.max_history:
                    self.adjustment_history.pop(0)
                
                self.last_adjustment_time = current_time
                
                print(f"Auto exposure: Brightness {brightness:.1f} -> Target {self.target_brightness}, "
                      f"Exposure {current_exposure} -> {new_exposure} ({exposure_adjustment:+d})")
                      
        except Exception as e:
            print(f"Auto exposure adjustment failed: {e}")


class ImageMetrics:
    """
    Calculate various image quality metrics for camera feed comparison.
    """
    
    def __init__(self):
        """Initialize the image metrics calculator"""
        self.reference_metrics = None
        
    def calculate_brightness(self, frame):
        """
        Calculate mean brightness of the frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Mean brightness value (0-255)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def calculate_contrast(self, frame):
        """
        Calculate contrast (standard deviation) of the frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Standard deviation of pixel intensities
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    def calculate_saturation(self, frame):
        """
        Calculate mean saturation of the frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Mean saturation value
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))  # Saturation channel
    
    def calculate_sharpness(self, frame):
        """
        Calculate sharpness using Laplacian variance.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Sharpness metric (higher = sharper)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def calculate_clipping(self, frame):
        """
        Calculate percentage of clipped pixels (pure black or white).
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            float: Percentage of clipped pixels (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total_pixels = gray.size
        
        # Count pixels that are pure black (0-1) or pure white (254-255)
        clipped_pixels = hist[0] + hist[1] + hist[254] + hist[255]
        
        return float(clipped_pixels) / total_pixels
    
    def calculate_all_metrics(self, frame):
        """
        Calculate all image quality metrics for a frame.
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        return {
            'brightness': self.calculate_brightness(frame),
            'contrast': self.calculate_contrast(frame),
            'saturation': self.calculate_saturation(frame),
            'sharpness': self.calculate_sharpness(frame),
            'clipping': self.calculate_clipping(frame)
        }
    
    def set_reference(self, frame):
        """
        Set reference frame for comparison metrics.
        
        Args:
            frame: Reference image frame (BGR format)
        """
        self.reference_metrics = self.calculate_all_metrics(frame)
        print("Reference metrics set:")
        for key, value in self.reference_metrics.items():
            print(f"  {key}: {value:.2f}")
    
    def compare_to_reference(self, frame):
        """
        Compare current frame metrics to reference frame.
        
        Args:
            frame: Current image frame (BGR format)
            
        Returns:
            dict: Comparison results with current values and differences
        """
        if self.reference_metrics is None:
            raise ValueError("Reference metrics not set. Call set_reference() first.")
        
        current_metrics = self.calculate_all_metrics(frame)
        comparison = {}
        
        for key in current_metrics:
            reference_val = self.reference_metrics[key]
            current_val = current_metrics[key]
            
            # Calculate percentage difference (avoid division by zero)
            if reference_val != 0:
                percent_diff = ((current_val - reference_val) / reference_val) * 100
            else:
                percent_diff = 0 if current_val == 0 else float('inf')
            
            comparison[key] = {
                'current': current_val,
                'reference': reference_val,
                'difference': current_val - reference_val,
                'percent_difference': percent_diff
            }
        
        return comparison


def exposure_analysis(gray_frame):
    """
    Analyze exposure characteristics of a grayscale frame.
    
    Args:
        gray_frame: Grayscale image frame
        
    Returns:
        tuple: (mean_brightness, contrast, clipping_percentage)
    """
    mean_brightness = float(np.mean(gray_frame))
    contrast = float(np.std(gray_frame))
    
    # Calculate clipping
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    total_pixels = gray_frame.size
    clipped_pixels = hist[0] + hist[1] + hist[254] + hist[255]
    clipping_percentage = float(clipped_pixels) / total_pixels
    
    return mean_brightness, contrast, clipping_percentage


def calculate_clipping(gray_frame):
    """
    Calculate clipping percentage - pixels that are pure black (0-1) or pure white (254-255).
    
    Args:
        gray_frame: Grayscale image frame
        
    Returns:
        float: Clipping percentage (0-1)
    """
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    total = gray_frame.size
    clip = float(hist[:2].sum() + hist[254:].sum()) / total
    return clip


class HUDOverlay:
    """
    Heads-Up Display overlay for showing camera metrics and status.
    """
    
    def __init__(self):
        """Initialize HUD overlay"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        self.color = (255, 255, 255)  # White text
        self.bg_color = (0, 0, 0)     # Black background
        
    def draw_text_with_background(self, frame, text, position, font_scale=None, color=None, bg_color=None):
        """
        Draw text with background rectangle for better visibility.
        
        Args:
            frame: Image frame to draw on
            text: Text string to draw
            position: (x, y) position for text
            font_scale: Font scale (optional)
            color: Text color (optional)
            bg_color: Background color (optional)
        """
        font_scale = font_scale or self.font_scale
        color = color or self.color
        bg_color = bg_color or self.bg_color
        
        # Get text size
        text_size = cv2.getTextSize(text, self.font, font_scale, self.thickness)[0]
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (position[0] - 2, position[1] - text_size[1] - 2),
                     (position[0] + text_size[0] + 2, position[1] + 2),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, position, self.font, font_scale, color, self.thickness)
    
    def draw_metrics_overlay(self, frame, metrics, reference_metrics=None):
        """
        Draw image metrics overlay on frame.
        
        Args:
            frame: Image frame to draw on
            metrics: Current frame metrics
            reference_metrics: Reference metrics for comparison (optional)
        """
        y_offset = 30
        line_height = 25
        
        # Draw metrics
        for i, (key, value) in enumerate(metrics.items()):
            y_pos = y_offset + (i * line_height)
            
            if reference_metrics and key in reference_metrics:
                ref_value = reference_metrics[key]
                diff = value - ref_value
                text = f"{key.title()}: {value:.2f} ({diff:+.2f})"
                
                # Color code based on difference
                if key == 'clipping':
                    # Red if clipping is high
                    color = (0, 0, 255) if value > 0.1 else (0, 255, 0)
                elif abs(diff) < 0.1 * abs(ref_value) if ref_value != 0 else abs(diff) < 5:
                    color = (0, 255, 0)  # Green for small differences
                else:
                    color = (0, 255, 255)  # Yellow for larger differences
            else:
                text = f"{key.title()}: {value:.2f}"
                color = self.color
            
            self.draw_text_with_background(frame, text, (10, y_pos), color=color)
    
    def draw_auto_exposure_status(self, frame, controller):
        """
        Draw auto exposure status information.
        
        Args:
            frame: Image frame to draw on
            controller: AutoExposureController instance
        """
        if not controller.adjustment_history:
            status_text = "Auto Exposure: No adjustments"
            color = (128, 128, 128)  # Gray
        else:
            last_adj = controller.adjustment_history[-1]
            brightness = last_adj['brightness']
            exposure = last_adj['new_exposure']
            
            status_text = f"Auto Exp: B={brightness:.0f} E={exposure:.0f}us"
            
            # Color based on how close to target brightness
            diff = abs(brightness - controller.target_brightness)
            if diff < controller.adjustment_threshold:
                color = (0, 255, 0)  # Green - on target
            elif diff < controller.adjustment_threshold * 2:
                color = (0, 255, 255)  # Yellow - close
            else:
                color = (0, 0, 255)  # Red - far from target
        
        # Draw in top-right corner
        frame_height, frame_width = frame.shape[:2]
        text_size = cv2.getTextSize(status_text, self.font, self.font_scale, self.thickness)[0]
        position = (frame_width - text_size[0] - 10, 30)
        
        self.draw_text_with_background(frame, status_text, position, color=color)