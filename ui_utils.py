import streamlit as st
from PIL import Image
import numpy as np
import cv2
import json
import re
from typing import Dict, Any

def apply_roi_inpainting(image: Image.Image, method: str = 'inpaint') -> Image.Image:
    if method == 'none': return image
    img_cv = np.array(image); img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return image
    largest_contour = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(img_gray)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    if method == 'mask':
        masked_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=contour_mask)
        final_image_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    elif method == 'inpaint':
        mean_color_bgr = cv2.mean(img_bgr, mask=contour_mask)[:3]
        background = np.full(img_bgr.shape, mean_color_bgr, dtype=np.uint8)
        inpainted_bgr = np.where(contour_mask[:, :, None] == 255, img_bgr, background)
        final_image_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    else: final_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_image_rgb)

def generate_saliency_overlay(image, saliency_map, heatmap=None):
    img_cv = np.array(image)
    if saliency_map.ndim == 3 and saliency_map.shape[2] == 3:
        saliency_map = np.max(np.abs(saliency_map), axis=-1)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    if heatmap is not None:
        heatmap_resized = cv2.resize(heatmap, (saliency_map.shape[1], saliency_map.shape[0]))
        saliency_map *= heatmap_resized
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(img_gray, dtype=np.float32)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi_mask, [largest_contour], -1, 1.0, thickness=cv2.FILLED)
    roi_mask_resized = cv2.resize(roi_mask, (saliency_map.shape[1], saliency_map.shape[0]))
    saliency_map *= roi_mask_resized
    saliency_map_uint8 = (saliency_map * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(saliency_map_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.resize(heatmap_colored, (img_cv.shape[1], img_cv.shape[0]))
    overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def parse_model_output(decoded_text: str):
    match = re.search(r"```json\s*([\s\S]+?)\s*```", decoded_text)
    if not match: return {"error": "No JSON block found."}, decoded_text, {}
    try:
        # ***** THE DEFINITIVE FIX FOR THE TypeError IS HERE *****
        # Changed match.group[1] to the correct syntax: match.group(1)
        json_data = json.loads(match.group(1))
        
        report_str = json_data.get("report", "No report found.")
        explanation_choices = {f'"{k}": {json.dumps(v)}': f'"{k}": {json.dumps(v)}' for k, v in json_data.items() if k != "report"}
        return json_data, report_str, explanation_choices
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format."}, "Could not parse JSON.", {}