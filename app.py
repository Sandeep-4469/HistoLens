import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import json
from typing import Dict, Any, Tuple
import os

from ui_utils import apply_roi_inpainting, generate_saliency_overlay, parse_model_output
from xai_utils import GradCAM, GradCAMpp, HiResCAM, GuidedBackprop
from llm_copilot import generate_professional_prompt

st.set_page_config(layout="wide", page_title="HistoLens Demo")
MODEL_PATH = "/data2/sandeep/medgemma-4b-it"
PROMPTS_FILE = "prompts.json"
DEFAULT_PROMPT_KEY = "PDL1 - Detailed"
DEFAULT_PROMPT_CONTENT = """You are a pathology assistant specialized in analyzing stained histopathology images, including PDL1 staining evaluation.
Please analyze the provided image and return your findings in the following JSON format, inside markdown triple backticks:
NOTE - Tumor cells are lightly stained and immune cells are heavily stained. Base the staining_intensity_grade on this. Be careful to differentiate brain cells.
```json
{
  "stain_type": "PDL1", "number_of_cells_stained": integer, "type_of_cells_stained": "tumor cells" or "immune cells" or "both",
  "staining_location_per_cell": "cytoplasmic" or "surface membrane" or "both", "staining_intensity_grade": integer (0-3),
  "report": "A detailed pathology-style report summarizing the observed PDL1 staining pattern, including distribution of positive cells, staining location, intensity, and any morphological observations. Avoid repeating field names."
}
```"""

def load_prompts():
    if not os.path.exists(PROMPTS_FILE): return {}
    try:
        with open(PROMPTS_FILE, 'r') as f: return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Error: Could not parse `{PROMPTS_FILE}`."); return {}

def save_prompts(prompts):
    with open(PROMPTS_FILE, 'w') as f: json.dump(prompts, f, indent=2)

@st.cache_resource
def load_model_and_processor():
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        model.eval()
        return model, processor
    except Exception as e:
        st.error(f"FATAL: Failed to load model from '{MODEL_PATH}'. Details: {e}"); return None, None

def get_sample_image_paths(folder="sample_images"):
    if not os.path.isdir(folder):
        st.warning(f"'{folder}' directory not found."); return []
    supported_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(supported_extensions)]
    return sorted(image_files)

def _find_grid_shape(num_patches: int) -> Tuple[int, int]:
    for h in range(int(np.sqrt(num_patches)), 0, -1):
        if num_patches % h == 0: return (h, num_patches // h)
    return (1, num_patches)

def perform_analysis(model, processor, image, user_prompt):
    model.eval()
    messages = [{"role": "system", "content": [{"type": "text", "text": "You are a medical vision model specialized in analyzing stained tissue images."}]}, {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return inputs, generation

def generate_explanation(model, processor, inputs, generation, target_text, xai_method):
    input_len = inputs["input_ids"].shape[-1]; generated_ids = generation[0][input_len:]
    full_generated_ids = torch.cat([inputs['input_ids'][0], generated_ids], dim=0)
    target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False); target_ids_tensor = torch.tensor(target_ids).to(model.device)
    search_result = (full_generated_ids == target_ids_tensor[0]).nonzero()
    for start_idx in search_result:
        end_idx = start_idx + len(target_ids)
        if end_idx <= len(full_generated_ids) and torch.equal(full_generated_ids[start_idx:end_idx], target_ids_tensor): break
    else: raise ValueError(f"Target text '{target_text}' not found in model's generation.")
    
    labels = torch.full_like(full_generated_ids, -100); labels[start_idx:end_idx] = full_generated_ids[start_idx:end_idx]
    pixel_values = inputs['pixel_values'].clone().requires_grad_(True)
    target_layer = model.vision_tower.vision_model.encoder.layers[-1]
    
    cam_generator = GradCAM(model, target_layer)
    gbp_generator = GuidedBackprop(model) if xai_method == 'Guided Grad-CAM' else None
    
    try:
        model.train()
        with torch.enable_grad():
            outputs = model(input_ids=full_generated_ids.unsqueeze(0), pixel_values=pixel_values, labels=labels.unsqueeze(0)); loss = outputs.loss
        model.zero_grad()
        loss.backward(retain_graph=True)

        if xai_method in ['Grad-CAM', 'Grad-CAM++', 'HiResCAM']:
            cam_map = cam_generator.calculate_cam()
            h, w = _find_grid_shape(cam_map.shape[0])
            return cam_map.reshape(h, w)
        elif xai_method == 'Guided Grad-CAM':
            saliency_map = gbp_generator.calculate_gradients(pixel_values)
            cam_heatmap = cam_generator.calculate_cam()
            h, w = _find_grid_shape(cam_heatmap.shape[0])
            return saliency_map, cam_heatmap.reshape(h, w)
    finally:
        cam_generator.remove_hooks()
        if gbp_generator: gbp_generator.remove_hooks()
        model.eval()

st.title("🔬 HistoLens: AI Co-Pilot for Pathology")

if 'prompts' not in st.session_state:
    st.session_state.prompts = load_prompts()
    if not st.session_state.prompts:
        st.warning(f"`{PROMPTS_FILE}` not found or empty."); st.session_state.prompts = {DEFAULT_PROMPT_KEY: {"prompt": DEFAULT_PROMPT_CONTENT}}; save_prompts(st.session_state.prompts)
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'prompt_index' not in st.session_state: st.session_state.prompt_index = 0
if 'model' not in st.session_state: st.session_state.model = None
if 'processor' not in st.session_state: st.session_state.processor = None

if st.session_state.model is None:
    st.header("Welcome to HistoLens")
    st.info("The application is ready. Please load the AI model to begin analysis.")
    if st.button("Load HistoLens AI Model (requires >10GB RAM)", type="primary", use_container_width=True):
        with st.spinner("Loading MedGemma model... This may take a few moments."):
            model, processor = load_model_and_processor()
            if model and processor:
                st.session_state.model = model
                st.session_state.processor = processor
                st.success("Model loaded successfully!")
                st.rerun()
            else:
                st.error("Model loading failed. Please check the terminal for errors and ensure the model path is correct.")
else:
    model = st.session_state.model
    processor = st.session_state.processor

    st.sidebar.header("Controls")
    st.sidebar.subheader("1. Select Image")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    st.sidebar.markdown("--- Or ---")
    
    example_options = get_sample_image_paths()
    if not example_options and not uploaded_file:
        st.error("No sample images found in 'sample_images' folder."); st.stop()
    selected_example = st.sidebar.selectbox("Select a sample image", example_options)
    image_path = uploaded_file or selected_example
    if not image_path:
        st.warning("Please select or upload an image."); st.stop()
    
    st.sidebar.subheader("2. AI Co-Pilot")
    with st.sidebar.expander("Generate Prompt with AI"):
        st.info("Let the AI Co-Pilot write the professional prompt for you.")
        vague_query = st.text_input("Enter Doctor's Prompt / Clinical Query:", placeholder="e.g., check for strong pdl1 in immune cells")
        if st.button("Generate Professional Prompt"):
            if vague_query:
                with st.spinner("AI is generating a structured prompt..."):
                    generated_prompt = generate_professional_prompt(vague_query)
                    if generated_prompt:
                        st.session_state.prompt_editor_text = generated_prompt
                        st.rerun()
            else:
                st.warning("Please enter a query.")
    
    st.sidebar.subheader("3. Engineer Prompt")
    prompt_keys = list(st.session_state.prompts.keys())
    selected_prompt_index = st.sidebar.selectbox("Load from Library", index=st.session_state.prompt_index, options=range(len(prompt_keys)), format_func=lambda x: prompt_keys[x])
    st.session_state.prompt_index = selected_prompt_index
    selected_prompt_key = prompt_keys[selected_prompt_index]

    if 'prompt_editor_text' not in st.session_state:
        st.session_state.prompt_editor_text = st.session_state.prompts[selected_prompt_key]['prompt']
    if st.session_state.get('last_selected_key') != selected_prompt_key:
        st.session_state.prompt_editor_text = st.session_state.prompts[selected_prompt_key]['prompt']
        st.session_state.last_selected_key = selected_prompt_key

    st.sidebar.text_area("Current Prompt", st.session_state.prompt_editor_text, height=250, key='prompt_editor_text')
    
    with st.sidebar.expander("Save Prompt"):
        new_prompt_name = st.text_input("Save current editor content as:")
        if st.button("Save"):
            if new_prompt_name:
                st.session_state.prompts[new_prompt_name] = {'prompt': st.session_state.prompt_editor_text}; save_prompts(st.session_state.prompts)
                new_keys = list(st.session_state.prompts.keys()); st.session_state.prompt_index = new_keys.index(new_prompt_name)
                st.success(f"Prompt '{new_prompt_name}' saved!"); st.rerun()
            else: st.warning("Please enter a name for the new prompt.")

    st.sidebar.subheader("4. Pre-processing")
    processing_method = st.sidebar.radio("Mitigate Shortcut Learning:", options=['inpaint', 'mask', 'none'],
        format_func=lambda x: {'inpaint': 'In-paint ROI (Best)', 'mask': 'Mask ROI', 'none': 'None'}[x], index=0)

    st.sidebar.subheader("5. Run Analysis")
    if st.sidebar.button("Analyze Image", use_container_width=True, type="primary"):
        st.session_state.analysis_results = {}
        with st.spinner("Performing analysis..."):
            original_image = Image.open(image_path).convert("RGB")
            image_to_analyze = apply_roi_inpainting(original_image, method=processing_method)
            inputs, generation = perform_analysis(model, processor, image_to_analyze, st.session_state.prompt_editor_text)
            decoded_text = processor.decode(generation[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            json_report, text_report, explanation_choices = parse_model_output(decoded_text)
            st.session_state.analysis_results = {'original_image': original_image, 'analyzed_image': image_to_analyze, 'inputs': inputs, 'generation': generation, 'json_report': json_report, 'text_report': text_report, 'explanation_choices': explanation_choices, 'explanation_image': None, 'processing_method': processing_method}
    
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results; col1, col2 = st.columns(2)
        with col1:
            st.header("Input Image"); st.image(res['original_image'], caption="Original Image.", use_container_width=True)
            if res.get('processing_method', 'none') != 'none': st.image(res['analyzed_image'], caption=f"Image Sent to AI (Method: {res['processing_method']}).", use_container_width=True)
            st.header("Visual Explanation")
            if res['explanation_choices']:
                xai_method = st.radio("Choose XAI Method:", options=['Grad-CAM', 'Grad-CAM++', 'HiResCAM', 'Guided Grad-CAM'], horizontal=True, key="xai_selector")
                target_text = st.selectbox("Explain a specific finding:", options=list(res['explanation_choices'].keys()), index=None, placeholder="Select a finding...")
                if target_text and st.button("Generate Explanation", use_container_width=True):
                    with st.spinner(f"Generating {xai_method} explanation..."):
                        try:
                            result = generate_explanation(model, processor, res['inputs'], res['generation'], res['explanation_choices'][target_text], xai_method)
                            if xai_method == 'Guided Grad-CAM':
                                saliency_map, heatmap = result
                                res['explanation_image'] = generate_saliency_overlay(res['original_image'], saliency_map, heatmap=heatmap)
                            else:
                                res['explanation_image'] = generate_saliency_overlay(res['original_image'], result)
                        except Exception as e:
                            st.error(f"Could not generate explanation: {e}"); import traceback; traceback.print_exc()
            if res.get('explanation_image'): st.image(res['explanation_image'], caption=f"Generated {st.session_state.xai_selector} Explanation.", use_container_width=True)
            else: st.info("Select a finding and XAI method, then click 'Generate Explanation'.")
        with col2:
            st.header("VLM Analytical Report"); st.json(res['json_report']); st.markdown(res['text_report'])