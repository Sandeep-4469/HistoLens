# llm_copilot.py

import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer, util
import torch

# SOLVED: The prompt is refined to ensure the model generates the full text,
# including the persona ("Point of View"), not just the JSON block.
# - A new, explicit rule (Rule 2) was added to force the model to start
#   its output with the persona/instructional text.
# - The other rules were rephrased for maximum clarity.
META_PROMPT_TEMPLATE = """You are an expert assistant for pathologists. Your task is to convert a user's simple, vague query into a highly-structured, professional prompt suitable for a Vision-Language Model (VLM) that analyzes histopathology images.

You must adhere to the following rules:
1.  The final output MUST be ONLY the complete generated prompt. Do not include any preamble, apologies, or explanations like "Here is the prompt:".
2.  The generated prompt MUST begin with a persona and instructions for the VLM (e.g., "You are a pathology assistant...").
3.  The prompt MUST end with a JSON structure, which is preceded by the word 'json' on a new line, as shown in the example.
4.  Infer the correct JSON schema and parameters based on the user's query and the provided example.
---
EXAMPLE 1:
User's Vague Query: "check for pdl1 staining intensity on immune cells"

Your Generated Prompt:
You are a pathology assistant specialized in analyzing stained histopathology images, including PDL1 staining evaluation.
Please analyze the provided image and return your findings in the following JSON format:
NOTE - Tumor cells are lightly stained and immune cells are heavily stained. Base the staining_intensity_grade on this. Be careful to differentiate brain cells.
json
{{
  "stain_type": "PDL1",
  "number_of_cells_stained": "integer",
  "type_of_cells_stained": "immune cells",
  "staining_location_per_cell": "cytoplasmic or surface membrane or both",
  "staining_intensity_grade": "integer (0-3)",
  "report": "A detailed pathology-style report summarizing the observed PDL1 staining pattern on immune cells, including distribution, location, and intensity. Avoid repeating field names."
}}
---

User's Vague Query: "{user_query}"

Your Generated Prompt:"""


@st.cache_data(show_spinner=False)
def generate_professional_prompt(user_query: str):
    """
    Generates a structured, professional prompt for a VLM from a simple user query.
    """
    try:
        full_prompt = META_PROMPT_TEMPLATE.format(user_query=user_query)
        response = ollama.chat(
            model='llama3:8b',
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=False,
            options={'temperature': 0.2}
        )
        return response['message']['content'].strip()
    except ollama.ResponseError as e:
        st.error(f"Ollama connection failed. Is the Ollama server running? Details: {e.error}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred with the AI Co-Pilot: {e}")
        return None

@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer model for embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def recommend_best_prompt(vlm_model, vlm_processor, image, prompts_library):
    """
    Recommends the best prompt from a library based on image content.
    """
    vlm_model.eval()
    image_desc_prompt = "Briefly describe the main features of this histopathology slide in one sentence."

    messages = [
        {"role": "user", "content": [{"type": "text", "text": image_desc_prompt}, {"type": "image", "image": image}]}
    ]
    inputs = vlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(vlm_model.device)

    with torch.inference_mode():
        generation = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)

    image_description = vlm_processor.decode(generation[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    if not image_description:
        return None

    embedding_model = load_embedding_model()
    prompt_names = list(prompts_library.keys())
    prompt_texts = [p['prompt'] for p in prompts_library.values()]

    image_embedding = embedding_model.encode(image_description, convert_to_tensor=True)
    prompt_embeddings = embedding_model.encode(prompt_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(image_embedding, prompt_embeddings)
    best_prompt_index = torch.argmax(cosine_scores).item()
    return prompt_names[best_prompt_index]