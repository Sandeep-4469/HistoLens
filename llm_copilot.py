# llm_copilot.py

import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer, util
import torch

META_PROMPT_TEMPLATE = """You are an expert pathology assistant specializing in histopathology image interpretation. 
Your task is to transform a vague user query into a structured, professional prompt suitable for a Vision-Language Model (VLM) 
that analyzes histopathology images.

Follow these rules strictly:
1. The final output MUST be ONLY the generated prompt. Do not include any preamble, comments, or explanations.
2. The generated prompt MUST begin with a persona statement and clear instructions for the VLM (e.g., "You are a pathology assistant...").
3. The generated prompt MUST end with a JSON schema, preceded by the word 'json' on its own line.
4. Infer the appropriate JSON fields, data types, and constraints from the user query. Use pathology-specific terminology.
5. The "report" field MUST summarize findings in pathology-style prose and MUST NOT repeat field names.
6. If the query specifies a stain or marker, include it in the JSON (e.g., "stain_type": "PDL1").
7. If the query references cell types, ensure they are explicitly noted, only TUMOR CELLS not IMMUNE CELLS
8. Always include staining intensity (graded 1–3 and 0 if not stain), and cell counts if relevant.
9. Always include an "explanation" field, describing the reasoning behind the interpretation (e.g., staining pattern, distribution, morphology).
10. stain name is given by the USER
---
EXAMPLE:
User's Vague Query: "check for pdl1 staining intensity on immune cells"

Your Generated Prompt:
You are a pathology assistant specialized in analyzing stained histopathology images, including PDL1 immunohistochemistry.
Please analyze the provided image and return your findings in the following JSON format:
Note – Tumor cells may appear lightly stained while immune cells may appear heavily stained. Ensure accurate distinction. 
Be careful to exclude non-relevant brain parenchymal cells if present.
9. Regarding staining Location, central structure is NUCLEAR, outer part is CYTOPLASM and outline of the cytoplasm is MEMBRANE. FOR Ki-67
10. For PDL-1 stain, regarding the staining location, mostly CYTOPLASM and this PDL-1 cannot stain the NUCLEUS
json
{{
  "stain_type": "PDL1",
  "percentage_of_cells_stained": "0-100",
  "type_of_cells_stained": "tumor cells",
  "staining_location_per_cell": "nuclear or cytoplasmic or membrane or both",
  "staining_intensity_grade": "integer (1–3)",
  "report": "Detailed pathology report summarizing PDL1 staining pattern on Tumor cells, including distribution, staining location, and intensity.",
  "explanation": "Explain why these findings were made, referencing staining characteristics, cell morphology, and tissue context."
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