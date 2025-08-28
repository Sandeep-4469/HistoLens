# HistoLens: AI Co-Pilot for Pathology

HistoLens is a demonstration system designed to bridge the "interaction gap" in VLM-powered histopathology. It combines the analytical power of a Vision-Language Model with a suite of Explainable AI (XAI) techniques and an innovative AI Co-Pilot to create a transparent, interactive, and collaborative diagnostic tool.

### Core Features

- **Multi-Modal XAI Toolkit:** Go beyond simple heatmaps with Grad-CAM, Grad-CAM++, HiResCAM, and pixel-precise Guided Grad-CAM.
- **Shortcut Learning Mitigation:** Includes advanced pre-processing techniques (ROI Masking and In-painting) to diagnose and fix "Clever Hans" effects in the VLM's reasoning.
- **AI Co-Pilot (Llama 3 Powered):**
    - **Prompt Generation:** Convert simple, vague user queries into professionally structured JSON prompts.
    - **Prompt Recommendation:** Let the AI analyze an image and recommend the best prompt from your library using semantic search.
- **Interactive Prompt Engineering:** A full UI to create, edit, and save prompts to a JSON library for iterative experimentation.

### ONE-TIME SETUP

1.  **Install Ollama:** Follow the instructions at [https://ollama.com](https://ollama.com).

2.  **Download Llama 3:** Open your terminal and run the following command. This will download the model for the AI Co-Pilot to use locally.
    ```bash
    ollama pull llama3:8b
    ```

### Running the Application

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Add Sample Images:** Place your histopathology `.jpg` or `.png` files into the `sample_images` directory. They will appear in the app's dropdown menu automatically.

3.  **Verify Model Path:** Open `app.py` and ensure the `MODEL_PATH` variable points to the correct location of your MedGemma model files.
    ```python
    MODEL_PATH = "/data2/sandeep/medgemma-4b-it"
    ```

4.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```
