"""
MedTranslate: Patient-Centric Medical AI Explainer
Powered by Google MedGemma (Gemma 3 + MedSigLIP)

Features:
- Multimodal Imaging (Chest X-Rays, Dermoscopy, Radiology scans)
- Clinical Report & Lab Value Translation to plain English
- Physician Question Generator for Patient Advocacy
- Dual Execution: Live Hugging Face MedGemma inference or Interactive Demo Mode
"""

import os
import json
from pathlib import Path
from PIL import Image

try:
    import gradio as gr
except ImportError:
    raise ImportError("Please install Gradio: pip install gradio")

# Optional PyTorch & Transformers loading
HAS_TORCH = False
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_cases.json"

# Load sample dataset
sample_cases = {"imaging_cases": [], "text_cases": []}
if SAMPLE_DATA_PATH.exists():
    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        sample_cases = json.load(f)

# Global variables for model
model = None
processor = None
model_load_status = "Not initialized"

def initialize_model():
    """Attempts to load MedGemma if CUDA/GPU is available and user has access."""
    global model, processor, model_load_status
    if not HAS_TORCH:
        model_load_status = "Demo Mode (PyTorch/Transformers not installed locally)"
        return False

    if not torch.cuda.is_available():
        model_load_status = "Demo Mode (No CUDA GPU detected. MedGemma 4B is optimized for GPU/Colab)"
        return False

    try:
        print(f"Loading {MODEL_ID} on GPU...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model_load_status = f"Live Model Active ({MODEL_ID} on {torch.cuda.get_device_name(0)})"
        return True
    except Exception as e:
        model_load_status = f"Demo Mode (Live model load skipped: {str(e)[:80]}...)"
        print(f"Model load notice: {e}")
        return False

# Attempt initialization
initialize_model()

def format_output(clinical_findings: str, patient_summary: str, doctor_questions: list) -> str:
    """Formats markdown response into clean, patient-friendly UI sections."""
    questions_md = "\n".join([f"- **Q{i+1}:** {q}" for i, q in enumerate(doctor_questions)])
    return f"""### 🩺 Clinical Findings (Draft for Clinicians)
{clinical_findings}

---

### 💡 Plain-English Summary (For Patients)
{patient_summary}

---

### ❓ Questions to Ask Your Doctor at Your Next Visit
{questions_md}
"""

def analyze_image(image: Image.Image, context: str, sample_case_choice: str):
    """Processes multimodal medical image input."""
    # 1. Check if user selected a preloaded sample case
    if sample_case_choice and sample_case_choice != "Custom Upload":
        for case in sample_cases.get("imaging_cases", []):
            if case["title"] == sample_case_choice:
                return (
                    format_output(
                        case["clinical_findings"],
                        case["patient_explanation"],
                        case["doctor_questions"]
                    ),
                    f"Dataset Reference: {case.get('dataset_source', 'Open Medical Archive')}"
                )

    if image is None:
        return "⚠️ Please upload a medical image or select a sample case above.", ""

    # 2. Live inference if model is loaded on GPU
    if model is not None and processor is not None:
        try:
            prompt = (
                "You are MedGemma, a helpful clinical AI assistant. "
                "Examine this medical image and clinical context. Provide:\n"
                "1. Structured Clinical Findings\n"
                "2. A patient-friendly explanation at a 6th-grade reading level\n"
                "3. Three clarifying questions the patient should ask their physician.\n"
                f"Context: {context if context else 'None provided'}"
            )
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_tokens = model.generate(**inputs, max_new_tokens=600)
            result = processor.decode(output_tokens[0], skip_special_tokens=True)
            return result, "Inference executed via MedGemma 4B on GPU"
        except Exception as err:
            return f"Error during model inference: {err}", "Model Error"

    # 3. Fallback / Demonstration Mode for local laptop testing
    return format_output(
        clinical_findings="Visual inspection indicates localized radiopacity / anatomical variation consistent with user query. (Demo Mode: Connect to MedGemma on Colab GPU for full generative analysis)",
        patient_summary="The scan has been processed. While awaiting physician review, the image details highlight specific tissue densities that correlate with your medical history.",
        doctor_questions=[
            "What specific findings on this scan require confirmation through blood work or follow-up imaging?",
            "Are there any physical symptoms I should monitor while waiting for the official radiologist report?",
            "What is the recommended timeline for my follow-up appointment?"
        ]
    ), "Demo Simulation Mode (Host on Colab GPU / HF Spaces to activate live weights)"

def analyze_report(report_text: str, sample_case_choice: str):
    """Processes clinical note or lab report text."""
    if sample_case_choice and sample_case_choice != "Custom Input":
        for case in sample_cases.get("text_cases", []):
            if case["title"] == sample_case_choice:
                glossary = "\n".join([f"- **{item['term']}**: {item['meaning']}" for item in case.get("jargon_breakdown", [])])
                questions = "\n".join([f"- **Q{i+1}:** {q}" for i, q in enumerate(case["doctor_questions"])])
                output = f"""### 💡 Plain-English Summary
{case['patient_explanation']}

---

### 📖 Medical Jargon Explained
{glossary}

---

### ❓ Questions for Your Doctor
{questions}
"""
                return output, f"Source: {case.get('dataset_source', 'MTSamples Archive')}"

    if not report_text or len(report_text.strip()) < 10:
        return "⚠️ Please paste a clinical report or select a preloaded sample.", ""

    # Live model generation if available
    if model is not None and processor is not None:
        try:
            prompt = (
                "You are MedGemma. Review the following medical note and explain it in clear, reassuring, "
                "patient-friendly language. Define any complex medical terms and provide 3 questions for their doctor.\n"
                f"Medical Note:\n{report_text}"
            )
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_tokens = model.generate(**inputs, max_new_tokens=500)
            result = processor.decode(output_tokens[0], skip_special_tokens=True)
            return result, "Inference executed via MedGemma 4B"
        except Exception as err:
            return f"Error during model inference: {err}", "Model Error"

    # Fallback explanation
    return """### 💡 Plain-English Summary
Your medical notes have been reviewed. The report discusses key vital signs and test parameters. Standard medical reports frequently contain specialized abbreviations (such as eGFR, LDL, SpO2) that summarize organ function and metabolic health.

---

### 📖 Common Terms Demystified
- **Clinical Observation**: Notes recorded by the attending healthcare professional during evaluation.
- **Reference Range**: The normal bracket within which healthy test results typically fall.

---

### ❓ Suggested Questions for Your Physician
- Could you explain what these specific values mean for my day-to-day routine?
- Are any of these readings outside your target range for my specific age and history?
- What follow-up checks or lifestyle modifications do you recommend?
""", "Demo Simulation Mode"

# Build Gradio UI
custom_css = """
.title-box { text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%); color: white; border-radius: 12px; margin-bottom: 1rem; }
.disclaimer-box { background-color: #fef3c7; border-left: 5px solid #d97706; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem; color: #92400e; font-size: 0.9rem; }
.badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; background: #e0f2fe; color: #0369a1; }
"""

with gr.Blocks(title="MedTranslate - Medical AI Explainer", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""
    <div class="title-box">
        <h1>🩺 MedTranslate: Patient-First Medical AI</h1>
        <p>Demystifying medical imaging and clinical reports using <strong>Google MedGemma</strong></p>
        <span class="badge">Model Status: {model_load_status}</span>
    </div>
    <div class="disclaimer-box">
        ⚠️ <strong>Important Medical AI Disclaimer:</strong> MedTranslate is built for educational, research demonstration, and health literacy purposes only. It is not an FDA-cleared diagnostic device and should never replace qualified clinical judgement or consultation with a licensed physician.
    </div>
    """)

    imaging_choices = ["Custom Upload"] + [c["title"] for c in sample_cases.get("imaging_cases", [])]
    text_choices = ["Custom Input"] + [c["title"] for c in sample_cases.get("text_cases", [])]

    with gr.Tabs():
        # Tab 1: Multimodal Radiology & Imaging
        with gr.TabItem("🖼️ Medical Image Explainer"):
            gr.Markdown("Upload a medical scan (Chest X-ray, Dermatology photo, Ultrasound) or select a benchmark case to see MedGemma's dual clinical and patient breakdown.")
            with gr.Row():
                with gr.Column(scale=1):
                    sample_img_dropdown = gr.Dropdown(
                        choices=imaging_choices,
                        value=imaging_choices[1] if len(imaging_choices) > 1 else "Custom Upload",
                        label="Select Open Dataset Benchmark Sample"
                    )
                    image_input = gr.Image(type="pil", label="Medical Scan / Image")
                    context_input = gr.Textbox(
                        lines=2,
                        placeholder="e.g., Patient has 4-day history of cough, fever, and right-sided pleuritic pain.",
                        label="Optional Clinical Context / Symptoms"
                    )
                    img_submit_btn = gr.Button("🔍 Analyze Medical Scan", variant="primary")

                with gr.Column(scale=1):
                    img_output = gr.Markdown(label="Analysis Results")
                    img_meta = gr.Label(label="Source & Execution Metadata")

            img_submit_btn.click(
                fn=analyze_image,
                inputs=[image_input, context_input, sample_img_dropdown],
                outputs=[img_output, img_meta]
            )

        # Tab 2: Clinical Report & Lab Translator
        with gr.TabItem("📄 Clinical Note & Lab Translator"):
            gr.Markdown("Paste clinical notes, discharge summaries, or lab test panels to generate plain-English explanations and a medical jargon dictionary.")
            with gr.Row():
                with gr.Column(scale=1):
                    sample_txt_dropdown = gr.Dropdown(
                        choices=text_choices,
                        value=text_choices[1] if len(text_choices) > 1 else "Custom Input",
                        label="Select Preloaded Medical Record (MTSamples / Anonymized Lab)"
                    )
                    text_input = gr.Textbox(
                        lines=7,
                        placeholder="Paste clinical note, discharge summary, or blood test panel here...",
                        label="Medical Text / Lab Panel"
                    )
                    txt_submit_btn = gr.Button("📑 Translate Report", variant="primary")

                with gr.Column(scale=1):
                    txt_output = gr.Markdown(label="Translation Results")
                    txt_meta = gr.Label(label="Source & Execution Metadata")

            txt_submit_btn.click(
                fn=analyze_report,
                inputs=[text_input, sample_txt_dropdown],
                outputs=[txt_output, txt_meta]
            )

        # Tab 3: About & Architecture
        with gr.TabItem("ℹ️ About & Technology"):
            gr.Markdown("""
            ### Technology Stack & Architecture
            - **Foundation Model:** `google/medgemma-4b-it` (Google Health & DeepMind)
            - **Multimodal Visual Encoder:** **MedSigLIP** (specifically tuned on biomedical imagery including radiology and histopathology)
            - **Interface:** Gradio 4.x
            - **Datasets Reference:**
              - **NIH ChestX-ray14:** Over 100,000 frontal chest radiographs (NIH Clinical Center).
              - **HAM10000:** Dermatoscopy benchmark dataset of multi-source dermatological lesions.
              - **MTSamples:** Open anonymized clinical transcription repository.

            ### Running Live on Cloud GPU
            Because full MedGemma weights require ~8GB-16GB VRAM, open `MedTranslate_MedGemma.ipynb` in **Google Colab** with a free T4 GPU to run live inference on real weights!
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
