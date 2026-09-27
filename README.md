# 🩺 MedTranslate: Patient-First Medical AI Explainer
> Powered by **Google MedGemma** (`google/medgemma-4b-it`) — built on **Gemma 3** & **MedSigLIP**

---

## 📌 Project Overview
**MedTranslate** bridges the gap between complex clinical data and patient understanding. Using Google's open-weights medical foundation model (**MedGemma 4B**), the app takes:
1. **Medical Scans** (Frontal Chest X-Rays, Dermoscopy skin lesions)
2. **Clinical Notes & Lab Panels** (Discharge summaries, blood work panels)

And produces a dual output:
* **🩺 Clinical Findings:** A structured draft for healthcare professionals.
* **💡 Plain-English Summary:** A patient-friendly breakdown (grade 6–8 reading level) explaining what the results mean.
* **❓ Physician Questions:** Tailored questions to help patients advocate for themselves at their next appointment.

---

## 📊 Datasets Used & Recommended

For testing and demonstrating MedTranslate, use these open-access, de-identified medical benchmark datasets:

| Dataset | Modality | Source / Access | Best For |
| :--- | :--- | :--- | :--- |
| **NIH ChestX-ray14** | Frontal Chest Radiographs | [NIH Clinical Center / Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Demonstrating pneumonia, cardiomegaly, effusion detection. |
| **COVID-19 Radiography Database** | Chest X-Rays (Normal, Viral Pneumonia, COVID) | [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | Quick, clean sample image downloads (PNG/JPEG). |
| **HAM10000** | Dermatoscopy Skin Lesions | [Harvard Dataverse / ISIC Archive](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | Skin lesion analysis, melanoma ABCDE assessment. |
| **MTSamples** | De-identified Clinical Notes & Transcriptions | [Hugging Face / MTSamples](https://huggingface.co/datasets/tatsu-lab/alpaca) / [MTSamples.com](https://mtsamples.com/) | Clinical discharge summaries, SOAP notes, lab reports. |

> *Note: Pre-extracted sample cases from these benchmarks are already included inside [`sample_data/sample_cases.json`](sample_data/sample_cases.json) for instant preview!*

---

## 💻 Hardware Considerations: Running on Laptop vs Cloud GPU

* **Your Laptop Hardware:** 
  * Intel UHD Graphics (integrated GPU, no dedicated NVIDIA CUDA).
  * 8 GB System RAM.
* **Can you run it locally on the laptop?**
  * MedGemma 4B is a 4-billion parameter multimodal model (vision + language). Loading unquantized model weights requires ~8–10 GB of memory just for the model. On an 8 GB Windows system without a dedicated GPU, running the full model on CPU will trigger extreme memory swapping (thrashing) or Out-Of-Memory errors.
  * **However:** You can run the Gradio interface locally in **Demo / Sample Simulation Mode** to test the UI, UX, and sample reports.
* **The Best Solution for Live Model Inference:**
  * **Google Colab (Free NVIDIA T4 GPU):** Run [`MedTranslate_MedGemma.ipynb`](MedTranslate_MedGemma.ipynb) on Colab. It gives you 16 GB of free GPU VRAM, downloads the real weights in seconds, and generates a live public Gradio URL (`https://xxxx.gradio.live`).
  * **Hugging Face Spaces:** You can also deploy this repository as a free Space on Hugging Face to have a permanent portfolio link.

---

## 🚀 How to Run

### Method 1: Google Colab (Recommended for Live Weights)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `MedTranslate_MedGemma.ipynb`.
3. In Colab, go to **Runtime > Change runtime type** and select **T4 GPU** (free).
4. Run all cells:
   - Accept the license on [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it).
   - Enter your Hugging Face Access Token when prompted.
5. Colab will generate a public link (`https://xxxx.gradio.live`). Open it in your browser!

### Method 2: Local Demo / Simulation Mode
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:7860` to interact with the preloaded cases.

---

## 🔬 Clinical Evaluation & Quality Checks

When running MedTranslate on clinical test cases:
1. **Clinical Alignment:** Verify that drafted findings correctly distinguish normal anatomical limits from pathology (e.g. consolidations, effusion, suspicious border asymmetry).
2. **Reading Level Verification:** Ensure patient explanations consistently stay within 6th- to 8th-grade readability without jargon overload.
3. **Safety Disclaimers:** Every clinical AI output must emphasize that it is intended for educational and literacy assistance rather than unsupervised diagnostic intervention.
