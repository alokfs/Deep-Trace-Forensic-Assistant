import os
import uuid
import warnings
import math
import tempfile
from pathlib import Path
from typing import List, Tuple, Union, Optional
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fpdf import FPDF
import mediapipe as mp
from facenet_pytorch import InceptionResnetV1, MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchcam.methods import GradCAM as TCGradCAM
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai

warnings.filterwarnings("ignore")

def _ensure_deps():
    required = ["mediapipe", "fpdf", "google-generativeai", "gradio", "opencv-python", "torch", "torchvision", "facenet-pytorch", "pytorch-grad-cam", "shap", "matplotlib", "pandas"]
    missing = []
    for lib in required:
        try:
            if lib == "google-generativeai":
                import google.generativeai
            elif lib == "opencv-python":
                import cv2
            elif lib == "facenet-pytorch":
                import facenet_pytorch
            elif lib == "pytorch-grad-cam":
                import pytorch_grad_cam
            else:
                __import__(lib.replace("-", "_"))
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}")
        os.system(f"pip install --quiet --upgrade {' '.join(missing)}")
        print("Dependencies installed.")

_ensure_deps()

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

api_key = "AIzaSyC-WEXgZFVgSq0LDiPPTqPcF7iQ4F4Mh2g"
genai.configure(api_key=api_key)

plt.set_loglevel("ERROR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_face_det = MTCNN(select_largest=False, post_process=False, device=device).eval().to(device)

_df_model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=device)
try:
    if os.path.exists("resnet_inception.pth"):
        _df_model.load_state_dict(torch.load("resnet_inception.pth", map_location=device)["model_state_dict"])
    else:
        print("WARNING: 'resnet_inception.pth' not found. Using default weights.")
except Exception as e:
    print(f"Error loading Deepfake model: {e}")
_df_model.to(device).eval()

try:
    _df_cam = GradCAM(_df_model, target_layers=[_df_model.block8.branch1[-1]], use_cuda=(device.type == "cuda"))
except Exception as e:
    print(f"Warning: GradCAM init failed: {e}")
    _df_cam = None

def _get_layer(model, name: str):
    mods = dict(model.named_modules())
    return mods.get(name) or next(m for n, m in mods.items() if n.endswith(name))

BIN_ID = "haywoodsloan/ai-image-detector-deploy"
try:
    _bin_proc = AutoImageProcessor.from_pretrained(BIN_ID)
    _bin_mod  = AutoModelForImageClassification.from_pretrained(BIN_ID).to(device).eval()
    _CAM_LAYER_BIN = "encoder.layers.3.blocks.1.layernorm_after"
    _bin_cam = TCGradCAM(_bin_mod, target_layer=_get_layer(_bin_mod, _CAM_LAYER_BIN))
except Exception as e:
    print(f"Error loading HuggingFace model: {e}")
    _bin_mod = None

try:
    if os.path.exists("Alokclass.pt"):
        _susy_mod = torch.jit.load("Alokclass.pt").to(device).eval()
    else:
        print("WARNING: 'Alokclass.pt' not found. Generator classification disabled.")
        _susy_mod = None
except Exception as e:
    print(f"Error loading Generator Classifier: {e}")
    _susy_mod = None

_GEN_CLASSES = ["Claude(Anthropic)", "Gemini(Google)", "MJ V5/V6", "LLama(Meta)", "chatgpt(OpenAI)"]
_PATCH, _TOP = 224, 5
_to_tensor = transforms.ToTensor()
_to_gray   = transforms.Compose([transforms.PILToTensor(), transforms.Grayscale()])

def _calibrate_df(p: float) -> float:
    return p

def _calibrate_ai(p: float) -> float:
    return p

UNCERTAIN_GAP = 0.10 

def _fuse(p_ai: float, p_df: float) -> float:
    return 1 - (1 - p_ai) * (1 - p_df)

def _verdict(p: float) -> str:
    if abs(p - 0.5) <= (UNCERTAIN_GAP / 2.0):
        return "uncertain"
    return "Fake" if p > 0.5 else "Real"

def _extract_landmarks(rgb: np.ndarray) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    res  = mesh.process(rgb)
    mesh.close()
    
    if not res.multi_face_landmarks:
        return rgb, None
    
    h, w, _ = rgb.shape
    out = rgb.copy()
    
    for lm in res.multi_face_landmarks[0].landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (cx, cy), 1, (0, 255, 0), -1)
    return out, None

def _overlay_cam(cam, base: np.ndarray):
    if torch.is_tensor(cam):
        cam = cam.detach().cpu().numpy()
    
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    heat = Image.fromarray((plt.cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)).resize((base.shape[1], base.shape[0]), Image.BICUBIC)
    return Image.blend(Image.fromarray(base).convert("RGBA"), heat.convert("RGBA"), alpha=0.45)

def _sanitize_for_pdf(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return ''.join((ch if ord(ch) < 256 else '?') for ch in text)

def _render_pdf(title: str, verdict: str, conf: dict, pages: List[Image.Image]) -> str:
    temp_dir = tempfile.gettempdir()
    out = Path(temp_dir) / f"report_{uuid.uuid4().hex}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    safe_title = _sanitize_for_pdf(title)
    safe_verdict = _sanitize_for_pdf(verdict)
    safe_conf = {k: _sanitize_for_pdf(v) if isinstance(v, str) else v for k, v in conf.items()}

    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, safe_title, ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)

    conf_text = f"Verdict: {safe_verdict}\nConfidence -> Real {safe_conf['Real']:.3f}  Fake {safe_conf['Fake']:.3f}"
    pdf.multi_cell(0, 6, _sanitize_for_pdf(conf_text))

    for idx, img in enumerate(pages):
        pdf.ln(4)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 6, _sanitize_for_pdf(f"Figure {idx+1}"), ln=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.convert("RGB").save(tmp, format="JPEG")
            tmp_path = tmp.name
        pdf.image(tmp_path, x=10, w=90)
        os.unlink(tmp_path)
    
    pdf.output(str(out))
    return str(out)

def generate_explanation(label, image: Image.Image, generator_info=None):
    gen_text = f"\nThe detected AI generator model: {generator_info}.\n" if generator_info else ""
    prompt = f"""
    You are an expert AI forensic assistant.
    An image or video was analyzed for deepfake detection.
    
    - Decision: {label}
    {gen_text}
    If this image appears to be of a celebrity or public figure, write a short forensic report (around 120 words) explaining the decision, the **Reason** (why it's fake or real), and describe the likely place and period the photo was taken.
    Name:=> The name of the person in the image (if identifiable, otherwise "Unknown")
    Location:=> The likely location where the photo was taken, based on visual cues (e.g., landmarks, background, clothing).
    If the image seems like a normal or unknown person (not a celebrity or public figure), do not add location or time details; simply write a basic explanation of real/fake **Reason** only and likely period file date the photo was taken.(around 110 words)
    date:=> The date when the photo was likely taken, based on visual cues and metadata (if available).
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini explanation: {e}")
        return f"Failed to generate forensic explanation. Error: {e}"

def generate_video_explanation(label, image: Image.Image):
    prompt = f"""
    You are an expert AI forensic assistant.
    A video was analyzed, and the final decision is: **{label}**.
    
    This decision was based on the most suspicious frame from the video (provided).
    Analyze this specific frame and write a forensic report (around 110 words) 
    explaining the **Reason** for this '{label}' verdict. 
    
    Focus on visual artifacts:
    - Unnatural skin texture/smoothness
    - Lighting/shadow inconsistencies
    - Blurring/artifacts around face boundaries
    - Unnatural facial features/expressions
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        print(f"Error generating Gemini explanation for video: {e}")
        return f"Failed to generate forensic explanation. Error: {e}"

@spaces.GPU
def _susy_predict(img: Image.Image):
    if _susy_mod is None:
        return {k: 0.0 for k in _GEN_CLASSES}

    w, h = img.size
    npx, npy = max(1, w // _PATCH), max(1, h // _PATCH)
    patches  = np.zeros((npx * npy, _PATCH, _PATCH, 3), dtype=np.uint8)
    
    idx = 0
    for i in range(npx):
        for j in range(npy):
            x, y = i * _PATCH, j * _PATCH
            crop = img.crop((x, y, x+_PATCH, y+_PATCH)).resize((_PATCH, _PATCH))
            patches[idx] = np.array(crop)
            idx += 1
    
    contrasts = []
    for p in patches:
        g = _to_gray(Image.fromarray(p)).squeeze(0).numpy()
        glcm = graycomatrix(g, [5], [0], 256, symmetric=True, normed=True)
        contrasts.append(graycoprops(glcm, "contrast")[0, 0])
        
    sort_idx  = np.argsort(contrasts)[::-1][:_TOP]
    selected_patches = patches[sort_idx]
    
    tens = torch.from_numpy(selected_patches.transpose(0, 3, 1, 2)).float() / 255.0
    
    with torch.no_grad():
        probs = _susy_mod(tens.to(device)).softmax(-1).mean(0).cpu().numpy()[1:]
    return dict(zip(_GEN_CLASSES, probs))

@spaces.GPU
def _predict_image(pil: Image.Image):
    if pil is None:
        return "No Image", {}, [], gr.update(visible=False), None, "Please upload an image."

    gallery: List[Image.Image] = []
    generator_info = None

    try:
        face = _face_det(pil)
    except Exception:
        face = None
        
    if face is not None:
        ft = F.interpolate(face.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False).float() / 255.0
        p_df_raw = torch.sigmoid(_df_model(ft.to(device))).item()
        p_df = _calibrate_df(p_df_raw)
        
        crop_np = (ft.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        if _df_cam:
            cam_df  = _df_cam(ft, [ClassifierOutputTarget(0)])[0]
            gallery.append(_overlay_cam(cam_df, crop_np))
        
        gallery.append(Image.fromarray(_extract_landmarks(cv2.cvtColor(np.array(pil), cv2.COLOR_BGR2RGB))[0]))
    else:
        p_df = 0.5
    
    if _bin_mod:
        inp_bin = _bin_proc(images=pil, return_tensors="pt").to(device)
        logits  = _bin_mod(**inp_bin).logits.softmax(-1)[0]
        p_ai_raw = logits[0].item() 
        p_ai = _calibrate_ai(p_ai_raw)
        
        winner_idx = 0 if p_ai_raw >= logits[1].item() else 1
        inp_bin_h = {k: v.clone().detach().requires_grad_(True) for k, v in inp_bin.items()}
        cam_bin = _bin_cam(winner_idx, scores=_bin_mod(**inp_bin_h).logits)[0]
        gallery.append(_overlay_cam(cam_bin, np.array(pil)))
    else:
        p_ai_raw = 0.0
        p_ai = 0.0
        logits = [0, 0]

    bar_plot = gr.update(visible=False)
    if p_ai_raw > 0.5: 
        gen_probs = _susy_predict(pil)
        df_res = pd.DataFrame(gen_probs.items(), columns=["class", "prob"])
        bar_plot = gr.update(value=df_res, visible=True)
        if not df_res.empty:
            generator_info = df_res.sort_values(by="prob", ascending=False).iloc[0]["class"]

    p_final = _fuse(p_ai, p_df)
    verdict = _verdict(p_final)
    conf    = {"Real": round(1-p_final, 4), "Fake": round(p_final, 4)}
    
    pdf = _render_pdf("Unified Detector Report", verdict, conf, gallery[:3])

    print("Generating Gemini explanation...")
    explanation = generate_explanation(verdict, pil, generator_info)
    print("Analysis complete.")

    return verdict, conf, gallery, bar_plot, pdf, explanation

@spaces.GPU
def _predict_video(video_path, progress=gr.Progress(track_tqdm=True)):
    if video_path is None:
        return "No video uploaded", {"Real": 0.5, "Fake": 0.5}, "Please upload a video to analyze."

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error opening video", {"Real": 0.5, "Fake": 0.5}, "Could not open file."

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = max(1, int(fps)) 
        
        frame_num = 0
        best_score = -1.0
        best_frame = None
        
        progress(0, desc="Starting video analysis...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frame_skip == 0:
                progress(frame_num / frame_count, desc=f"Scanning frame {frame_num}/{frame_count}")
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                try:
                    face = _face_det(pil_img)
                except Exception:
                    face = None
                
                if face is not None:
                    ft = F.interpolate(face.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False).float() / 255.0
                    
                    with torch.no_grad():
                        p_df_raw = torch.sigmoid(_df_model(ft.to(device))).item()
                    
                    if p_df_raw > best_score:
                        best_score = p_df_raw
                        best_frame = pil_img.copy()

            frame_num += 1
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return f"Error: {e}", {"Real": 0.5, "Fake": 0.5}, f"Processing error: {e}"
    finally:
        cap.release()

    if best_frame is None:
        return "Uncertain", {"Real": 0.5, "Fake": 0.5}, "No faces detected in the video."

    p_final = _calibrate_df(best_score)
    verdict = _verdict(p_final)
    conf    = {"Real": round(1-p_final, 4), "Fake": round(p_final, 4)}
  
    print("Generating Gemini explanation for video...")
    explanation = generate_video_explanation(verdict, best_frame)
    
    return verdict, conf, explanation

_css = """
footer { visibility:hidden!important }
.logo, #logo { display:none!important }
#forensic_reason_box p, #forensic_reason_box_vid p { font-size: 1.8em !important; line-height: 1.4; }
"""

with gr.Blocks(css=_css, title="Deep Trace Forensic Assistant") as demo:
    gr.Markdown("## Deep Trace Forensic Assistant (Image & Video)")

    with gr.Tab("Forensic Analysis (Image)"):
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(label="Upload image", type="pil")
                btn_i  = gr.Button("Analyze Image", variant="primary")
        
            with gr.Column(scale=2):
                txt_v  = gr.Textbox(label="Verdict", interactive=False)
                lbl_c  = gr.Label(label="Confidence")
        
        gal   = gr.Gallery(label="Forensic Heatmaps & Landmarks", columns=3, height=320)
        bar   = gr.BarPlot(x="class", y="prob", title="Likely AI Generator", y_label="Probability", visible=False)
        pdf_f = gr.File(label="Download Forensic Report (PDF)")
        reason_box = gr.Markdown(label="Forensic Explanation", elem_id="forensic_reason_box")

        btn_i.click(_predict_image, [img_in], [txt_v, lbl_c, gal, bar, pdf_f, reason_box])

    with gr.Tab("Video Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                vid_in = gr.Video(label="Upload video")
                btn_v  = gr.Button("Analyze Video", variant="primary")
           
            with gr.Column(scale=2):
                txt_v_vid  = gr.Textbox(label="Verdict", interactive=False)
                lbl_c_vid  = gr.Label(label="Confidence")
                reason_box_vid = gr.Markdown(label="Forensic Explanation", elem_id="forensic_reason_box_vid")
    
        btn_v.click(_predict_video, [vid_in], [txt_v_vid, lbl_c_vid, reason_box_vid])

if __name__ == "__main__":
    demo.launch(share=True, show_api=False)