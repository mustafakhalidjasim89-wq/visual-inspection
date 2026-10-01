import base64
import io
import json
import os
import streamlit as st
from PIL import Image
from openai import OpenAI
from ultralytics import YOLO

# Streamlit Page Config
st.set_page_config(
    page_title="Multimodal Visual Inspection",
    page_icon="🔍",
    layout="wide"
)

# Initialize Vision API Client (Groq, OpenRouter, or OpenAI)
api_key = st.secrets.get("VISION_API_KEY") or os.getenv("VISION_API_KEY", "")
base_url = st.secrets.get("VISION_BASE_URL") or os.getenv("VISION_BASE_URL", "https://api.groq.com/openai/v1")
model_name = st.secrets.get("VISION_MODEL_NAME") or os.getenv("VISION_MODEL_NAME", "llama-3.2-11b-vision-preview")

client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

# Cache YOLO model in Streamlit session memory
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# Helper: Object Detection
def run_object_detection(image: Image.Image):
    results = yolo_model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            coords = [int(x) for x in box.xyxy[0].tolist()]
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = result.names[cls_id]
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": coords
            })
    return detections

# Helper: VLM Audit Reasoning
def run_vlm_audit(image: Image.Image, site_id: str, checklist_rules: str):
    if not client:
        return {
            "overall_status": "WARNING",
            "audit_findings": [{"item_name": "API Key", "status": "WARNING", "finding": "VISION_API_KEY is not configured in Streamlit Secrets."}],
            "recommendations": ["Add VISION_API_KEY under Settings > Secrets in Streamlit Cloud."]
        }

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""
You are an expert site auditor and telecom equipment visual inspector.
Evaluate the uploaded image for Site ID: {site_id}.

Checklist Rules:
{checklist_rules}

Respond ONLY with a valid JSON object strictly matching this schema:
{{
  "overall_status": "PASS" | "WARNING" | "FAIL",
  "audit_findings": [
    {{
      "item_name": "Rule or component evaluated",
      "status": "PASS" | "WARNING" | "FAIL",
      "finding": "Specific observation from visual inspection"
    }}
  ],
  "recommendations": [
    "Actionable remedy or maintenance recommendation"
  ]
}}
Do not include markdown code block backticks (`json`), intro, or outro text. Return raw JSON.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.1,
        )
        raw_text = response.choices[0].message.content.strip()

        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            raw_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()

        return json.loads(raw_text)
    except Exception as e:
        return {
            "overall_status": "WARNING",
            "audit_findings": [{"item_name": "Cloud API Execution", "status": "WARNING", "finding": f"Request failed: {str(e)}"}],
            "recommendations": ["Verify API key, endpoint base URL, and provider health."]
        }

# Streamlit User Interface
st.title("🔍 Multimodal Visual Inspection System")
st.caption("AI-powered Quality Control & Site Audit running strictly in Streamlit Cloud")

st.sidebar.header("⚙️ Configuration")
site_id = st.sidebar.text_input("Site / Asset ID", value="SITE-BAG-6263")

st.sidebar.subheader("📋 Checklist / Audit Rules")
checklist_rules = st.sidebar.text_area(
    "Specify inspection criteria:",
    value=(
        "1. Check if cabinet doors are intact and shut properly.\n"
        "2. Verify status LED indicators (Green = PASS, Red/Yellow = FAIL/WARNING).\n"
        "3. Inspect cable dressing and flag any loose or burnt wires.\n"
        "4. Check for water leakage or physical corrosion."
    ),
    height=180
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Upload Site Image")
    uploaded_file = st.file_uploader("Choose an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Target Image", use_container_width=True)

with col2:
    st.subheader("📊 Inspection Findings")
    
    if uploaded_file and st.button("🚀 Run Visual Inspection", type="primary"):
        with st.spinner("Executing YOLO Object Detection & VLM Analysis..."):
            image = Image.open(uploaded_file)
            
            # Step 1: Object Detection
            detections = run_object_detection(image)
            
            # Step 2: Vision-Language Model Audit
            vlm_results = run_vlm_audit(image, site_id, checklist_rules)
            
            # Display Status Verdict
            status = vlm_results.get("overall_status", "UNKNOWN")
            if status == "PASS":
                st.success(f"### Verdict: {status}")
            elif status == "WARNING":
                st.warning(f"### Verdict: {status}")
            else:
                st.error(f"### Verdict: {status}")
            
            # Display YOLO Detections
            st.write("#### 📦 Detected Components")
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No bounding box objects detected.")
            
            # Display Audit Findings
            st.write("#### 🔎 Rule Audit Findings")
            findings = vlm_results.get("audit_findings", [])
            for item in findings:
                f_status = item.get("status", "INFO")
                icon = "✅" if f_status == "PASS" else ("⚠️" if f_status == "WARNING" else "❌")
                st.markdown(f"**{icon} {item.get('item_name')}**: {item.get('finding')}")
            
            # Display Recommendations
            st.write("#### 🛠️ Actionable Recommendations")
            recs = vlm_results.get("recommendations", [])
            for rec in recs:
                st.write(f"- {rec}")
