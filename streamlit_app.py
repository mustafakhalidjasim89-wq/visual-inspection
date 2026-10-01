import streamlit as st
import requests
from PIL import Image
import io
import json

st.set_page_config(
    page_title="Multimodal Visual Inspection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Visual Inspection System")
st.caption("AI-powered Quality Control & Site Audit using YOLOv8 and Qwen2.5-VL")

st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://localhost:8000/api/v1/inspect")
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
    height=150
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
        with st.spinner("Analyzing image via Object Detector & Vision-Language Model..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"site_id": site_id, "checklist_rules": checklist_rules}
                
                response = requests.post(api_url, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    res_json = response.json()
                    
                    status = res_json.get("overall_status", "UNKNOWN")
                    if status == "PASS":
                        st.success(f"### Verdict: {status}")
                    elif status == "WARNING":
                        st.warning(f"### Verdict: {status}")
                    else:
                        st.error(f"### Verdict: {status}")
                    
                    st.write("#### 📦 Detected Components")
                    detections = res_json.get("detected_components", [])
                    if detections:
                        st.dataframe(detections, use_container_width=True)
                    else:
                        st.info("No specific target bounding objects returned.")
                    
                    st.write("#### 🔎 Rule Audit Findings")
                    findings = res_json.get("audit_findings", [])
                    for item in findings:
                        f_status = item.get("status", "INFO")
                        icon = "✅" if f_status == "PASS" else ("⚠️" if f_status == "WARNING" else "❌")
                        st.markdown(f"**{icon} {item.get('item_name')}**: {item.get('finding')}")
                    
                    st.write("#### 🛠️ Actionable Recommendations")
                    recs = res_json.get("recommendations", [])
                    for rec in recs:
                        st.write(f"- {rec}")
                        
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to backend: {str(e)}")
