import io
import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.models import InspectionResponse
from app.vision_engine import VisionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Multimodal Visual Inspection API",
    version="1.0.0",
    description="FastAPI Backend for YOLOv8 & Qwen2.5-VL automated site visual inspections."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
vision_engine = VisionEngine()


@app.get("/")
def health_check():
    return {"status": "online", "system": "Multimodal Visual Inspection Backend"}


@app.post("/api/v1/inspect", response_model=InspectionResponse)
async def inspect_site_image(
    file: UploadFile = File(...),
    site_id: str = Form(...),
    checklist_rules: str = Form(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a JPEG or PNG image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # 1. Bounding box detections via YOLOv8
        detections = vision_engine.run_object_detection(image)

        # 2. Visual-Language model assessment via Qwen2.5-VL
        vlm_results = vision_engine.run_vlm_audit(image, site_id, checklist_rules)

        return InspectionResponse(
            site_id=site_id,
            overall_status=vlm_results.get("overall_status", "UNKNOWN"),
            detected_components=detections,
            audit_findings=vlm_results.get("audit_findings", []),
            recommendations=vlm_results.get("recommendations", [])
        )

    except Exception as e:
        logger.error(f"Inspection processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
