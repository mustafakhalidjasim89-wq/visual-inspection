import io
import json
import logging
from typing import Any, Dict, List
import ollama
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger("vision_engine")


class VisionEngine:
    def __init__(self, yolo_model_path: str = "yolov8n.pt", ollama_model: str = "qwen2.5-vl"):
        self.ollama_model = ollama_model
        try:
            self.yolo_model = YOLO(yolo_model_path)
            logger.info(f"Loaded YOLOv8 model from {yolo_model_path}")
        except Exception as e:
            logger.warning(f"Could not load custom YOLO weights ({e}). Falling back to default YOLOv8n.")
            self.yolo_model = YOLO("yolov8n.pt")

    def run_object_detection(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.yolo_model(image)
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

    def run_vlm_audit(self, image: Image.Image, site_id: str, checklist_rules: str) -> Dict[str, Any]:
        img_byte_arr = io.BytesIO()
        image.convert("RGB").save(img_byte_arr, format="JPEG")
        img_bytes = img_byte_arr.getvalue()

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
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                images=[img_bytes]
            )
            raw_text = response.get("response", "").strip()
            
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"Error querying Ollama VLM: {e}")
            return {
                "overall_status": "WARNING",
                "audit_findings": [
                    {
                        "item_name": "VLM Processing",
                        "status": "WARNING",
                        "finding": f"Failed to execute vision-language reasoning: {str(e)}"
                    }
                ],
                "recommendations": ["Ensure Ollama is running and qwen2.5-vl is pulled."]
            }
