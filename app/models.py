from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    label: str = Field(..., description="Class name of detected object")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    box: List[int] = Field(..., description="Bounding box coordinates [xmin, ymin, xmax, ymax]")


class AuditFinding(BaseModel):
    item_name: str = Field(..., description="Checklist rule or component evaluated")
    status: str = Field(..., description="Status verdict: PASS, WARNING, or FAIL")
    finding: str = Field(..., description="Detailed observation from visual analysis")


class InspectionResponse(BaseModel):
    site_id: str
    overall_status: str = Field(..., description="Overall verdict: PASS, WARNING, or FAIL")
    detected_components: List[DetectionItem] = []
    audit_findings: List[AuditFinding] = []
    recommendations: List[str] = []


class InspectionRequestData(BaseModel):
    site_id: str
    checklist_rules: str
