# Multimodal Visual Inspection System

An automated quality control and visual site auditing platform powered by **FastAPI**, **Streamlit**, **YOLOv8**, and **Qwen2.5-VL / Ollama**.

## 🚀 Project Structure

```text
multimodal-visual-inspection/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI backend
│   ├── models.py              # Pydantic schemas
│   └── vision_engine.py       # YOLOv8 + Qwen2.5-VL engine
├── streamlit_app.py           # Streamlit Web UI
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Containerization setup
├── .gitignore                 # Git ignore configuration
└── README.md                  # Project documentation
```

## 📋 Quick Start

### 1. Model Setup
Ensure [Ollama](https://ollama.com) is installed and pull the Vision-Language Model:
```bash
ollama pull qwen2.5-vl
```

### 2. Local Execution
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000 &
streamlit run streamlit_app.py
```
