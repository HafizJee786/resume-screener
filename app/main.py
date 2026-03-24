from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pdfplumber
import pandas as pd
import io

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run on startup
    import os
    if not os.path.exists('models/resume_model.pkl'):
        print("Training model on startup...")
        from app.model import train_model
        train_model()
    yield

from app.model import get_match_score, predict_category
from app.utils import extract_skills

# ── App Setup ─────────────────────────────────────────────
app = FastAPI(
    title="Resume Screener API",
    description="AI-powered resume screening and ranking system",
    version="1.0.0",
    lifespan=lifespan
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── CORS Middleware ───────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: Extract text from PDF ─────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# ── Helper: Generate recommendation label ─────────────────
def get_recommendation(score: float) -> str:
    if score >= 60:
        return "Strong Match ✅"
    elif score >= 35:
        return "Moderate Match 🟡"
    else:
        return "Weak Match ❌"


# ── Route: Health Check ───────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "API is running ✅", "version": "1.0.0"}


# ── Route: Screen Single Resume ───────────────────────────
@app.post("/screen")
async def screen_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Read PDF
        file_bytes = await resume.read()
        resume_text = extract_text_from_pdf(file_bytes)

        if not resume_text:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from PDF"}
            )

        # Get match score and category
        match_score = get_match_score(resume_text, job_description)
        category = predict_category(resume_text)
        skills = extract_skills(resume_text)

        return {
            "filename": resume.filename,
            "match_score": match_score,
            "predicted_category": category,
            "skills_found": skills,
            "recommendation": get_recommendation(match_score)
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# ── Route: Bulk Screen Resumes ────────────────────────────
@app.post("/bulk-screen")
async def bulk_screen(
    resumes: list[UploadFile] = File(...),
    job_description: str = Form(...)
):
    try:
        results = []

        for resume in resumes:
            file_bytes = await resume.read()
            resume_text = extract_text_from_pdf(file_bytes)

            if not resume_text:
                results.append({
                    "filename": resume.filename,
                    "error": "Could not extract text from PDF"
                })
                continue

            match_score = get_match_score(resume_text, job_description)
            category = predict_category(resume_text)
            skills = extract_skills(resume_text)

            results.append({
                "filename": resume.filename,
                "match_score": match_score,
                "predicted_category": category,
                "skills_found": skills,
                "recommendation": get_recommendation(match_score)
            })

        # Sort by match score highest first
        results = sorted(results, key=lambda x: x.get("match_score", 0), reverse=True)

        return {
            "total_resumes": len(results),
            "job_description_preview": job_description[:100] + "...",
            "results": results
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

# @app.get("/")
# async def home(request: Request):
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request}
#     )


