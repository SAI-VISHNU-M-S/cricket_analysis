import os
import uuid
import shutil
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from openai import OpenAI

# Import local modules
from .shot_analyzer import process_video

app = FastAPI()

# Enable CORS for ngrok and mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client (Add your key here)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Pathing: backend/main.py -> backend/ -> pjt/
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"

for folder in [UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory user storage
users_db = {}

class AuthSchema(BaseModel):
    username: str
    password: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
async def register(user: AuthSchema):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    users_db[user.username] = user.password
    return {"message": "Success"}

@app.post("/login")
async def login(user: AuthSchema):
    if user.username not in users_db or users_db[user.username] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Success"}

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    output_filename = f"out_{file_id}.mp4"
    output_path = OUTPUT_DIR / output_filename
    report_path = REPORT_DIR / f"report_{file_id}.pdf"
    
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Run local biomechanics
    is_compact, feedback = await run_in_threadpool(
        process_video, str(input_path), str(output_path), str(report_path)
    )
    
    # 2. Add OpenAI Pro Coach Tip
    try:
        # feedback[0] = Shot Type, feedback[1] = Average Angle
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a pro cricket coach. Give one aggressive coaching tip based on these metrics."},
                {"role": "user", "content": f"Player played a {feedback[0]} with {feedback[1]}. Form is {'Compact' if is_compact else 'Collapsed'}."}
            ]
        )
        feedback.append(f"AI COACH: {response.choices[0].message.content}")
    except:
        feedback.append("AI COACH: Keep your head still and focus on the ball.")

    base_url = str(request.base_url).rstrip("/")
    return {
        "video_url": f"{base_url}/outputs/{output_filename}",
        "report_url": f"{base_url}/reports/report_{file_id}.pdf",
        "feedback": feedback
    }