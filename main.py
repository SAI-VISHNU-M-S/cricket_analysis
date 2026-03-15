import os
import uuid
import shutil
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
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

# Enable CORS for cross-platform access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI Client (Replace with your actual key)
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"
TEMPLATES_DIR = BASE_DIR / "templates"

for folder in [UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Static and Template mounting
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory user storage
users_db = {}

# --- SCHEMAS ---
# Registration requires 3 fields: Username, Email, Password
class RegisterSchema(BaseModel):
    username: str = Field(..., min_length=1)
    email: str = Field(..., min_length=1) 
    password: str = Field(..., min_length=1)

# Login requires only 2 fields: Username and Password
class LoginSchema(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register")
async def register(user: RegisterSchema):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Store user data including email
    users_db[user.username] = {
        "password": user.password,
        "email": user.email
    }
    return {"message": "Success"}

@app.post("/login")
async def login(user: LoginSchema):
    stored_user = users_db.get(user.username)
    if not stored_user or stored_user["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Success"}

@app.post("/analyze")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    output_filename = f"out_{file_id}.mp4"
    output_path = OUTPUT_DIR / output_filename
    report_path = REPORT_DIR / f"report_{file_id}.pdf"
    
    # Save uploaded file
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 1. Run Local Biomechanics Analysis
    # Returns: (is_compact, [Detected Shot, Avg Elbow, Posture Status, Total Frames, local feedback])
    is_compact, feedback = await run_in_threadpool(
        process_video, str(input_path), str(output_path), str(report_path)
    )
    
    # 2. Generative AI Coaching Integration
    try:
        # feedback[0] = Shot Type, feedback[1] = Average Angle
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a pro cricket coach. Give one aggressive coaching tip based on these metrics."
                },
                {
                    "role": "user", 
                    "content": f"Player played a {feedback[0]} with {feedback[1]}. Form is {'Compact' if is_compact else 'Collapsed'}."
                }
            ]
        )
        # Tagged for frontend conditional styling
        feedback.append(f"AI COACH: {response.choices[0].message.content}")
    except Exception:
        # Fallback tip if API fails
        feedback.append("AI COACH: Keep your head still and focus on the ball.")

    base_url = str(request.base_url).rstrip("/")
    return {
        "video_url": f"{base_url}/outputs/{output_filename}",
        "report_url": f"{base_url}/reports/report_{file_id}.pdf",
        "feedback": feedback
    }