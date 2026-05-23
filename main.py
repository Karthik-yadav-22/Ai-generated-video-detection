"""
main.py
-------
FastAPI application — 3 endpoints:
    POST /auth/signup        — register a new user
    POST /auth/login         — login and get a 24-hour JWT
    POST /video/upload       — upload a video, run all 3 FFT models, return classification

Dependencies:
    pip install fastapi uvicorn python-jose[cryptography] python-multipart argon2-cffi torch torchvision opencv-python-headless

Run:
    uvicorn main:app --reload
"""

import io
import os
import re
import tempfile
import numpy as np
import cv2
from PIL import Image
from datetime import datetime, timedelta, timezone
from typing import Annotated

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, field_validator

from db_setup import init_db, create_user, verify_user, add_video

# ── Config ────────────────────────────────────────────────────────────────────

JWT_SECRET      = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
JWT_ALGORITHM   = "HS256"
JWT_EXPIRES_HRS = 24

# ── Checkpoint paths — update these before running ───────────────────────────
CKPT_PLAIN   = os.getenv("CKPT_PLAIN",   "ckpt_plain_fft.pth")
CKPT_RGB     = os.getenv("CKPT_RGB",     "ckpt_fft_rgb.pth")
CKPT_PERCHAN = os.getenv("CKPT_PERCHAN", "ckpt_fft_perchannel.pth")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 224
_MEAN     = [0.485, 0.456, 0.406]
_STD      = [0.229, 0.224, 0.225]

ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".avi", ".mov", ".mkv"}


# ── App & DB init ─────────────────────────────────────────────────────────────

app = FastAPI(title="AI Video Detector", version="1.0.0")
init_db()


# ── Model architecture (must match training) ──────────────────────────────────

class FFTResNet18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = tv_models.resnet18(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ── Model loader (lazy — loads once on first request) ─────────────────────────

_models: dict[str, nn.Module] = {}


def _load_model(name: str, ckpt_path: str) -> nn.Module | None:
    """Load a checkpoint into FFTResNet18. Returns None if file not found."""
    if not os.path.exists(ckpt_path):
        print(f"[warn] Checkpoint not found for '{name}': {ckpt_path}")
        return None
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = FFTResNet18().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[info] Loaded model '{name}' from {ckpt_path}")
    return model


def get_models() -> dict[str, nn.Module]:
    if not _models:
        _models["plain_fft"]      = _load_model("plain_fft",      CKPT_PLAIN)
        _models["fft_rgb"]        = _load_model("fft_rgb",        CKPT_RGB)
        _models["fft_per_channel"]= _load_model("fft_per_channel",CKPT_PERCHAN)
    return _models


# ── FFT transforms (mirrors Colab notebook) ───────────────────────────────────

_normalize = transforms.Normalize(mean=_MEAN, std=_STD)


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return _normalize(torch.from_numpy(arr)).unsqueeze(0).to(DEVICE)


def _nr(a: np.ndarray) -> np.ndarray:
    a = a - a.min()
    return (a / (a.max() + 1e-8)).astype(np.float32)


def fft_to_3channel(img: np.ndarray) -> np.ndarray:
    img   = img.astype(np.float32)
    gray  = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    fft_g = np.fft.fftshift(np.fft.fft2(gray))
    fft_G = np.fft.fftshift(np.fft.fft2(img[:,:,1]))
    mag_g = np.log1p(np.abs(fft_g))
    mag_G = np.log1p(np.abs(fft_G))
    phase = (np.angle(fft_g) + np.pi) / (2 * np.pi)
    def _r(a): return cv2.resize(_nr(a), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.stack([_r(mag_g), _r(mag_G), _r(phase)], axis=0)


def rgb_fft_blend(img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    img_f = cv2.resize(img_f, (IMG_SIZE, IMG_SIZE))
    gray  = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
    fft   = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.log1p(np.abs(fft))
    mag   = (mag - mag.min()) / (mag.max() + 1e-8)
    blended = img_f.copy()
    for c in range(3):
        blended[:,:,c] = (1 - alpha) * img_f[:,:,c] + alpha * mag
    return np.clip(blended, 0, 1).transpose(2, 0, 1).astype(np.float32)


def fft_per_channel(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    fft_r = np.fft.fftshift(np.fft.fft2(img_f[:,:,0]))
    fft_g = np.fft.fftshift(np.fft.fft2(img_f[:,:,1]))
    fft_b = np.fft.fftshift(np.fft.fft2(img_f[:,:,2]))
    def _mag(fc): return np.log1p(np.abs(fc))
    def _r(a):    return cv2.resize(_nr(a), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    diff = np.angle(fft_r) - np.angle(fft_b)
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    ch2  = _r((diff + np.pi) / (2 * np.pi))
    return np.stack([_r(_mag(fft_r)), _r(_mag(fft_g)), ch2], axis=0)


# ── Frame extraction from video ───────────────────────────────────────────────

def extract_frames(video_bytes: bytes, max_frames: int = 15) -> list[np.ndarray]:
    """Write bytes to a temp file, sample up to max_frames with OpenCV."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap    = cv2.VideoCapture(tmp_path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Could not read frames from video.")
        step   = max(1, total // max_frames)
        frames = []
        idx    = 0
        while len(frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            idx += step
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames


# ── Inference — majority vote over frames ─────────────────────────────────────

@torch.no_grad()
def predict_video(frames: list[np.ndarray],
                  models: dict[str, nn.Module]) -> dict:
    """
    Run each available model over all frames, majority-vote per model,
    then ensemble-vote across models.

    Returns a dict with per-model votes and the final verdict.
    """
    transform_map = {
        "plain_fft"      : fft_to_3channel,
        "fft_rgb"        : rgb_fft_blend,
        "fft_per_channel": fft_per_channel,
    }

    results      = {}
    model_votes  = []   # 0 = real, 1 = AI per model

    for name, model in models.items():
        if model is None:
            results[name] = {"status": "checkpoint_not_loaded"}
            continue

        transform = transform_map[name]
        frame_preds = []
        for frame in frames:
            tensor = _to_tensor(transform(frame))
            pred   = model(tensor).argmax(1).item()
            frame_preds.append(pred)

        counts     = np.bincount(frame_preds, minlength=2)
        video_pred = int(np.argmax(counts))   # 0=real, 1=AI
        model_votes.append(video_pred)

        results[name] = {
            "frame_predictions" : frame_preds,
            "frames_ai"         : int(counts[1]),
            "frames_real"       : int(counts[0]),
            "video_prediction"  : "AI" if video_pred == 1 else "Real",
        }

    # ensemble: majority vote across the 3 models
    if model_votes:
        final_counts = np.bincount(model_votes, minlength=2)
        final_pred   = int(np.argmax(final_counts))
    else:
        final_pred   = 0   # default real if no model loaded

    results["ensemble"] = {
        "models_voted_ai"  : int(sum(v == 1 for v in model_votes)),
        "models_voted_real": int(sum(v == 0 for v in model_votes)),
        "final_prediction" : "AI" if final_pred == 1 else "Real",
    }
    return results, bool(final_pred)


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_access_token(user_id: int, username: str) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_HRS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload["sub"]), "username": payload["username"]}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username : str
    email    : EmailStr
    password : str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest):
    """Register a new user."""
    try:
        uid = create_user(body.username, body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return {"message": "User created successfully.", "user_id": uid}


@app.post("/auth/login")
def login(body: LoginRequest):
    """Authenticate and return a 24-hour JWT bearer token."""
    user = verify_user(body.email, body.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token = create_access_token(user["user_id"], user["username"])
    return {
        "access_token" : token,
        "token_type"   : "bearer",
        "expires_in_hrs": JWT_EXPIRES_HRS,
    }


@app.post("/video/upload")
def upload_video(
    current_user: Annotated[dict, Depends(get_current_user)],
    video: UploadFile = File(...),
):
    """
    Upload a video file, run it through all 3 FFT models, and save the result.
    Requires: Authorization: Bearer <token>
    """
    # ── validate extension ────────────────────────────────────────────────────
    ext = os.path.splitext(video.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    video_bytes = video.file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── extract frames ────────────────────────────────────────────────────────
    try:
        frames = extract_frames(video_bytes, max_frames=15)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame extraction failed: {e}")

    if not frames:
        raise HTTPException(status_code=400, detail="No readable frames found in video.")

    # ── run models ────────────────────────────────────────────────────────────
    models  = get_models()
    results, is_ai = predict_video(frames, models)

    # ── persist to DB ─────────────────────────────────────────────────────────
    vid_id = add_video(
        user_id        = current_user["user_id"],
        classification = is_ai,
        video_bytes    = video_bytes,
    )

    return {
        "vid"              : vid_id,
        "filename"         : video.filename,
        "frames_analysed"  : len(frames),
        "classification"   : "AI" if is_ai else "Real",
        "model_results"    : results,
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
