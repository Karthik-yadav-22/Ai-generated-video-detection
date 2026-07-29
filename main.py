"""
main.py
------------------------------------------------------------
FastAPI application for real-vs-AI video classification.
 
Endpoints
    POST /auth/signup    - register a new user
    POST /auth/login     - login, get a 24h JWT
    POST /video/upload   - upload a video, run all 5 trained
                            models, return each classification
                            plus an overall majority-vote verdict
                            (auth required)
 
FFT preprocessing (plain_fft_transform, rgb_fft_blend_transform,
perchannel_fft_transform) and the FFTResNet18 model architecture
are exact ports of karthik_s_script.ipynb, verified numerically
against the notebook's functions.

build_pujyank_resnet18() returns an exact port of pujyank.ipynb's
RGB-trained resnet18 (stock torchvision resnet18 with only .fc
replaced, returned directly with no wrapper class -- its checkpoint
uses flat keys like "conv1.weight", "fc.weight").

HybridResNet is an exact port of harshith.ipynb's dual-branch
RGB+FFT model. NOTE: its label order is {"real_videos": 0,
"ai_videos": 1} -- reversed relative to the other 4 models -- so
it uses its own CLASS_NAMES_HARSHITH list rather than CLASS_NAMES.
------------------------------------------------------------
"""
 
import os
import tempfile
from datetime import datetime, timedelta
from typing import Optional
 
import bcrypt
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session
from torchvision import models, transforms
 
from backend.dbsetup import init_db, get_db, User
 
# ----------------------------------------------------------------------
# Config — mirrors the constants from the training notebook
# ----------------------------------------------------------------------
IMG_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["ai_videos", "real_videos"]
 
FFT_LOG_SCALE = True
RGB_ALPHA = 0.5          # 0 = pure RGB, 1 = pure FFT (rgb+fft blend mode)
USE_CROSS_PHASE = True   # per-channel mode: R-mag, G-mag, cross-phase
 
NUM_FRAMES_SAMPLED = 16  # frames evenly sampled per uploaded video
 
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
#CKPT_PLAIN = os.path.join(BACKEND_DIR, "ckpt_plain_fft.pth") #remove
CKPT_PLAIN = ""#remove
CKPT_RGB = os.path.join(BACKEND_DIR, "ckpt_fft_rgb.pth")
#CKPT_PERCHAN = os.path.join(BACKEND_DIR, "ckpt_fft_perchannel.pth")
CKPT_PERCHAN = ""#remove
CKPT_PUJYANK = os.path.join(BACKEND_DIR, "resnet18_rgb_best.pth")
CKPT_HARSHITH = os.path.join(BACKEND_DIR, "best_hybrid_model_v2.pth")

# harshith.ipynb's HybridVideoDataset.CLASSES = {"real_videos": 0, "ai_videos": 1}
# -- reversed vs CLASS_NAMES above, which is {"ai_videos": 0, "real_videos": 1}
# for the other 4 models. Keep this separate rather than reordering CLASS_NAMES,
# since that would silently break the other 4 models' checkpoints.
CLASS_NAMES_HARSHITH = ["real_videos", "ai_videos"]
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# JWT config — CHANGE SECRET_KEY before deploying to production.
# Prefer setting this via an environment variable instead of hardcoding.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-only-change-me-before-deploying")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
 
app = FastAPI(title="Deepfake Video Detection API")
 
 
# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()
    load_models()
 
 
# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class SignupRequest(BaseModel):
    username: str
    password: str
 
 
class LoginRequest(BaseModel):
    username: str
    password: str
 
 
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int = ACCESS_TOKEN_EXPIRE_HOURS
 
 
# ----------------------------------------------------------------------
# Auth helpers
# ----------------------------------------------------------------------
def hash_password(password: str) -> str:
    # bcrypt has a hard 72-byte input limit — truncate defensively rather
    # than let it raise on unusually long passwords.
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")
 
 
def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
 
 
def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
 
 
def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
 
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
 
 
# ----------------------------------------------------------------------
# Auth endpoints
# ----------------------------------------------------------------------
@app.post("/auth/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
 
    user = User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created successfully", "username": user.username}
 
 
@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(user.username)
    return TokenResponse(access_token=token)
 
 
# ----------------------------------------------------------------------
# Model definitions + loading
# ----------------------------------------------------------------------
_models = {}  # populated at startup: "plain" / "rgb" / "perchannel" -> nn.Module
 
 
class FFTResNet18(nn.Module):
    """
    Matches karthik_s_script.ipynb exactly: a resnet18 backbone with a
    custom classifier head (Dropout->Linear(512,256)->ReLU->Dropout->Linear(256,2)).
    State dict keys are prefixed with "backbone." because of this wrapping.
    """
 
    def __init__(self, num_classes=2, pretrained=False, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
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
 
 
def build_resnet18() -> nn.Module:
    # pretrained=False at inference time — real weights come from the checkpoint
    return FFTResNet18(num_classes=NUM_CLASSES, pretrained=False)


def build_pujyank_resnet18() -> nn.Module:
    """
    Exact port of pujyank.ipynb's build_resnet18(): a stock torchvision
    resnet18 returned directly (not wrapped in a custom nn.Module), with
    only .fc replaced by a single Linear(512, num_classes) layer. Checkpoint
    keys are flat ("conv1.weight", "fc.weight", etc.) -- no "backbone." or
    "model." prefix, unlike FFTResNet18/HybridResNet.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class HybridResNet(nn.Module):
    """
    Exact port of harshith.ipynb's dual-branch model: two independent
    resnet18 backbones (one for RGB, one for FFT-magnitude input), each with
    its own fc replaced by nn.Identity so they output raw 512-dim features.
    The two feature vectors are concatenated and passed through a small
    classifier head. Label order for this model is reversed relative to the
    other 4 -- see CLASS_NAMES_HARSHITH above.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_branch = models.resnet18(weights=None)
        num_ftrs_rgb = self.rgb_branch.fc.in_features
        self.rgb_branch.fc = nn.Identity()

        self.fft_branch = models.resnet18(weights=None)
        num_ftrs_fft = self.fft_branch.fc.in_features
        self.fft_branch.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs_rgb + num_ftrs_fft, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb, fft):
        feat_rgb = self.rgb_branch(rgb)
        feat_fft = self.fft_branch(fft)
        return self.classifier(torch.cat((feat_rgb, feat_fft), dim=1))


def build_harshith_hybrid() -> nn.Module:
    return HybridResNet(num_classes=NUM_CLASSES)
 
 
def _load_checkpoint(model: nn.Module, ckpt_path: str) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Copy your trained .pth file into the backend/ folder with this name."
        )
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    # Handle a bare state_dict, or a dict wrapping it under a known key
    # (your checkpoints store it under "model_state", alongside "epoch"/"val_acc")
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model
 
 
def load_models():
    #_models["plain"] = _load_checkpoint(build_resnet18(), CKPT_PLAIN)
    _models["rgb"] = _load_checkpoint(build_resnet18(), CKPT_RGB)
   # _models["perchannel"] = _load_checkpoint(build_resnet18(), CKPT_PERCHAN)
    _models["pujyank_rgb"] = _load_checkpoint(build_pujyank_resnet18(), CKPT_PUJYANK)
    _models["harshith_hybrid"] = _load_checkpoint(build_harshith_hybrid(), CKPT_HARSHITH)
    print("All 3 models loaded successfully.")
 
 
# ----------------------------------------------------------------------
# FFT preprocessing — exact port of Cell 5 in karthik_s_script.ipynb
# ----------------------------------------------------------------------
# ImageNet stats, as used by _MEAN/_STD in the notebook's dataset classes
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_normalize = transforms.Normalize(mean=_MEAN, std=_STD)
 
 
def _nr(a: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Min-max normalize to [0,1] then resize — matches notebook's _nr helper."""
    a = a - a.min()
    a = a / (a.max() + 1e-8)
    return cv2.resize(a, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
 
 
def fft_to_3channel(img_rgb_np: np.ndarray, img_size: int = IMG_SIZE, log_scale: bool = FFT_LOG_SCALE) -> np.ndarray:
    """
    "Plain FFT" mode. Channels = [gray FFT magnitude, green-channel FFT magnitude, gray FFT phase].
    Returns CHW float32 array in [0,1] (pre-normalization).
    """
    img = img_rgb_np.astype(np.float32)
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    fft_g = np.fft.fftshift(np.fft.fft2(gray))
    fft_G = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
    mag_g = np.log1p(np.abs(fft_g)) if log_scale else np.abs(fft_g)
    mag_G = np.log1p(np.abs(fft_G)) if log_scale else np.abs(fft_G)
    phase = (np.angle(fft_g) + np.pi) / (2 * np.pi)
 
    return np.stack([_nr(mag_g, img_size), _nr(mag_G, img_size), _nr(phase, img_size)], axis=0)
 
 
def rgb_fft_blend(img_rgb_np: np.ndarray, img_size: int = IMG_SIZE, alpha: float = RGB_ALPHA, log_scale: bool = FFT_LOG_SCALE) -> np.ndarray:
    """
    "RGB + FFT blend" mode. Each RGB channel blended with a single luminance FFT
    magnitude map. Returns CHW float32 array in [0,1] (pre-normalization).
    """
    img = img_rgb_np.astype(np.float32) / 255.0
    img = cv2.resize(img, (img_size, img_size))
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft)) if log_scale else np.abs(fft)
    mag = (mag - mag.min()) / (mag.max() + 1e-8)
    mag = cv2.resize(mag, (img_size, img_size))
    blended = img.copy()
    for c in range(3):
        blended[:, :, c] = (1 - alpha) * img[:, :, c] + alpha * mag
    return np.clip(blended, 0, 1).transpose(2, 0, 1).astype(np.float32)
 
 
def fft_per_channel_3ch(img_rgb_np: np.ndarray, img_size: int = IMG_SIZE, log_scale: bool = FFT_LOG_SCALE, use_cross_phase: bool = USE_CROSS_PHASE) -> np.ndarray:
    """
    "Per-channel FFT" mode. Channels = [R magnitude, G magnitude, cross-phase(R,B) or B magnitude].
    Returns CHW float32 array in [0,1] (pre-normalization).
    """
    img = img_rgb_np.astype(np.float32)
    fft_r = np.fft.fftshift(np.fft.fft2(img[:, :, 0]))
    fft_g = np.fft.fftshift(np.fft.fft2(img[:, :, 1]))
    fft_b = np.fft.fftshift(np.fft.fft2(img[:, :, 2]))
 
    def _mag(fc):
        m = np.abs(fc)
        return np.log1p(m) if log_scale else m
 
    ch0 = _nr(_mag(fft_r), img_size)
    ch1 = _nr(_mag(fft_g), img_size)
    if use_cross_phase:
        diff = np.angle(fft_r) - np.angle(fft_b)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        ch2 = _nr((diff + np.pi) / (2 * np.pi), img_size)
    else:
        ch2 = _nr(_mag(fft_b), img_size)
 
    return np.stack([ch0, ch1, ch2], axis=0)
 
 
def _prep_frame_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """
    cv2 BGR frame -> resized RGB numpy array, matching the notebook's
    Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    as closely as practical for a video frame (rather than a stored file).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(pil_img)
 
 
def plain_fft_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = _prep_frame_rgb(frame_bgr)
    out = fft_to_3channel(img_rgb, IMG_SIZE, log_scale=FFT_LOG_SCALE)
    return _normalize(torch.from_numpy(out))
 
 
def rgb_fft_blend_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = _prep_frame_rgb(frame_bgr)
    out = rgb_fft_blend(img_rgb, IMG_SIZE, alpha=RGB_ALPHA, log_scale=FFT_LOG_SCALE)
    return _normalize(torch.from_numpy(out))
 
 
def perchannel_fft_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = _prep_frame_rgb(frame_bgr)
    out = fft_per_channel_3ch(img_rgb, IMG_SIZE, log_scale=FFT_LOG_SCALE, use_cross_phase=USE_CROSS_PHASE)
    return _normalize(torch.from_numpy(out))
 
 
def pujyank_rgb_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Plain RGB, no FFT -- matches pujyank.ipynb's get_val_transforms()
    (Resize -> ToTensor -> Normalize, no augmentation at inference time).
    """
    img_rgb = _prep_frame_rgb(frame_bgr)
    arr = img_rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return _normalize(torch.from_numpy(arr))


TRANSFORM_FOR_MODE = {
    "plain": plain_fft_transform,
    "rgb": rgb_fft_blend_transform,
    "perchannel": perchannel_fft_transform,
    "pujyank_rgb": pujyank_rgb_transform,
}


# ----------------------------------------------------------------------
# harshith.ipynb's dual-branch model needs two different tensors (RGB +
# FFT-magnitude) per frame fed into two separate forward-pass inputs, so it
# doesn't fit the single-input TRANSFORM_FOR_MODE / run_model_on_frames
# pattern above -- handled with its own transforms + inference function.
# ----------------------------------------------------------------------
def harshith_fft_image_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Exact port of HybridVideoDataset.to_fft_image in harshith.ipynb.
    Uses 20*log(|FFT|+eps) scaling -- NOT log1p like the other 3 models'
    FFT transforms -- normalized to 0-255 and replicated across 3 channels.
    """
    img_rgb = _prep_frame_rgb(frame_bgr)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(f_shift) + 1e-8)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([mag, mag, mag], axis=-1).astype(np.uint8)


def harshith_rgb_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = _prep_frame_rgb(frame_bgr)
    arr = img_rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return _normalize(torch.from_numpy(arr))


def harshith_fft_transform(frame_bgr: np.ndarray) -> torch.Tensor:
    fft_rgb = harshith_fft_image_from_frame(frame_bgr)
    arr = fft_rgb.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return _normalize(torch.from_numpy(arr))
 
 
# ----------------------------------------------------------------------
# Video frame sampling + inference
# ----------------------------------------------------------------------
def sample_frames(video_path: str, num_frames: int = NUM_FRAMES_SAMPLED):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Could not read frames from uploaded video")
 
    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames)).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
 
    if not frames:
        raise HTTPException(status_code=400, detail="No readable frames found in uploaded video")
    return frames
 
 
@torch.no_grad()
def run_model_on_frames(mode: str, frames) -> dict:
    model = _models[mode]
    transform_fn = TRANSFORM_FOR_MODE[mode]
 
    batch = torch.stack([transform_fn(f) for f in frames]).to(DEVICE)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0)  # average across sampled frames
 
    pred_idx = int(torch.argmax(avg_probs).item())
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(avg_probs[pred_idx]), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(avg_probs[i]), 4) for i in range(NUM_CLASSES)
        },
    }


@torch.no_grad()
def run_harshith_on_frames(frames) -> dict:
    """
    Same job as run_model_on_frames, but for HybridResNet's two-input
    forward(rgb, fft) signature, and using CLASS_NAMES_HARSHITH for the
    reversed label order.
    """
    model = _models["harshith_hybrid"]

    rgb_batch = torch.stack([harshith_rgb_transform(f) for f in frames]).to(DEVICE)
    fft_batch = torch.stack([harshith_fft_transform(f) for f in frames]).to(DEVICE)

    logits = model(rgb_batch, fft_batch)
    probs = torch.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0)

    pred_idx = int(torch.argmax(avg_probs).item())
    return {
        "prediction": CLASS_NAMES_HARSHITH[pred_idx],
        "confidence": round(float(avg_probs[pred_idx]), 4),
        "probabilities": {
            CLASS_NAMES_HARSHITH[i]: round(float(avg_probs[i]), 4) for i in range(NUM_CLASSES)
        },
    }
 
 
# ----------------------------------------------------------------------
# Video upload endpoint
# ----------------------------------------------------------------------
@app.post("/video/upload")
async def upload_video(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported video format")
 
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
 
    try:
        frames = sample_frames(tmp_path)
 
        results = {
            #"plain_fft": run_model_on_frames("plain", frames),
            "rgb_fft_blend": run_model_on_frames("rgb", frames),
            #"perchannel_fft": run_model_on_frames("perchannel", frames),
            "pujyank_rgb": run_model_on_frames("pujyank_rgb", frames),
            "harshith_hybrid": run_harshith_on_frames(frames),
        }
 
        # simple majority vote across all 5 models for an overall verdict.
        # Safe to compare "prediction" strings directly across models even
        # though harshith_hybrid uses a reversed internal label->index order
        # (CLASS_NAMES_HARSHITH) -- the actual string values ("ai_videos" /
        # "real_videos") are the same vocabulary as the other 4 models.
        votes = [r["prediction"] for r in results.values()]
        overall = max(set(votes), key=votes.count)
 
        return {
            "filename": file.filename,
            "frames_sampled": len(frames),
            "model_results": results,
            "overall_verdict": overall,
        }
    finally:
        os.remove(tmp_path)
 
 
@app.get("/")
def root():
    return {"status": "ok", "message": "Deepfake Video Detection API is running"}
 