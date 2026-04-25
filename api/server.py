"""
FastAPI Backend สำหรับระบบแนะนำผู้สูงอายุ
=========================================
รัน: uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                       — Health check
  GET  /api/users              — รายชื่อผู้ใช้ทั้งหมด
  GET  /api/items              — รายการกิจกรรมทั้งหมด
  POST /api/recommend          — ขอคำแนะนำให้ผู้ใช้
  POST /api/feedback           — ส่ง QoL feedback หลังเข้าร่วมกิจกรรม
  GET  /api/explain/{uid}/{iid} — XAI: อธิบายว่าทำไมแนะนำ
  GET  /api/qol/{uid}          — สถานะ QoL ของผู้ใช้
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from data.preprocessing import load_and_build
from models.bilevel_gat import BiLevelGAT, AdaptiveFeedbackLoop
from evaluation.xai import extract_attention_weights, explain_recommendation


app = FastAPI(
    title="Elderly Adaptive Recommender API",
    description="ระบบแนะนำกิจกรรมพัฒนาทักษะปัญญาผู้สูงวัย — Bi-Level GAT",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global state (loaded at startup) ────────────────────────────────────────
class State:
    model: Optional[BiLevelGAT] = None
    graph = None
    users_df: Optional[pd.DataFrame] = None
    items_df: Optional[pd.DataFrame] = None
    feedback_loop: Optional[AdaptiveFeedbackLoop] = None
    attn_data: Optional[dict] = None
    device: str = "cpu"
    history: dict = {}     # user_id -> list of interactions
    qol_log: dict = {}     # user_id -> list of {timestamp, total, domains}


state = State()


HISTORY_FILE = Path(__file__).parent.parent / "results" / "user_history.json"


def save_history():
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    data = {
        "history": {str(k): v for k, v in state.history.items()},
        "qol_log": {str(k): v for k, v in state.qol_log.items()},
    }
    HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")


def load_history():
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            state.history = {int(k): v for k, v in data.get("history", {}).items()}
            state.qol_log = {int(k): v for k, v in data.get("qol_log", {}).items()}
        except Exception:
            state.history = {}
            state.qol_log = {}


# ── Pydantic Models ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 5

class RecommendItem(BaseModel):
    item_id: int
    title: str
    category: str
    duration_min: int
    difficulty: str
    cognitive_benefit: float
    predicted_score: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendItem]

class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    rating: int                  # 1-5
    qol_change: float            # -2 to +5

class FeedbackResponse(BaseModel):
    success: bool
    new_qol_state: List[float]
    message: str

class QoLAssessmentRequest(BaseModel):
    user_id: int
    # คะแนน 1-5 (ต่ำสุด-สูงสุด) สำหรับ 8 คำถาม
    physical_1: int       # สุขภาพกายโดยรวม
    physical_2: int       # ระดับพลังงาน
    psychological_1: int  # ความพึงพอใจในชีวิต
    psychological_2: int  # อารมณ์เชิงบวก
    social_1: int         # ความสัมพันธ์กับคนรอบข้าง
    social_2: int         # การได้รับการสนับสนุนทางสังคม
    environment_1: int    # ความปลอดภัย/ที่อยู่
    environment_2: int    # การเข้าถึงข้อมูล/กิจกรรม

class QoLAssessmentResponse(BaseModel):
    success: bool
    domain_scores: dict      # คะแนนต่อมิติ (0-100)
    total_score: float
    delta_from_baseline: dict
    message: str

class ExplanationReason(BaseModel):
    type: str
    message: str
    attention_weight: float

class ExplanationResponse(BaseModel):
    user_id: int
    item_id: int
    item_title: str
    reasons: List[ExplanationReason]


# ── Startup: load model + data ───────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    print("Loading recommender model & data...")

    train_graph, _, users_df, items_df, _, _, _ = load_and_build(Config.DATA_DIR)

    user_in_dim = train_graph["user"].x.shape[1]
    item_in_dim = train_graph["item"].x.shape[1]

    model = BiLevelGAT(
        user_in_dim=user_in_dim,
        item_in_dim=item_in_dim,
        hidden_dim=Config.GAT_HIDDEN_DIM,
        out_dim=Config.GAT_OUTPUT_DIM,
        heads=Config.GAT_HEADS,
        dropout=Config.DROPOUT,
        qol_dim=Config.QOL_DIM,
        qol_alpha=Config.QOL_ALPHA,
    )

    ckpt_path = Path(Config.MODEL_DIR) / "best_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"✓ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠ No checkpoint found — using untrained model. Run train.py first!")

    model.eval()

    state.model = model
    state.graph = train_graph
    state.users_df = users_df
    state.items_df = items_df
    state.feedback_loop = AdaptiveFeedbackLoop(
        num_users=len(users_df), qol_dim=Config.QOL_DIM
    )
    state.attn_data = extract_attention_weights(model, train_graph, device="cpu")
    load_history()
    print("✓ API ready.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Elderly Adaptive Recommender",
        "users": len(state.users_df) if state.users_df is not None else 0,
        "items": len(state.items_df) if state.items_df is not None else 0,
        "app_url": "/app/",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    """Health check endpoint สำหรับ Render"""
    return {"status": "healthy", "model_loaded": state.model is not None}


@app.get("/api/users")
def list_users(limit: int = 50):
    if state.users_df is None:
        raise HTTPException(503, "Model not loaded")
    df = state.users_df.head(limit)
    return df.to_dict(orient="records")


@app.get("/api/items")
def list_items(limit: int = 100):
    if state.items_df is None:
        raise HTTPException(503, "Model not loaded")
    return state.items_df.head(limit).to_dict(orient="records")


@app.get("/api/qol/{user_id}")
def get_qol(user_id: int):
    if state.users_df is None:
        raise HTTPException(503, "Model not loaded")
    if user_id < 0 or user_id >= len(state.users_df):
        raise HTTPException(404, f"User {user_id} not found")

    base = state.users_df.iloc[user_id]
    delta = state.feedback_loop.qol_state[user_id].numpy().tolist()

    return {
        "user_id": user_id,
        "baseline": {
            "physical": float(base["qol_physical"]),
            "psychological": float(base["qol_psychological"]),
            "social": float(base["qol_social"]),
            "environment": float(base["qol_environment"]),
        },
        "current_delta": {
            "physical": delta[0],
            "psychological": delta[1],
            "social": delta[2],
            "environment": delta[3],
        },
        "current_total": float(base[["qol_physical", "qol_psychological",
                                     "qol_social", "qol_environment"]].mean())
                          + sum(delta) / 4,
    }


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    if req.user_id < 0 or req.user_id >= len(state.users_df):
        raise HTTPException(404, "User not found")

    with torch.no_grad():
        n_items = len(state.items_df)
        n_users = len(state.users_df)

        x_dict = state.graph.x_dict
        edge_index_dict = state.graph.edge_index_dict

        all_uids = torch.arange(n_users)
        qol_context = state.feedback_loop.get_context(all_uids)

        user_emb, item_emb = state.model(x_dict, edge_index_dict, qol_context)

        u_idx = torch.tensor([req.user_id] * n_items, dtype=torch.long)
        i_idx = torch.arange(n_items, dtype=torch.long)
        scores = state.model.predict(user_emb, item_emb, u_idx, i_idx)
        scores = scores.cpu().numpy()

    top_idx = np.argsort(scores)[::-1][: req.top_k]

    recs = []
    for idx in top_idx:
        item = state.items_df.iloc[int(idx)]
        recs.append(RecommendItem(
            item_id=int(item["item_id"]),
            title=str(item["title"]),
            category=str(item["category"]),
            duration_min=int(item["duration_min"]),
            difficulty=str(item["difficulty"]),
            cognitive_benefit=round(float(item["cognitive_benefit"]), 2),
            predicted_score=round(float(scores[idx]), 3),
        ))

    return RecommendResponse(user_id=req.user_id, recommendations=recs)


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest):
    if state.feedback_loop is None:
        raise HTTPException(503, "Model not loaded")

    rating_to_qol = {1: -1.0, 2: -0.5, 3: 0.0, 4: 1.5, 5: 3.0}
    estimated_qol = rating_to_qol.get(req.rating, 0.0)
    actual_qol = req.qol_change if req.qol_change != 0 else estimated_qol

    qol_delta = torch.tensor([[actual_qol] * Config.QOL_DIM], dtype=torch.float)
    state.feedback_loop.update(torch.tensor([req.user_id]), qol_delta)

    new_state = state.feedback_loop.qol_state[req.user_id].numpy().tolist()

    # บันทึกประวัติ
    item = state.items_df.iloc[req.item_id]
    state.history.setdefault(req.user_id, []).insert(0, {
        "timestamp": pd.Timestamp.now().isoformat(),
        "item_id": req.item_id,
        "title": str(item["title"]),
        "category": str(item["category"]),
        "rating": req.rating,
        "qol_change": round(actual_qol, 2),
    })
    save_history()

    return FeedbackResponse(
        success=True,
        new_qol_state=new_state,
        message=f"บันทึกผลการเข้าร่วมกิจกรรมเรียบร้อย (Δ QoL = {actual_qol:+.2f})",
    )


@app.get("/api/profile/{user_id}")
def get_profile(user_id: int):
    """ข้อมูลโปรไฟล์ผู้ใช้ + สรุปสถิติ"""
    if state.users_df is None:
        raise HTTPException(503, "Model not loaded")
    if user_id < 0 or user_id >= len(state.users_df):
        raise HTTPException(404, "User not found")

    user = state.users_df.iloc[user_id]
    history = state.history.get(user_id, [])
    qol_log = state.qol_log.get(user_id, [])

    # สถิติประวัติ
    categories = {}
    avg_rating = 0.0
    total_qol_gain = 0.0
    if history:
        for h in history:
            categories[h["category"]] = categories.get(h["category"], 0) + 1
        avg_rating = sum(h["rating"] for h in history) / len(history)
        total_qol_gain = sum(h["qol_change"] for h in history)

    top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "-"

    return {
        "user_id": user_id,
        "age": int(user["age"]),
        "gender": str(user["gender"]),
        "tech_skill": str(user["tech_skill"]),
        "health_status": str(user["health_status"]),
        "baseline_qol": {
            "physical": float(user["qol_physical"]),
            "psychological": float(user["qol_psychological"]),
            "social": float(user["qol_social"]),
            "environment": float(user["qol_environment"]),
        },
        "stats": {
            "total_activities": len(history),
            "avg_rating": round(avg_rating, 2),
            "total_qol_gain": round(total_qol_gain, 2),
            "favorite_category": top_category,
            "categories_explored": len(categories),
            "qol_assessments": len(qol_log),
        },
    }


@app.get("/api/history/{user_id}")
def get_history(user_id: int, limit: int = 30):
    """ประวัติกิจกรรมที่เข้าร่วม"""
    history = state.history.get(user_id, [])
    return {"user_id": user_id, "count": len(history), "items": history[:limit]}


@app.get("/api/qol/trend/{user_id}")
def get_qol_trend(user_id: int):
    """แนวโน้มคะแนน QoL ตามเวลา"""
    qol_log = state.qol_log.get(user_id, [])
    return {"user_id": user_id, "count": len(qol_log), "trend": qol_log}


@app.post("/api/qol/assess", response_model=QoLAssessmentResponse)
def assess_qol(req: QoLAssessmentRequest):
    """
    ประเมิน WHOQOL-BREF (ฉบับย่อ 8 ข้อ)
    คำนวณคะแนน 4 มิติ → ส่งเข้า Adaptive Feedback Loop
    """
    if state.feedback_loop is None:
        raise HTTPException(503, "Model not loaded")

    # แปลงคะแนน 1-5 → 0-100 (สเกลมาตรฐาน WHOQOL)
    def to_100(score):
        return (score - 1) / 4 * 100

    domains = {
        "physical": (to_100(req.physical_1) + to_100(req.physical_2)) / 2,
        "psychological": (to_100(req.psychological_1) + to_100(req.psychological_2)) / 2,
        "social": (to_100(req.social_1) + to_100(req.social_2)) / 2,
        "environment": (to_100(req.environment_1) + to_100(req.environment_2)) / 2,
    }
    total = sum(domains.values()) / 4

    # คำนวณ delta จาก baseline
    base = state.users_df.iloc[req.user_id]
    baseline = {
        "physical": float(base["qol_physical"]),
        "psychological": float(base["qol_psychological"]),
        "social": float(base["qol_social"]),
        "environment": float(base["qol_environment"]),
    }
    delta = {k: round(domains[k] - baseline[k], 2) for k in domains}

    # ส่งเข้า Adaptive Feedback Loop (ใช้ delta จริงต่อมิติ ไม่ใช่ค่าเดียวกัน 4 ครั้ง)
    delta_tensor = torch.tensor(
        [[delta["physical"], delta["psychological"],
          delta["social"], delta["environment"]]],
        dtype=torch.float,
    )
    state.feedback_loop.update(torch.tensor([req.user_id]), delta_tensor)

    # บันทึก QoL log สำหรับ trend chart
    state.qol_log.setdefault(req.user_id, []).append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "total": round(total, 1),
        "domains": {k: round(v, 1) for k, v in domains.items()},
    })
    save_history()

    return QoLAssessmentResponse(
        success=True,
        domain_scores={k: round(v, 1) for k, v in domains.items()},
        total_score=round(total, 1),
        delta_from_baseline=delta,
        message=f"ประเมินคุณภาพชีวิตเรียบร้อย คะแนนรวม {total:.1f}/100",
    )


@app.get("/api/explain/{user_id}/{item_id}", response_model=ExplanationResponse)
def explain(user_id: int, item_id: int):
    if state.attn_data is None:
        raise HTTPException(503, "Attention data not available")
    if user_id >= len(state.users_df):
        raise HTTPException(404, "User not found")
    if item_id >= len(state.items_df):
        raise HTTPException(404, "Item not found")

    explanation = explain_recommendation(
        user_id=user_id, item_id=item_id,
        attention_data=state.attn_data,
        users_df=state.users_df, items_df=state.items_df, top_n=3,
    )

    item_title = str(state.items_df.iloc[item_id]["title"])

    reasons = [
        ExplanationReason(
            type=r["type"],
            message=r["message"],
            attention_weight=float(r["attention_weight"]),
        )
        for r in explanation["reasons"]
    ]

    return ExplanationResponse(
        user_id=user_id, item_id=item_id,
        item_title=item_title, reasons=reasons,
    )


# ── Serve mobile frontend ────────────────────────────────────────────────────
mobile_path = Path(__file__).parent.parent / "mobile"
if mobile_path.exists():
    app.mount("/app", StaticFiles(directory=str(mobile_path), html=True), name="mobile")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
