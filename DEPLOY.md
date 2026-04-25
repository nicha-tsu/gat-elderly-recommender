# 🚀 Deploy บน Render.com

## Pre-requisites

- บัญชี GitHub (ฟรี)
- บัญชี Render.com (ฟรี — สมัครได้ที่ https://render.com)

---

## ขั้นที่ 1 — เตรียมไฟล์ในเครื่อง

```bash
python precompute_embeddings.py
```

ตรวจว่าไฟล์เหล่านี้มีอยู่:
```
dataset/users.csv
dataset/items.csv
dataset/train.csv
dataset/test.csv
dataset/social_graph.csv
dataset/content_embeddings.npy   ← สำคัญ
checkpoints/best_model.pt        ← สำคัญ
```

---

## ขั้นที่ 2 — สร้าง GitHub Repository

1. ไปที่ https://github.com/new
2. ตั้งชื่อ repo: `gat-elderly-recommender`
3. **Public** (ฟรีบน Render ต้องเป็น public)

ใน PowerShell:

```bash
cd D:\0-Cowork\GAT_Recommender
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gat-elderly-recommender.git
git push -u origin main
```

> เปลี่ยน `YOUR_USERNAME` เป็น username ของคุณ

---

## ขั้นที่ 3 — Deploy บน Render

1. ไปที่ https://dashboard.render.com
2. กด **New +** → **Web Service**
3. เชื่อม GitHub → เลือก `gat-elderly-recommender`
4. ใช้ค่าเริ่มต้นจาก `render.yaml`:
   - Build: `pip install -r requirements-prod.txt`
   - Start: `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
   - Plan: **Free**
5. กด **Create Web Service**

⏱ รอประมาณ **5-10 นาที** สำหรับ build แรก

---

## ขั้นที่ 4 — เปิดใช้งาน

URL จะอยู่ในรูป:
```
https://gat-elderly-recommender.onrender.com
```

- API Docs: `https://YOUR_URL.onrender.com/docs`
- Mobile App: `https://YOUR_URL.onrender.com/app/`

แชร์ลิงก์ให้ผู้สูงอายุได้เลย!

---

## ⚠️ ข้อจำกัด Free Tier

| ข้อจำกัด | วิธีรับมือ |
|---------|-----------|
| RAM 512MB | ใช้ pre-computed embeddings (ทำไว้แล้ว) |
| Sleep หลังว่าง 15 นาที | โหลดครั้งแรกอาจช้า ~30 วินาที |
| ไม่มี persistent disk | history reset เมื่อ restart |

### แก้ปัญหา persistent storage (ถ้าต้องการ)

ใช้ Render's **Disk** add-on (ฟรี 1GB) หรือ external DB (Supabase ฟรี)

---

## 🔧 Local Testing ก่อน Deploy

ทดสอบว่า production setup ใช้ได้ไหม:

```bash
pip install -r requirements-prod.txt
python -m uvicorn api.server:app --port 8000
```

ถ้าไม่ error ก็ deploy ได้เลย

---

## 🐛 Debug ถ้า Deploy ล้มเหลว

ดู Logs ใน Render Dashboard:
- ถ้า "Out of memory" → ลด `NUM_USERS` ใน `config.py`
- ถ้า "Module not found" → เพิ่มใน `requirements-prod.txt`
- ถ้า "Cannot find best_model.pt" → ต้องรัน `train.py` แล้ว push ไฟล์ขึ้น git

---

## 📲 Alternative ที่ง่ายกว่า — ngrok (สำหรับทดสอบเร็วๆ)

ถ้าไม่อยากตั้งค่า cloud ใช้ ngrok แชร์ลิงก์ชั่วคราว:

```bash
# 1. ดาวน์โหลด ngrok จาก https://ngrok.com/download
# 2. รัน server ปกติ
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# 3. เปิด PowerShell ใหม่
ngrok http 8000
```

จะได้ URL ชั่วคราวเช่น `https://abc123.ngrok-free.app/app/` ใช้ได้ทันที
แต่ปิดเครื่องเมื่อไหร่ลิงก์ก็หาย
