import streamlit as st
import pandas as pd
import requests
import base64
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="VS Station", page_icon="🩺", layout="centered")

# ── Minimal white / no-colour background ──────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #ffffff; }
</style>
""", unsafe_allow_html=True)

GITHUB_TOKEN  = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO   = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GITHUB_FILE   = st.secrets.get("GITHUB_FILE",   "student_registry_log.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def bkk_now():
    return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")


def read_qr_from_image(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)
    return data.strip() if data else ""


def extract_student_id(qr_text):
    if not qr_text:
        return ""
    if "student_ID=" in qr_text:
        parsed = urlparse(qr_text)
        qs = parse_qs(parsed.query)
        return qs.get("student_ID", [""])[0]
    return qr_text.strip()


# ── Colour / label helpers ────────────────────────────────────────────────────

def bp_color(sbp, dbp):
    try:
        sbp, dbp = int(sbp), int(dbp)
    except Exception:
        return "gray"
    if sbp >= 140 or dbp >= 90:
        return "red"
    if sbp >= 130 or dbp >= 80 or sbp < 90 or dbp < 60:
        return "yellow"
    return "green"


def temp_color(t):
    try:
        t = float(t)
    except Exception:
        return "gray"
    if t >= 38.0 or t < 35.0:
        return "red"
    if t >= 37.5:
        return "yellow"
    return "green"


def spo2_color(v):
    try:
        v = int(v)
    except Exception:
        return "gray"
    if v < 92:
        return "red"
    if v < 95:
        return "yellow"
    return "green"


def pr_color(pr):
    try:
        pr = int(pr)
    except Exception:
        return "gray"
    if pr < 50 or pr > 120:
        return "red"
    if pr < 60 or pr > 100:
        return "yellow"
    return "green"


def bmi_color(bmi):
    try:
        bmi = float(bmi)
    except Exception:
        return "gray"
    if bmi < 18.5 or bmi >= 30.0:
        return "red"
    if bmi >= 25.0:
        return "yellow"
    return "green"


LABELS = {
    "green":  "🟢 ปกติ",
    "yellow": "🟡 เฝ้าระวัง",
    "red":    "🔴 แจ้งเจ้าหน้าที่",
    "gray":   "⚪ อ่านค่าไม่ได้",
}

def label(color):
    return LABELS.get(color, "⚪ อ่านค่าไม่ได้")


# ── GitHub helpers ────────────────────────────────────────────────────────────

def github_get_file():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH})
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8-sig")
    return content, data["sha"]


def github_save_csv(df):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    _, sha = github_get_file()
    csv_text = df.to_csv(index=False, encoding="utf-8-sig")
    encoded = base64.b64encode(csv_text.encode("utf-8-sig")).decode("utf-8")
    payload = {
        "message": f"Append VS station data {bkk_now()}",
        "content": encoded,
        "branch":  GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, json=payload)
    r.raise_for_status()


def append_to_github(row):
    old_content, _ = github_get_file()
    new_row_df = pd.DataFrame([row])
    if old_content:
        old_df   = pd.read_csv(StringIO(old_content), dtype=str).fillna("")
        all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_row_df.columns)))
        old_df      = old_df.reindex(columns=all_cols, fill_value="")
        new_row_df  = new_row_df.reindex(columns=all_cols, fill_value="")
        new_df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        new_df = new_row_df
    github_save_csv(new_df)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🩺 VS Station")
st.caption("Scan QR Code นักศึกษา หรือกรอก student_ID แทนได้")

# ── Session-state defaults ────────────────────────────────────────────────────
for k, v in {
    "student_ID": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── 1) QR / Student ID ────────────────────────────────────────────────────────
st.subheader("1) Scan QR Code นักศึกษา")

qr_img = st.camera_input("ถ่าย QR Code ของนักศึกษา (กล้องเปิดทันที)", key="qr_camera")

if qr_img:
    qr_text = read_qr_from_image(qr_img)
    if qr_text:
        sid = extract_student_id(qr_text)
        st.session_state["student_ID"] = sid
        st.success(f"อ่าน QR สำเร็จ: {sid}")
    else:
        st.error("ยังอ่าน QR ไม่ได้ กรุณาถ่ายใหม่ให้ QR ชัดและอยู่กลางภาพ")

st.subheader("หรือกรอก student_ID แทน")
manual_id = st.text_input("กรอก student_ID หาก scan QR ไม่ได้",
                          value=st.session_state["student_ID"])
if manual_id:
    st.session_state["student_ID"] = manual_id.strip()

if not st.session_state["student_ID"]:
    st.warning("กรุณา scan QR หรือกรอก student_ID ก่อน")
    st.stop()

student_id = st.session_state["student_ID"]
st.info(f"Student ID: {student_id}")


# ── 2) Access code ────────────────────────────────────────────────────────────
st.subheader("2) เลือกผู้ใช้งาน")
access_code = st.text_input("กรอกรหัส  1 = นักศึกษา / 01 = เจ้าหน้าที่",
                             type="password")
if access_code == "1":
    user_type = "student"
    st.success("เข้าสู่โหมดนักศึกษา")
elif access_code == "01":
    user_type = "staff"
    st.success("เข้าสู่โหมดเจ้าหน้าที่")
else:
    if access_code:
        st.error("รหัสไม่ถูกต้อง")
    st.stop()


# ── 3) Blood Pressure ─────────────────────────────────────────────────────────
st.header("1) Blood Pressure")
col_sbp, col_dbp = st.columns(2)
with col_sbp:
    sbp = st.number_input(
        "SBP (mmHg)", min_value=40, max_value=300,
        value=None, step=1, placeholder="เช่น 120"
    )
with col_dbp:
    dbp = st.number_input(
        "DBP (mmHg)", min_value=20, max_value=200,
        value=None, step=1, placeholder="เช่น 80"
    )

bp_status = bp_color(sbp, dbp)
if sbp is not None and dbp is not None:
    st.markdown(f"### BP: {sbp}/{dbp} mmHg — {label(bp_status)}")


# ── 4) Temperature ────────────────────────────────────────────────────────────
st.header("2) Temperature")
temp = st.number_input(
    "T (°C)", min_value=30.0, max_value=45.0,
    value=None, step=0.1, format="%.1f", placeholder="เช่น 36.5"
)
temp_status = temp_color(temp)
if temp is not None:
    st.markdown(f"### T: {temp:.1f} °C — {label(temp_status)}")


# ── 5) SpO2 + Pulse Rate ──────────────────────────────────────────────────────
st.header("3) SpO2 & Pulse Rate")
col1, col2 = st.columns(2)
with col1:
    spo2 = st.number_input(
        "SpO2 (%)", min_value=50, max_value=100,
        value=None, step=1, placeholder="เช่น 98"
    )
with col2:
    pr = st.number_input(
        "Pulse Rate (bpm)", min_value=30, max_value=250,
        value=None, step=1, placeholder="เช่น 80"
    )

spo2_status = spo2_color(spo2)
pr_status   = pr_color(pr)
if spo2 is not None:
    st.markdown(f"### SpO2: {spo2}% — {label(spo2_status)}")
if pr is not None:
    st.markdown(f"### PR: {pr} bpm — {label(pr_status)}")


# ── 6) Body Weight, Height, BMI ───────────────────────────────────────────────
st.header("4) Body Weight & Height")

col_bw, col_ht = st.columns(2)
with col_bw:
    bw = st.number_input("น้ำหนัก BW (kg)", min_value=1.0, max_value=300.0,
                          value=60.0, step=0.1)
with col_ht:
    ht = st.number_input("ส่วนสูง Ht (cm)", min_value=50.0, max_value=250.0,
                          value=165.0, step=0.5)

if ht > 0:
    bmi = bw / ((ht / 100) ** 2)
    bmi_status = bmi_color(bmi)

    bmi_cat = (
        "ผอม (Underweight)"   if bmi < 18.5 else
        "ปกติ (Normal)"        if bmi < 25.0 else
        "น้ำหนักเกิน (Overweight)" if bmi < 30.0 else
        "อ้วน (Obese)"
    )
    st.markdown(f"### BMI: {bmi:.1f} kg/m² — {bmi_cat} — {label(bmi_status)}")
else:
    bmi = None
    bmi_status = "gray"
    st.warning("กรุณากรอกส่วนสูง")


# ── 7) Final confirm & save ───────────────────────────────────────────────────
st.header("ยืนยันก่อนบันทึก")

missing_vs = []
if sbp is None:
    missing_vs.append("SBP")
if dbp is None:
    missing_vs.append("DBP")
if temp is None:
    missing_vs.append("T")
if spo2 is None:
    missing_vs.append("SpO2")
if pr is None:
    missing_vs.append("PR")

if missing_vs:
    st.warning("กรุณากรอกข้อมูล VS ให้ครบ: " + ", ".join(missing_vs))
    st.stop()

summary = {
    "student_ID":    student_id,
    "timestamp_BKK": bkk_now(),
    "station":       "VS",
    "user_type":     user_type,
    "SBP":           sbp,
    "DBP":           dbp,
    "BP_status":     bp_status,
    "T":             round(temp, 1),
    "T_status":      temp_status,
    "SpO2":          spo2,
    "SpO2_status":   spo2_status,
    "PR":            pr,
    "PR_status":     pr_status,
    "BW":            round(bw, 1),
    "Ht":            round(ht, 1),
    "BMI":           round(bmi, 1) if bmi else "",
    "BMI_status":    bmi_status,
}

st.dataframe(pd.DataFrame([summary]))

final_ok = st.checkbox("ยืนยันว่าข้อมูลทั้งหมดถูกต้อง")

if st.button("Save ลง GitHub CSV"):
    if not final_ok:
        st.error("กรุณาติ๊กยืนยันก่อนบันทึก")
        st.stop()
    try:
        append_to_github(summary)
        st.success("บันทึกข้อมูล VS ลง GitHub CSV พร้อม timestamp_BKK แล้ว ✅")
    except Exception as e:
        st.error(f"บันทึก GitHub ไม่สำเร็จ: {e}")

