import streamlit as st
import pandas as pd
import requests
import base64
import json
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo
from openai import OpenAI
import cv2
import numpy as np
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="VS Station", page_icon="🩺", layout="centered")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]          # PythonianSt/PE2026
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
GITHUB_FILE = st.secrets.get("GITHUB_FILE", "student_registry_log.csv")

MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5.5")

def read_qr_from_image(uploaded_img):
    file_bytes = np.asarray(bytearray(uploaded_img.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)

    if data:
        return data.strip()
    return ""
    

def extract_student_id(qr_text):
    if not qr_text:
        return ""

    if "student_ID=" in qr_text:
        parsed = urlparse(qr_text)
        qs = parse_qs(parsed.query)
        return qs.get("student_ID", [""])[0]

    return qr_text.strip()

def bkk_now():
    return datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")


def get_student_id_from_url():
    return st.query_params.get("student_ID", "")


def image_to_base64(uploaded_img):
    return base64.b64encode(uploaded_img.getvalue()).decode("utf-8")


def ai_extract(image_file, mode):
    img64 = image_to_base64(image_file)

    if mode == "bp":
        instruction = """
Extract blood pressure from the device screen.
Return JSON only:
{"SBP": number or null, "DBP": number or null}
"""
    elif mode == "temp":
        instruction = """
Extract body temperature in Celsius from the device screen.
Return JSON only:
{"T": number or null}
"""
    else:
        instruction = """
Extract SpO2 percentage from the device screen.
Return JSON only:
{"SpO2": number or null}
"""

    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img64}",
                    },
                ],
            }
        ],
    )

    text = response.output_text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {}


def bp_color(sbp, dbp):
    try:
        sbp = int(sbp)
        dbp = int(dbp)
    except Exception:
        return "gray"

    if sbp >= 180 or dbp >= 120:
        return "red"
    if sbp >= 140 or dbp >= 90:
        return "red"
    if sbp >= 130 or dbp >= 80:
        return "yellow"
    if sbp < 90 or dbp < 60:
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


def spo2_color(spo2):
    try:
        spo2 = int(spo2)
    except Exception:
        return "gray"

    if spo2 < 92:
        return "red"
    if spo2 < 95:
        return "yellow"
    return "green"


def color_label(color):
    return {
        "green": "🟢 ปกติ/ยอมรับได้",
        "yellow": "🟡 ควรทวนซ้ำ/เฝ้าระวัง",
        "red": "🔴 ควรแจ้งเจ้าหน้าที่",
        "gray": "⚪ อ่านค่าไม่ได้"
    }.get(color, "⚪ อ่านค่าไม่ได้")


def github_get_file():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    params = {"ref": GITHUB_BRANCH}

    r = requests.get(url, headers=headers, params=params)

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
        "branch": GITHUB_BRANCH,
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload)
    r.raise_for_status()


def append_to_github(row):
    old_content, _ = github_get_file()

    new_row_df = pd.DataFrame([row])

    if old_content:
        old_df = pd.read_csv(StringIO(old_content), dtype=str).fillna("")

        # รวมคอลัมน์เก่า + ใหม่
        all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_row_df.columns)))

        old_df = old_df.reindex(columns=all_cols, fill_value="")
        new_row_df = new_row_df.reindex(columns=all_cols, fill_value="")

        new_df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        new_df = new_row_df

    github_save_csv(new_df)

st.title("📷 VS Station")
st.caption("Scan QR Code นักศึกษา หรือกรอก student_ID แทนได้")

if "student_ID" not in st.session_state:
    st.session_state["student_ID"] = ""

st.subheader("1) Scan QR Code นักศึกษา")

qr_img = st.camera_input(
    "เปิดกล้องมือถือเพื่อถ่าย QR Code ของนักศึกษา",
    key="qr_camera"
)

if qr_img:
    qr_text = read_qr_from_image(qr_img)

    if qr_text:
        student_id_from_qr = extract_student_id(qr_text)
        st.session_state["student_ID"] = student_id_from_qr
        st.success(f"อ่าน QR สำเร็จ: {student_id_from_qr}")
    else:
        st.error("ยังอ่าน QR ไม่ได้ กรุณาถ่ายใหม่ให้ QR ชัดและอยู่กลางภาพ")

st.subheader("หรือกรอก student_ID แทน")

manual_student_id = st.text_input(
    "กรอก student_ID หาก scan QR ไม่ได้",
    value=st.session_state["student_ID"]
)

if manual_student_id:
    st.session_state["student_ID"] = manual_student_id.strip()

if not st.session_state["student_ID"]:
    st.warning("กรุณา scan QR หรือกรอก student_ID ก่อน")
    st.stop()

student_id = st.session_state["student_ID"]

st.info(f"Student ID: {student_id}")

st.subheader("2) เลือกผู้ใช้งาน")

access_code = st.text_input(
    "กรอกรหัส 1 = นักศึกษา, 01 = เจ้าหน้าที่",
    type="password"
)

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

#student_id = st.session_state["student_ID"]
#user_type = st.session_state["user_type"]

#st.success(f"เข้าสู่หน้า VS Station: {student_id} / {user_type}")


# ---------- BP ----------
st.header("1) Blood Pressure")

bp_img = st.camera_input("ถ่ายภาพหน้าจอเครื่องวัด BP", key="bp_cam")

if bp_img:
    st.image(bp_img)

    if st.button("AI อ่านค่า BP"):
        data = ai_extract(bp_img, "bp")
        st.session_state["SBP"] = data.get("SBP")
        st.session_state["DBP"] = data.get("DBP")

    sbp = st.number_input(
        "SBP",
        min_value=0,
        max_value=300,
        value=int(st.session_state.get("SBP") or 0),
        step=1
    )

    dbp = st.number_input(
        "DBP",
        min_value=0,
        max_value=200,
        value=int(st.session_state.get("DBP") or 0),
        step=1
    )

    bp_ok = st.checkbox("ยอมรับค่า BP", key="bp_ok")

    bp_status = bp_color(sbp, dbp)
    st.markdown(f"### BP: {sbp}/{dbp} mmHg — {color_label(bp_status)}")

    if not bp_ok:
        st.stop()
else:
    st.stop()


# ---------- TEMP ----------
st.header("2) Temperature")

temp_img = st.camera_input("ถ่ายภาพหน้าจอเครื่องวัดอุณหภูมิ", key="temp_cam")

if temp_img:
    st.image(temp_img)

    if st.button("AI อ่านค่า T"):
        data = ai_extract(temp_img, "temp")
        st.session_state["T"] = data.get("T")

    temp = st.number_input(
        "T °C",
        min_value=30.0,
        max_value=45.0,
        value=float(st.session_state.get("T") or 36.5),
        step=0.1
    )

    temp_ok = st.checkbox("ยอมรับค่า T", key="temp_ok")

    temp_status = temp_color(temp)
    st.markdown(f"### T: {temp:.1f} °C — {color_label(temp_status)}")

    if not temp_ok:
        st.stop()
else:
    st.stop()


# ---------- SPO2 ----------
st.header("3) SpO2")

spo2_img = st.camera_input("ถ่ายภาพหน้าจอเครื่องวัด SpO2", key="spo2_cam")

if spo2_img:
    st.image(spo2_img)

    if st.button("AI อ่านค่า SpO2"):
        data = ai_extract(spo2_img, "spo2")
        st.session_state["SpO2"] = data.get("SpO2")

    spo2 = st.number_input(
        "SpO2 %",
        min_value=0,
        max_value=100,
        value=int(st.session_state.get("SpO2") or 0),
        step=1
    )

    spo2_ok = st.checkbox("ยอมรับค่า SpO2", key="spo2_ok")

    spo2_status = spo2_color(spo2)
    st.markdown(f"### SpO2: {spo2}% — {color_label(spo2_status)}")

    if not spo2_ok:
        st.stop()
else:
    st.stop()


# ---------- FINAL CONFIRM ----------
st.header("ยืนยันก่อนบันทึก")

timestamp = bkk_now()

summary = {
    "student_ID": student_id,
    "timestamp_BKK": bkk_now(),
    "station": "VS",
    "user_type": user_type,
    "SBP": sbp,
    "DBP": dbp,
    "BP_status": bp_status,
    "T": round(temp, 1),
    "T_status": temp_status,
    "SpO2": spo2,
    "SpO2_status": spo2_status,
}

if st.button("Save ลง GitHub CSV"):
    if not final_ok:
        st.error("กรุณาติ๊กยืนยันก่อนบันทึก")
        st.stop()

    try:
        append_to_github(summary)
        st.success("บันทึกข้อมูล VS ลง GitHub CSV พร้อม timestamp_BKK แล้ว")
    except Exception as e:
        st.error(f"บันทึก GitHub ไม่สำเร็จ: {e}")
