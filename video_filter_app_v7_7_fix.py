# ======================================
# Miromi Retro Filter  (Streamlit)
# Developed by THE PLATFORM COMPANY
# ======================================
import streamlit as st
import cv2
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Miromi Retro Filter", layout="centered")
st.title("🎞 Miromi Retro Filter")
st.caption("Developed by THE PLATFORM COMPANY")

# --- Project folders ---
ROOT = os.getcwd()
LUT_DIR      = os.path.join(ROOT, "filters")
OVERLAY_DIR  = os.path.join(ROOT, "overlays")
NOISE_DIR    = os.path.join(ROOT, "noise_videos")
for d in [LUT_DIR, OVERLAY_DIR, NOISE_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- LUT loader (LUT_3D_SIZE 기반, v1 호환) ----------
def load_cube_lut(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    size = None
    for ln in lines:
        if ln.upper().startswith("LUT_3D_SIZE"):
            size = int(ln.split()[-1])
            break
    if size is None:
        raise ValueError("LUT_3D_SIZE not found in: " + path)

    triplets = []
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3:
            try:
                triplets.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except Exception:
                pass
    data = np.asarray(triplets, dtype=np.float32)
    if data.size != size * size * size * 3:
        raise ValueError(f"Cube data size mismatch. expected {size**3} rows, got {data.shape[0]}")
    lut = data.reshape(size, size, size, 3)
    if lut.max() > 1.001:  # 0..255 형태일 때 정규화
        lut = lut / 255.0
    return np.clip(lut, 0.0, 1.0).astype(np.float32)

# ---------- LUT apply (기본 RGB, BGR 옵션) ----------
def apply_lut(frame_bgr: np.ndarray, lut: np.ndarray, order: str = "RGB") -> np.ndarray:
    size = lut.shape[0]
    bgr = frame_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgr)
    ib = np.clip((b * (size - 1)).astype(np.int32), 0, size - 1)
    ig = np.clip((g * (size - 1)).astype(np.int32), 0, size - 1)
    ir = np.clip((r * (size - 1)).astype(np.int32), 0, size - 1)

    if order == "BGR":
        mapped_rgb = lut[ib, ig, ir]      # v1과 동일(B,G,R 순으로 인덱싱)
    else:  # "RGB" (여러 상용 LUT이 이쪽에 맞음)
        mapped_rgb = lut[ir, ig, ib]

    mapped_u8 = (np.clip(mapped_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(mapped_u8, cv2.COLOR_RGB2BGR)

# ---------- Static overlay (PNG/JPG, 알파 지원) ----------
def apply_overlay(frame_bgr: np.ndarray, overlay_path: str, alpha: float) -> np.ndarray:
    ov = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if ov is None:
        return frame_bgr
    ov = cv2.resize(ov, (frame_bgr.shape[1], frame_bgr.shape[0]))
    out = frame_bgr.astype(np.float32)

    if ov.shape[2] == 4:
        rgb = ov[:, :, :3].astype(np.float32)
        a   = (ov[:, :, 3:4].astype(np.float32) / 255.0) * alpha
        out = (1.0 - a) * out + a * rgb
    else:
        out = cv2.addWeighted(out, 1.0 - alpha, ov.astype(np.float32), alpha, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- Moving noise (video) ----------
def apply_moving_noise(frame_bgr: np.ndarray, noise_cap, strength: float):
    if noise_cap is None:
        return frame_bgr
    ret, nf = noise_cap.read()
    if not ret:
        noise_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, nf = noise_cap.read()
        if not ret:
            return frame_bgr
    nf = cv2.resize(nf, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return cv2.addWeighted(frame_bgr, 1.0 - strength, nf, strength, 0.0)

# ---------- Film grain (procedural) ----------
def apply_film_grain(frame_bgr: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    noise = np.random.normal(0, 25 * intensity, (h, w, 1)).astype(np.float32)
    noise = np.repeat(noise, 3, axis=2)
    out = frame_bgr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- Dream blur ----------
def apply_dream_blur(frame_bgr: np.ndarray, amount: int) -> np.ndarray:
    if amount <= 0:
        return frame_bgr
    k = amount * 2 + 1   # 홀수 커널
    blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    alpha = min(0.8, 0.06 * amount)
    out = cv2.addWeighted(frame_bgr, 1.0 - alpha, blurred, alpha, 0.0)
    return out

# ---------- Previews (overlay & noise thumbnails) ----------
@st.cache_data(show_spinner=False)
def load_overlay_preview(path: str, max_w=480):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@st.cache_data(show_spinner=False)
def load_noise_first_frame(path: str, max_w=480):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    if frame.shape[1] > max_w:
        scale = max_w / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------- UI ----------
video = st.file_uploader("🎥 영상 업로드 (MP4/MOV/AVI, ≤ 200MB)", type=["mp4", "mov", "avi"])

lut_list     = sorted([f for f in os.listdir(LUT_DIR) if f.lower().endswith(".cube")], key=str.lower)
overlay_list = sorted([f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))], key=str.lower)
noise_list   = sorted([f for f in os.listdir(NOISE_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v"))], key=str.lower)

st.subheader("🎨 LUT & 옵션")
colA, colB = st.columns(2)
with colA:
    lut_name = st.selectbox("LUT 프리셋", ["None"] + lut_list, index=0)
with colB:
    # 네 환경에 맞춰 기본값을 RGB로
    lut_order = st.radio("LUT 색상 순서", ["RGB (기본)", "BGR"], horizontal=True, index=0)

st.subheader("📼 오버레이 & 노이즈")
col1, col2 = st.columns(2)
with col1:
    overlay_name = st.selectbox("정지 오버레이 이미지", ["None"] + overlay_list, index=0)
    overlay_alpha_step = st.slider("오버레이 강도 (1–10)", 1, 10, 3)
    overlay_alpha = overlay_alpha_step / 10.0
    if overlay_name != "None":
        ov_img = load_overlay_preview(os.path.join(OVERLAY_DIR, overlay_name))
        if ov_img is not None:
            st.image(ov_img, caption=f"Overlay: {overlay_name}", use_container_width=True)

with col2:
    noise_name = st.selectbox("움직이는 노이즈(비디오)", ["None"] + noise_list, index=0)
    noise_strength_step = st.slider("노이즈 강도 (1–10)", 1, 10, 2)
    noise_strength = noise_strength_step / 10.0
    if noise_name != "None":
        nf = load_noise_first_frame(os.path.join(NOISE_DIR, noise_name))
        if nf is not None:
            st.image(nf, caption=f"Noise preview (first frame): {noise_name}", use_container_width=True)

st.subheader("✨ 추가 효과")
col3, col4 = st.columns(2)
with col3:
    grain_step = st.slider("필름 그레인 (1–10)", 1, 10, 2)
    grain_intensity = grain_step / 10.0
with col4:
    blur_step = st.slider("드림 블러 (1–10)", 1, 10, 1)

st.subheader("⚡ 빠른 렌더")
fast_mode = st.checkbox(
    "빠른 렌더 (권장)",
    value=True,
    help="FPS를 최대 15로 제한하고 프레임 샘플링으로 CPU 처리량을 낮춥니다. 해상도는 유지. (Streamlit Cloud는 GPU 없음)"
)

run = st.button("🚀 필터 적용하기")

# ---------- Processing ----------
if run and video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        src_path = tmp.name

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        st.error("비디오를 열 수 없습니다.")
    else:
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        target_fps = min(15.0, fps) if fast_mode else fps
        frame_skip = 1
        if fast_mode and fps > 15:
            frame_skip = int(round(fps / 15.0))  # 대략 15fps 샘플링

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h))

        # 준비물
        lut = None
        if lut_name != "None":
            try:
                lut = load_cube_lut(os.path.join(LUT_DIR, lut_name))
            except Exception as e:
                st.error(f"LUT 로드 실패: {e}")

        noise_cap = None
        if noise_name != "None":
            noise_cap = cv2.VideoCapture(os.path.join(NOISE_DIR, noise_name))

        prog = st.progress(0)
        done = 0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if lut is not None:
                frame = apply_lut(frame, lut, order=("RGB" if "RGB" in lut_order else "BGR"))

            if grain_intensity > 0:
                frame = apply_film_grain(frame, grain_intensity)

            if blur_step > 0:
                frame = apply_dream_blur(frame, blur_step)

            if overlay_name != "None":
                frame = apply_overlay(frame, os.path.join(OVERLAY_DIR, overlay_name), overlay_alpha)

            if noise_cap is not None and noise_strength > 0:
                frame = apply_moving_noise(frame, noise_cap, noise_strength)

            out.write(frame)
            done += 1

            if fast_mode and frame_skip > 1:
                idx += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + (frame_skip - 1))

            if total > 0:
                prog.progress(min(1.0, done / max(total, 1)))

        cap.release()
        out.release()
        if noise_cap is not None:
            noise_cap.release()

        st.success("✅ 처리 완료!")
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes, format="video/mp4", start_time=0)
        st.download_button(
            "💾 결과 영상 다운로드",
            data=video_bytes,
            file_name=f"Miromi_{(lut_name if lut_name!='None' else 'NoLUT').replace('.cube','')}.mp4",
            mime="video/mp4",
        )
elif run and video is None:
    st.warning("📁 먼저 영상을 업로드해주세요.")
