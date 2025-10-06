# video_filter_app_v7_7_fix.py
# Miromi Retro Filter – Developed by THE PLATFORM COMPANY
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# --------------------------
# Page / theme (UI 고정)
# --------------------------
st.set_page_config(page_title="Miromi Retro Filter", page_icon="🎞", layout="centered")
st.markdown("""
<style>
/* 본문 폭 고정 */
.block-container {max-width: 900px !important;}
/* 위·아래 여백 살짝 축소 */
.main {padding-top: 1.2rem; padding-bottom: 2rem;}
/* 버튼 라운드+굵기 */
.stButton>button {border-radius: 10px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("🎞 Miromi Retro Filter")
st.caption("Developed by THE PLATFORM COMPANY")

# --------------------------
# Paths
# --------------------------
ROOT = Path(os.getcwd())
LUT_DIR     = ROOT / "filters"
OVERLAY_DIR = ROOT / "overlays"
NOISE_DIR   = ROOT / "noise_videos"
for p in [LUT_DIR, OVERLAY_DIR, NOISE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# --------------------------
# Utils
# --------------------------
def list_files(folder: Path, exts):
    if not folder.exists():
        return []
    exts = tuple(e.lower() for e in exts)
    return sorted([f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts], key=str.lower)

def load_overlay_image(name):
    path = str(OVERLAY_DIR / name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # PNG alpha 지원
    return img

def first_frame_of_video(path: str):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---- LUT: 확실한 파서 (LUT_3D_SIZE 기반)
def load_cube_lut(path: str):
    size = None
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 헤더
            if parts[0].upper() == "LUT_3D_SIZE":
                size = int(parts[-1])
                continue
            # 데이터
            if len(parts) == 3:
                try:
                    r, g, b = map(float, parts)
                    data.append([r, g, b])
                except:
                    pass
    if size is None:
        # fallback: cube가 정방일 때 루트로 유추
        n = int(round(len(data) ** (1/3)))
        size = max(2, n)
    arr = np.array(data, dtype=np.float32)
    arr = arr.reshape((size, size, size, 3))           # [R, G, B] 순
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr  # RGB 결과표

# ---- LUT 적용: 기본은 BGR(권장). 필요 시 RGB 토글
def apply_lut_bgr(frame_bgr: np.ndarray, lut_rgb: np.ndarray, mode: str = "BGR"):
    """
    frame_bgr : 원본 BGR(0..255, uint8)
    lut_rgb   : LUT[R,G,B] -> (R,G,B)  uint8
    mode      : "BGR"(권장) | "RGB"
    """
    size = lut_rgb.shape[0]
    if mode == "BGR":
        # BGR 프레임에서 R,G,B 채널 인덱스 생성 (LUT은 RGB 인덱싱)
        r = frame_bgr[..., 2]
        g = frame_bgr[..., 1]
        b = frame_bgr[..., 0]
        idx_r = ((r.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        idx_g = ((g.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        idx_b = ((b.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        out_rgb = lut_rgb[idx_r, idx_g, idx_b]
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr
    else:
        # 프레임을 RGB로 간주하여 곧바로 인덱싱 (특수 LUT 대응)
        r = frame_bgr[..., 0]
        g = frame_bgr[..., 1]
        b = frame_bgr[..., 2]
        idx_r = ((r.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        idx_g = ((g.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        idx_b = ((b.astype(np.float32) / 255.0) * (size - 1)).astype(np.int32)
        out_rgb = lut_rgb[idx_r, idx_g, idx_b]
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def apply_static_grain(frame_bgr, amount=0.0):
    if amount <= 0: 
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    noise = np.random.normal(0, 25, (h, w, 3)).astype(np.float32)  # 기본 노이즈
    out = np.clip(frame_bgr.astype(np.float32) + noise * amount, 0, 255).astype(np.uint8)
    return out

def apply_dream_blur(frame_bgr, strength=0.0):
    if strength <= 0:
        return frame_bgr
    k = max(1, int(3 + strength * 6))  # 3~9
    if k % 2 == 0: 
        k += 1
    blur = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    # screen blend: 1 - (1-A)(1-B)
    A = frame_bgr.astype(np.float32) / 255.0
    B = blur.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - A) * (1.0 - B)
    out = (A * (1 - 0.35*strength) + screen * (0.35*strength)) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_overlay(frame_bgr, overlay, opacity=0.3):
    if overlay is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    ov = cv2.resize(overlay, (w, h))
    if ov.shape[2] == 4:  # RGBA
        rgb = ov[..., :3]
        alpha = (ov[..., 3:].astype(np.float32) / 255.0) * opacity
        base = frame_bgr.astype(np.float32)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        out = base * (1 - alpha) + rgb_bgr * alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        # no alpha → 단순 가중합
        ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
        return cv2.addWeighted(frame_bgr, 1 - opacity, ov_bgr, opacity, 0)

def blend_moving_noise(frame_bgr, noise_frame_bgr, amount=0.25):
    if noise_frame_bgr is None or amount <= 0:
        return frame_bgr
    noise = cv2.resize(noise_frame_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    # addWeighted로 살짝 얹기
    return cv2.addWeighted(frame_bgr, 1 - amount, noise, amount, 0)

# --------------------------
# Sidebar / Controls
# --------------------------
video_file = st.file_uploader("🎥 영상 업로드 (MP4/MOV/AVI · 최대 1GB)", type=["mp4", "mov", "avi"])

lut_files = list_files(LUT_DIR, [".cube"])
overlay_files = list_files(OVERLAY_DIR, [".png", ".jpg", ".jpeg"])
noise_files = list_files(NOISE_DIR, [".mp4", ".mov", ".m4v"])

st.subheader("🎨 LUT & 이펙트")
col1, col2 = st.columns(2)
with col1:
    lut_name = st.selectbox("LUT 프리셋", ["None"] + lut_files, index=0)
    lut_mode = st.radio("LUT 색상 순서", ["BGR (권장)", "RGB"], horizontal=True, index=0,
                        help="얼굴이 파랗게 보이면 BGR이 맞지 않는 경우이므로 RGB로 바꿔 확인하세요.")
with col2:
    fast_mode = st.checkbox("⚡ 빠른 렌더 (30초 이내)", value=True,
                            help="FPS를 낮추고 프레임 처리량/해상도를 약간 줄여 렌더 속도를 높입니다. GPU는 사용하지 않습니다(클라우드 제한).")

st.markdown("### 📼 오버레이 & 노이즈")
ov_col, nz_col = st.columns(2)
with ov_col:
    overlay_name = st.selectbox("정지 오버레이 이미지", ["None"] + overlay_files, index=0)
    overlay_op = st.slider("오버레이 강도", 0, 10, 3)  # 0~10
    # 선택한 오버레이 썸네일
    if overlay_name != "None":
        ov_img = load_overlay_image(overlay_name)
        if ov_img is not None:
            st.image(cv2.cvtColor(ov_img[..., :3], cv2.COLOR_BGR2RGB), caption=overlay_name, use_column_width=True)

with nz_col:
    noise_name = st.selectbox("움직이는 노이즈(비디오)", ["None"] + noise_files, index=0)
    noise_mix = st.slider("노이즈 강도", 0, 10, 2)
    # 동영상 첫 프레임 썸네일
    if noise_name != "None":
        thumb = first_frame_of_video(str(NOISE_DIR / noise_name))
        if thumb is not None:
            st.image(thumb, caption=f"{noise_name} (thumbnail)", use_column_width=True)

st.markdown("### 🎚️ 기본 효과")
c1, c2 = st.columns(2)
with c1:
    grain_amt = st.slider("그레인(정지)", 0, 10, 2)
with c2:
    blur_amt = st.slider("드림 블러", 0, 10, 1)

st.divider()
preview_sec = 3
cprev, crun = st.columns([1,1])
btn_preview = cprev.button(f"🎬 {preview_sec}초 미리보기 만들기")
btn_render  = crun.button("🚀 전체 영상 렌더링")

# --------------------------
# Core processing
# --------------------------
def process_video(src_path: str, out_path: str, full_render: bool):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # 빠른 렌더: FPS 낮추고, 약간 스케일 다운
    if fast_mode:
        target_fps = min(fps, 18)
        scale = 0.90
    else:
        target_fps = fps
        scale = 1.0

    out_w, out_h = int(width * scale), int(height * scale)
    if out_w < 2 or out_h < 2:
        out_w, out_h = width, height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, target_fps, (out_w, out_h))

    # 미리 LUT/오버레이/노이즈 준비
    lut = None
    if lut_name != "None":
        lut = load_cube_lut(str(LUT_DIR / lut_name))

    ov_img = None
    if overlay_name != "None":
        ov_img = load_overlay_image(overlay_name)

    noise_cap = None
    if noise_name != "None":
        noise_cap = cv2.VideoCapture(str(NOISE_DIR / noise_name))

    max_frames = total
    if not full_render:
        # 미리보기는 N초까지만
        max_frames = int(min(total, preview_sec * fps))

    prog = st.progress(0.0)
    done = 0

    while done < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        # 빠른 렌더: 스케일 다운
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 1) LUT
        if lut is not None:
            frame = apply_lut_bgr(frame, lut, "BGR" if "BGR" in lut_mode else "RGB")

        # 2) 정지 그레인
        if grain_amt > 0:
            frame = apply_static_grain(frame, grain_amt / 10.0)

        # 3) 드림 블러
        if blur_amt > 0:
            frame = apply_dream_blur(frame, blur_amt / 10.0)

        # 4) 정지 오버레이
        if ov_img is not None:
            frame = apply_overlay(frame, ov_img, opacity=(overlay_op / 10.0))

        # 5) 움직이는 노이즈
        if noise_cap is not None and noise_mix > 0:
            ok2, nframe = noise_cap.read()
            if not ok2 or nframe is None:
                noise_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok2, nframe = noise_cap.read()
            if ok2 and nframe is not None:
                frame = blend_moving_noise(frame, nframe, amount=noise_mix / 10.0)

        writer.write(frame)
        done += 1
        prog.progress(min(1.0, done / max_frames))

        # 빠른 렌더에서 너무 긴 영상의 경우 일정 프레임만 샘플링해 속도 증가
        if fast_mode and full_render and fps > 24:
            # 24fps로 다운샘플(간단 프레임스키핑)
            skip = int(max(0, round(fps / 24) - 1))
            if skip:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip)

    writer.release()
    cap.release()
    if noise_cap is not None:
        noise_cap.release()

def run_pipeline(full_render: bool):
    if video_file is None:
        st.warning("📁 영상을 먼저 업로드해주세요.")
        return
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_file.read())
        src = tmp_in.name
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    process_video(src, out, full_render=full_render)

    if full_render:
        st.success("✅ 렌더 완료!")
        st.video(out)  # 인앱 재생
        st.download_button("💾 결과 다운로드", open(out, "rb").read(),
                           file_name=f"Miromi_{Path(video_file.name).stem}.mp4",
                           mime="video/mp4")
    else:
        st.success("🎬 미리보기 생성 완료 (약 3초)!")
        st.video(out)

# --------------------------
# Actions
# --------------------------
if btn_preview:
    run_pipeline(full_render=False)

if btn_render:
    run_pipeline(full_render=True)
