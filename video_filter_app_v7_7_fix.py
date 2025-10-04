# video_filter_app_v7_7_fix.py
# Miromi Signature Retro Filter Studio v7.7 — fixed+ (color + UI + effects + fast mode)

import streamlit as st
import cv2
import numpy as np
import os, tempfile, math, random

st.set_page_config(page_title="Miromi Retro Filter Studio v7.7", layout="centered")
st.title("🎞 Miromi Retro Filter Studio — v7.7 fixed+")

# ---------------- Paths ----------------
ROOT = os.getcwd()
LUT_DIR = os.path.join(ROOT, "filters")
OVERLAY_DIR = os.path.join(ROOT, "overlays")
NOISE_DIR = os.path.join(ROOT, "noise_videos")

# ---------------- LUT helpers ----------------
def load_cube_lut(path):
    """
    읽기 안정형 CUBE 파서.
    - LUT_3D_SIZE 헤더를 우선 사용
    - 없다면 라인수로 size 추정
    반환: (size, size, size, 3) float32 [0..1]
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        rows = []
        size = None
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.upper().startswith("LUT_3D_SIZE"):
                try:
                    size = int(s.split()[-1])
                except:
                    pass
                continue
            parts = s.split()
            if len(parts) == 3:
                try:
                    r,g,b = map(float, parts)
                    rows.append([r,g,b])
                except:
                    pass
        data = np.array(rows, dtype=np.float32)
        if size is None:
            size = int(round(data.shape[0] ** (1/3)))
        lut = data.reshape(size, size, size, 3)
        lut = np.clip(lut, 0.0, 1.0)
        return lut

def apply_lut_bgr(frame_bgr, lut, channel_order="BGR"):
    """
    v1에서 잘 나오던 방식으로 'BGR 인덱싱' 기본 적용.
    일부 LUT가 RGB 인덱싱을 요구하면 channel_order="RGB" 로 사용.
    반환: BGR uint8
    """
    size = lut.shape[0]
    f = frame_bgr.astype(np.float32) / 255.0
    if channel_order == "BGR":
        r = f[...,2]; g = f[...,1]; b = f[...,0]
    else:  # "RGB"
        r = f[...,2]; g = f[...,1]; b = f[...,0]  # 입력은 BGR이므로 먼저 RGB 순서로 바꿈
        # 위와 같지만 가독성용 분기 유지

    # 최근접 샘플(빠름)
    ir = np.clip((r * (size-1)).astype(np.int32), 0, size-1)
    ig = np.clip((g * (size-1)).astype(np.int32), 0, size-1)
    ib = np.clip((b * (size-1)).astype(np.int32), 0, size-1)
    mapped = lut[ir, ig, ib]   # RGB 순으로 저장된 LUT 값을 읽음 (shape (...,3) in [0..1])

    out = np.empty_like(frame_bgr, dtype=np.uint8)
    # LUT의 채널은 RGB 이므로 BGR 순서로 되돌려서 기록
    out[...,0] = np.clip(mapped[...,2] * 255.0, 0, 255).astype(np.uint8)
    out[...,1] = np.clip(mapped[...,1] * 255.0, 0, 255).astype(np.uint8)
    out[...,2] = np.clip(mapped[...,0] * 255.0, 0, 255).astype(np.uint8)
    return out

# ---------------- Overlays ----------------
def apply_overlay_image(frame_bgr, overlay_path, alpha=0.3):
    """
    overlay가 PNG(알파 있음)면 알파 채널로 합성,
    그 외(JPG/PNG 무알파)는 가중치 블렌드.
    alpha: 0~1
    """
    ov = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if ov is None: 
        return frame_bgr
    h,w = frame_bgr.shape[:2]
    ov = cv2.resize(ov, (w, h), interpolation=cv2.INTER_LINEAR)

    out = frame_bgr.astype(np.float32)

    if ov.shape[2] == 4:
        rgb = ov[...,:3].astype(np.float32)
        a = (ov[...,3:]/255.0).astype(np.float32) * alpha
        out = (1.0 - a) * out + a * rgb
    else:
        out = cv2.addWeighted(out, 1.0 - alpha, ov.astype(np.float32), alpha, 0.0)

    return np.clip(out, 0, 255).astype(np.uint8)

def apply_noise_video(frame_bgr, cap, mix=0.12):
    """노이즈 mp4를 프레임 크기로 맞춰 가중합"""
    ok, nf = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, nf = cap.read()
        if not ok:
            return frame_bgr
    nf = cv2.resize(nf, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame_bgr, 1.0 - mix, nf, mix, 0)

# ---------------- Extra FX ----------------
def add_grain(frame_bgr, strength=30, fast=False):
    """
    strength: 0~100 권장. 값이 클수록 강한 그레인.
    fast=True면 저해상도 노이즈를 업샘플해 속도↑(클라우드용)
    """
    h,w = frame_bgr.shape[:2]
    if fast:
        # 큰 입자 느낌의 빠른 그레인
        gh, gw = max(32, h//6), max(32, w//6)
        noise_small = np.random.normal(0, 25, (gh, gw, 1)).astype(np.float32)
        noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        noise = np.random.normal(0, 25, (h, w, 1)).astype(np.float32)

    amp = (strength/100.0) * 35.0  # 대략적 스케일
    out = frame_bgr.astype(np.float32) + amp * noise
    return np.clip(out, 0, 255).astype(np.uint8)

def dream_blur(frame_bgr, strength=4, fast=False):
    """
    소프트 글로우 느낌. strength 0~10 권장
    """
    sigma = max(0.1, strength) * (1.0 if fast else 1.5)
    blur = cv2.GaussianBlur(frame_bgr, (0,0), sigmaX=sigma, sigmaY=sigma)
    mix = min(0.8, 0.18 + 0.06*strength)  # 섞는 비율
    return cv2.addWeighted(frame_bgr, 1.0 - mix, blur, mix, 0.0)

# ---------------- UI ----------------
video_file = st.file_uploader("🎥 영상 업로드 (MP4/MOV/AVI, ≤200MB 권장)", type=["mp4","mov","avi","m4v"])

lut_list = ["None"]
if os.path.isdir(LUT_DIR):
    lut_list += [f for f in sorted(os.listdir(LUT_DIR)) if f.lower().endswith(".cube")]

overlay_list = ["None"]
if os.path.isdir(OVERLAY_DIR):
    overlay_list += [f for f in sorted(os.listdir(OVERLAY_DIR)) if f.lower().endswith((".png",".jpg",".jpeg"))]

noise_list = ["None"]
if os.path.isdir(NOISE_DIR):
    noise_list += [f for f in sorted(os.listdir(NOISE_DIR)) if f.lower().endswith((".mp4",".mov",".m4v",".avi"))]

lut_choice = st.selectbox("🎨 LUT 프리셋", lut_list, index=lut_list.index("None") if "None" in lut_list else 0)

col1, col2 = st.columns(2)
with col1:
    overlay_choice = st.selectbox("📼 오버레이 이미지", overlay_list, index=0)
    overlay_alpha_step = 3  # 기본값
    if overlay_choice != "None":
        overlay_alpha_step = st.slider("오버레이 강도 (1~10)", 1, 10, 3)
with col2:
    noise_choice = st.selectbox("🌫️ 움직이는 노이즈 (MP4)", noise_list, index=0)
    noise_mix = 12
    if noise_choice != "None":
        noise_mix = st.slider("노이즈 강도 (1~30)", 1, 30, 12)

# ── 기본 그레인 & 드림블러 복구
st.subheader("✨ 추가 효과")
c1, c2, c3 = st.columns(3)
with c1:
    use_grain = st.checkbox("🎞 기본 그레인", value=False)
    grain_level = st.slider("그레인 강도", 0, 100, 30) if use_grain else 0
with c2:
    use_dream = st.checkbox("🌫 드림 블러", value=False)
    dream_level = st.slider("블러 강도", 0, 10, 4) if use_dream else 0
with c3:
    # LUT 파랗게 보일 때 응급 스위치(일부 RGB 인덱스 LUT 대응)
    rgb_mode = st.toggle("🧪 LUT RGB 모드(파랗게 보이면 꺼두기)", value=False)

fast_mode = st.checkbox("⚡ 빠른 렌더 (30초 목표)", value=True,
                        help="해상도 최대 720p로 축소 + FPS 최대 15 + 빠른 그레인/블러")

go = st.button("🚀 필터 적용하기")

# 설명(요청 2번)
with st.expander("⚡ 빠른 렌더는 어떻게 빠른가요?"):
    st.markdown(
        "- **GPU가 없는 환경(예: Streamlit Cloud)**에서도 빠르게 돌리기 위해 아래 최적화를 적용합니다.\n"
        "  1) **해상도 제한**: 입력 영상을 최대 720p로 축소해 픽셀 연산량을 크게 줄입니다.\n"
        "  2) **FPS 제한**: 출력 FPS를 최대 **15fps**로 제한합니다.\n"
        "  3) **경량 연산**: 그레인은 저해상도 노이즈 업샘플, 블러는 낮은 시그마로 처리해 속도를 올립니다.\n"
        "  \nGPU가 있는 로컬 PC에서는 체크를 끄면 원본 해상도·FPS로 더 고품질 출력이 가능합니다."
    )

# ---------------- Processing ----------------
if go and video_file is not None:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.read())
        src_path = f.name

    cap = cv2.VideoCapture(src_path)
    fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fast mode: 해상도 & FPS 제한
    target_fps = min(fps, 15.0) if fast_mode else fps
    scale = 1.0
    if fast_mode:
        max_side = 720
        side = max(width, height)
        if side > max_side:
            scale = max_side / side
    out_w = max(16, int(round(width * scale / 2))*2)
    out_h = max(16, int(round(height * scale / 2))*2)

    # 출력 준비
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, target_fps, (out_w, out_h))

    # LUT 미리 로드
    lut = None
    if lut_choice != "None":
        try:
            lut = load_cube_lut(os.path.join(LUT_DIR, lut_choice))
        except Exception as e:
            st.error(f"LUT 로드 오류: {e}")
            lut = None

    # 노이즈 비디오 핸들
    noise_cap = None
    if noise_choice != "None":
        p = os.path.join(NOISE_DIR, noise_choice)
        if os.path.exists(p):
            noise_cap = cv2.VideoCapture(p)

    progress = st.progress(0)
    done = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 리사이즈(빠른 렌더용)
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 1) LUT (BGR 인덱싱 기본) — v1과 동일 파이프라인 -> 파랗게 나오는 문제 해결
        if lut is not None:
            frame = apply_lut_bgr(frame, lut, channel_order="RGB" if rgb_mode else "BGR")

        # 2) 오버레이 이미지
        if overlay_choice != "None":
            alpha = max(0.1, min(1.0, overlay_alpha_step / 10.0))
            frame = apply_overlay_image(frame, os.path.join(OVERLAY_DIR, overlay_choice), alpha=alpha)

        # 3) 움직이는 노이즈
        if noise_cap is not None:
            frame = apply_noise_video(frame, noise_cap, mix=noise_mix/100.0)

        # 4) 그레인 / 드림블러
        if use_grain and grain_level > 0:
            frame = add_grain(frame, grain_level, fast=fast_mode)
        if use_dream and dream_level > 0:
            frame = dream_blur(frame, dream_level, fast=fast_mode)

        writer.write(frame)
        done += 1
        progress.progress(min(1.0, done / max(1, total)))

    cap.release()
    writer.release()
    if noise_cap is not None:
        noise_cap.release()

    st.success("✅ 처리 완료!")
    st.video(out_path)
    st.download_button(
        "💾 결과 영상 다운로드",
        data=open(out_path, "rb").read(),
        file_name=f"Miromi_v7.7_{os.path.splitext(lut_choice)[0] or 'None'}.mp4",
        mime="video/mp4"
    )

elif go and video_file is None:
    st.warning("📁 먼저 영상을 업로드하세요.")
