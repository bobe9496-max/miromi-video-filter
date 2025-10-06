# -*- coding: utf-8 -*-
import os
import io
import cv2
import math
import time
import numpy as np
import streamlit as st
import tempfile

# ─────────────────────────────────────────────────────────────
# Miromi Retro Filter — Developed by THE PLATFORM COMPANY
# 통합: LUT + 오버레이(정지) + 움직이는 노이즈 + 그레인 + 드림블러 + 빠른 렌더
# 업로더 강제 리마운트/세션초기화 + 썸네일/미리보기 + 결과 영상 인앱 재생/다운로드
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Miromi Retro Filter", layout="wide")
st.title("🎞 Miromi Retro Filter")
st.caption("Developed by THE PLATFORM COMPANY")

# 프로젝트 폴더들
ROOT = os.getcwd()
LUT_DIR = os.path.join(ROOT, "filters")
OVERLAY_DIR = os.path.join(ROOT, "overlays")
NOISE_DIR = os.path.join(ROOT, "noise_videos")

os.makedirs(LUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
def read_first_frame(path):
    """비디오 첫 프레임을 RGB ndarray로 반환 (썸네일용). 실패 시 None"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def image_thumb_from_path(path, max_w=420):
    """이미지 파일을 읽어 리사이즈된 RGB thumb 반환"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 4:  # RGBA → RGB (premultiplied X)
        alpha = img[..., 3:] / 255.0
        bg = np.ones_like(img[..., :3]) * 255
        img = (alpha * img[..., :3] + (1 - alpha) * bg).astype(np.uint8)
    img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def safe_int(val, default):
    try:
        return int(val)
    except Exception:
        return default

# ─────────────────────────────────────────────────────────────
# LUT
# ─────────────────────────────────────────────────────────────
def load_cube_lut(path):
    """
    CUBE LUT 파서: LUT_3D_SIZE 기반으로 정확히 reshape.
    값은 0~1 범위 float로 유지 (Rec.709 가정, 감마 보정 없음)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    # 사이즈 찾기
    size = None
    for l in lines:
        if l.upper().startswith("LUT_3D_SIZE"):
            size = int(l.split()[-1])
            break
    if size is None:
        # 사이즈 라인이 없다면, 데이터 길이로 추정
        data_only = [l for l in lines if len(l.split()) == 3]
        size = round(len(data_only) ** (1/3))

    # 데이터
    data = np.array([list(map(float, l.split())) for l in lines if len(l.split()) == 3], dtype=np.float32)
    lut = data.reshape(size, size, size, 3)
    # LUT 값이 0~1 또는 0~255 둘 다 존재하므로 정규화
    if lut.max() > 1.0:
        lut = lut / 255.0
    lut = np.clip(lut, 0.0, 1.0)
    return lut, size

def apply_lut_nearest_bgr(frame_bgr, lut):
    """
    기본 BGR 채널 순서 그대로 LUT 인덱싱 (Rec.709 LUT 가정, 감마 처리 없음)
    frame_bgr: uint8 BGR
    lut: (S,S,S,3) float[0~1]
    """
    size = lut.shape[0]
    # BGR → 0~(S-1) 인덱스
    b = (frame_bgr[..., 0].astype(np.float32) / 255.0) * (size - 1)
    g = (frame_bgr[..., 1].astype(np.float32) / 255.0) * (size - 1)
    r = (frame_bgr[..., 2].astype(np.float32) / 255.0) * (size - 1)
    bi = np.clip(b.round().astype(np.int32), 0, size - 1)
    gi = np.clip(g.round().astype(np.int32), 0, size - 1)
    ri = np.clip(r.round().astype(np.int32), 0, size - 1)
    mapped = lut[ri, gi, bi]  # LUT이 RGB 순서로 저장돼 있는 경우 → (R,G,B) 인덱스
    mapped = np.clip(mapped * 255.0, 0, 255).astype(np.uint8)
    # mapped는 RGB, 다시 BGR로
    return cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR)

def apply_lut_nearest_rgb(frame_bgr, lut):
    """
    RGB로 변환 후 LUT 인덱싱 (필요시 토글로 사용)
    """
    size = lut.shape[0]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    r = rgb[..., 0] * (size - 1); g = rgb[..., 1] * (size - 1); b = rgb[..., 2] * (size - 1)
    ri = np.clip(r.round().astype(np.int32), 0, size - 1)
    gi = np.clip(g.round().astype(np.int32), 0, size - 1)
    bi = np.clip(b.round().astype(np.int32), 0, size - 1)
    mapped = lut[ri, gi, bi]  # RGB 인덱싱
    mapped = np.clip(mapped * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────────────────────
# 이펙트: 오버레이(정지), 그레인, 드림블러, 노이즈(동영상)
# ─────────────────────────────────────────────────────────────
def apply_overlay_image(frame_bgr, overlay_path, strength_1_to_10=3):
    """
    정지 오버레이 이미지 합성. strength는 1~10(=0.05~0.5 가중치)로 매핑.
    PNG alpha 지원, JPG는 가중치 블렌딩.
    """
    if not overlay_path or not os.path.exists(overlay_path):
        return frame_bgr
    ov = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if ov is None:
        return frame_bgr
    ov = cv2.resize(ov, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    strength = np.clip((strength_1_to_10 / 20.0), 0.02, 0.6)  # 1→0.05, 10→0.5 근사
    base = frame_bgr.astype(np.float32)

    if ov.shape[2] == 4:
        rgb = ov[..., :3].astype(np.float32)
        alpha = (ov[..., 3:].astype(np.float32) / 255.0) * strength
        out = (1 - alpha) * base + alpha * rgb
    else:
        rgb = ov[..., :3].astype(np.float32)
        out = cv2.addWeighted(base, 1 - strength, rgb, strength, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_grain(frame_bgr, amount_0_to_100=20):
    """
    필름 그레인: 밝기 독립 가우시안 노이즈 + 약한 소프트라이트 혼합
    """
    if amount_0_to_100 <= 0:
        return frame_bgr
    a = float(amount_0_to_100) / 100.0  # 0~1
    noise = np.random.normal(loc=0.0, scale=25.0, size=frame_bgr.shape).astype(np.float32)  # σ=25
    base = frame_bgr.astype(np.float32)
    grain = np.clip(base + noise * (a * 1.2), 0, 255)
    # 소프트라이트 느낌을 살짝 섞기
    out = np.clip(0.7 * base + 0.3 * grain, 0, 255).astype(np.uint8)
    return out

def apply_dream_blur(frame_bgr, radius_0_to_20=6, mix_0_to_100=35):
    """
    드림블러: 가우시안 블러 + 스크린 블렌드 계열 합성
    """
    if radius_0_to_20 <= 0 or mix_0_to_100 <= 0:
        return frame_bgr
    r = int(radius_0_to_20)
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=max(0.5, r))
    base = frame_bgr.astype(np.float32) / 255.0
    blur = blurred.astype(np.float32) / 255.0
    # screen blend
    screen = 1 - (1 - base) * (1 - blur)
    mix = float(mix_0_to_100) / 100.0
    out = np.clip((1 - mix) * base + mix * screen, 0, 1) * 255.0
    return out.astype(np.uint8)

def blend_noise_video(frame_bgr, noise_cap, strength_0_to_100=10):
    """
    움직이는 노이즈 영상 합성 (가중치 블렌드)
    """
    if noise_cap is None or strength_0_to_100 <= 0:
        return frame_bgr
    ret, nfrm = noise_cap.read()
    if not ret:
        noise_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, nfrm = noise_cap.read()
    if not ret or nfrm is None:
        return frame_bgr
    nfrm = cv2.resize(nfrm, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = float(strength_0_to_100) / 100.0
    out = cv2.addWeighted(frame_bgr.astype(np.float32), 1 - a, nfrm.astype(np.float32), a, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────
# 사이드바: 옵션
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Processing Options")

    # 업로더 (강제 리마운트용 key 수정)
    st.subheader("📤 영상 업로드")
    ALLOWED_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
    video = st.file_uploader(
        "MP4/MOV/AVI/M4V/WEBM — 최대 1GB 권장",
        type=None,
        accept_multiple_files=False,
        key="video_uploader_v2"
    )

    # 업로더 디버그/정보
    uploaded_ok = False
    if video is not None:
        vname = getattr(video, "name", "unknown")
        vext = os.path.splitext(vname)[1].lower()
        vsize = getattr(video, "size", None)
        st.caption(f"파일: **{vname}** | 크기: **{(vsize/1e6):.1f} MB**" if isinstance(vsize, (int, float)) else f"파일: **{vname}**")
        if vext not in ALLOWED_EXTS:
            st.error("지원 포맷이 아닙니다. mp4/mov/m4v/avi/webm 중 하나로 업로드하세요.")
        else:
            uploaded_ok = True

    reset_col, _ = st.columns([1, 2])
    with reset_col:
        if st.button("업로드 초기화/새로고침", use_container_width=True):
            st.session_state.pop("video_uploader_v2", None)
            st.experimental_rerun()

    st.markdown("---")

    # LUT
    st.subheader("🎨 LUT")
    lut_list = sorted([f for f in os.listdir(LUT_DIR) if f.lower().endswith(".cube")])
    lut_choice = st.selectbox("LUT 선택", ["None"] + lut_list, index=0)
    lut_indexing_mode = st.radio("LUT 채널 인덱싱", options=["BGR(권장)", "RGB"], horizontal=True)

    st.markdown("---")

    # 오버레이(정지)
    st.subheader("📼 정지 오버레이")
    overlay_list = sorted([f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    overlay_choice = st.selectbox("오버레이 이미지", ["None"] + overlay_list, index=0)
    overlay_strength = st.slider("오버레이 강도 (1-10)", min_value=1, max_value=10, value=3, step=1)

    # 썸네일 미리보기
    if overlay_choice != "None":
        ov_path = os.path.join(OVERLAY_DIR, overlay_choice)
        ov_thumb = image_thumb_from_path(ov_path)
        if ov_thumb is not None:
            st.image(ov_thumb, caption=f"선택됨: {overlay_choice}", use_container_width=True)

    st.markdown("---")

    # 움직이는 노이즈
    st.subheader("🌫️ 움직이는 노이즈")
    noise_list = sorted([f for f in os.listdir(NOISE_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v", ".webm"))])
    noise_choice = st.selectbox("노이즈 영상", ["None"] + noise_list, index=0)
    noise_strength = st.slider("노이즈 강도 (0-100)", 0, 100, 10, 1)

    # 노이즈 썸네일
    if noise_choice != "None":
        npath = os.path.join(NOISE_DIR, noise_choice)
        nthumb = read_first_frame(npath)
        if nthumb is not None:
            st.image(nthumb, caption=f"선택됨: {noise_choice} (첫 프레임)", use_container_width=True)

    st.markdown("---")

    # 필름 그레인 / 드림블러
    st.subheader("🎛️ 추가 효과")
    grain_amount = st.slider("그레인(필름 노이즈) (0-100)", 0, 100, 20, 1)
    dream_radius = st.slider("드림블러 반경 (0-20)", 0, 20, 6, 1)
    dream_mix = st.slider("드림블러 강도 (0-100)", 0, 100, 35, 1)

    st.markdown("---")

    # 빠른 렌더
    st.subheader("⚡ 빠른 렌더 (30초 안팎 목표)")
    fast_mode = st.checkbox("활성화 (FPS 15 제한 + 선택적 720p 리사이즈)", value=True)
    downscale_720 = st.checkbox("720p로 다운스케일", value=True)

# ─────────────────────────────────────────────────────────────
# 메인 영역
# ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("🎬 처리 및 미리보기")
    run_btn = st.button("🚀 필터 적용하기", type="primary", use_container_width=True)

with right:
    st.subheader("ℹ️ 안내")
    st.markdown(
        "- LUT은 Rec.709 가정을 따르며, 감마 선형화는 하지 않습니다.\n"
        "- “BGR(권장)” 인덱싱은 OpenCV 기본 채널 순서에 맞춰 LUT을 조회합니다.\n"
        "- 빠른 렌더: FPS를 15로 제한하고(선택) 720p로 리사이즈해 처리 시간을 크게 줄입니다.\n"
        "- 결과는 페이지 내 미리보기 가능하며, 다운로드도 지원합니다."
    )

# ─────────────────────────────────────────────────────────────
# 처리 로직
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_ok:
        st.error("먼저 영상을 업로드하세요.")
        st.stop()

    # 입력 임시 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video.read())
        in_path = tmp_in.name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("업로드한 영상을 열 수 없습니다.")
        st.stop()

    # 입력 메타
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 설정
    if fast_mode:
        target_fps = min(src_fps, 15.0)
    else:
        target_fps = src_fps

    if downscale_720 and src_h > 720:
        scale = 720.0 / src_h
        out_w = int(round(src_w * scale))
        out_h = 720
    else:
        out_w, out_h = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, fourcc, target_fps, (out_w, out_h))

    # LUT 사전 로드
    lut = None
    if lut_choice != "None":
        try:
            lut, _ = load_cube_lut(os.path.join(LUT_DIR, lut_choice))
        except Exception as e:
            st.warning(f"LUT 로드 실패: {e}")
            lut = None

    # 노이즈 캡처
    noise_cap = None
    if noise_choice != "None":
        npath = os.path.join(NOISE_DIR, noise_choice)
        nc = cv2.VideoCapture(npath)
        if nc.isOpened():
            noise_cap = nc

    # 진행바
    progress = st.progress(0.0)
    info = st.empty()

    frame_idx = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 리사이즈(선택)
        if (out_w, out_h) != (src_w, src_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # LUT
        if lut is not None:
            if lut_indexing_mode.startswith("BGR"):
                frame = apply_lut_nearest_bgr(frame, lut)
            else:
                frame = apply_lut_nearest_rgb(frame, lut)

        # 정지 오버레이
        if overlay_choice != "None":
            frame = apply_overlay_image(frame, os.path.join(OVERLAY_DIR, overlay_choice), overlay_strength)

        # 움직이는 노이즈
        if noise_cap is not None and noise_strength > 0:
            frame = blend_noise_video(frame, noise_cap, noise_strength)

        # 그레인
        if grain_amount > 0:
            frame = apply_grain(frame, grain_amount)

        # 드림블러
        if dream_radius > 0 and dream_mix > 0:
            frame = apply_dream_blur(frame, dream_radius, dream_mix)

        out.write(frame)
        frame_idx += 1

        # 진행 UI
        if total_frames > 0:
            progress.progress(min(1.0, frame_idx / total_frames))
        if frame_idx % 30 == 0:
            now = time.time()
            fps_est = 30.0 / max(1e-6, (now - last_time))
            last_time = now
            info.info(f"처리 중: {frame_idx}/{total_frames} 프레임 | 추정 처리 FPS: {fps_est:.1f}")

    cap.release()
    if noise_cap is not None: noise_cap.release()
    out.release()

    # 결과 미리보기 & 다운로드
    st.success("✅ 처리 완료!")
    st.subheader("🔎 결과 미리보기")
    st.video(out_path)  # 인앱 재생

    with open(out_path, "rb") as f:
        st.download_button(
            "💾 결과 영상 다운로드",
            data=f.read(),
            file_name=f"MiromiResult_{os.path.splitext(lut_choice)[0] if lut_choice!='None' else 'original'}.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    # 임시 파일은 사용자의 다운로드 이후 주기적으로 정리하길 권장
