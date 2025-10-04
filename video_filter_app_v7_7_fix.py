import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# ======================================
# Miromi Signature Retro Filter Studio v7.7 (Color Pipeline Fix)
# ======================================

st.set_page_config(page_title="Miromi Retro Filter Studio v7.7", layout="centered")
st.title("🎞 Miromi Retro Filter Studio v7.7 — Fixed Color Pipeline")

LUT_DIR = os.path.join(os.getcwd(), "filters")
OVERLAY_DIR = os.path.join(os.getcwd(), "overlays")
NOISE_DIR = os.path.join(os.getcwd(), "noise_videos")

def load_lut(path):
    """3D LUT (.cube) 파일 로드"""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    data = np.array([list(map(float, l.split())) for l in lines if len(l.split()) == 3])
    size = int(round(data.shape[0] ** (1 / 3)))
    lut = data.reshape((size, size, size, 3))
    return np.clip(lut, 0, 1)

def apply_lut(frame, lut):
    """LUT 적용 - 색 반전 방지 파이프라인"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
    frame = frame.astype(np.float32) / 255.0
    frame = np.power(frame, 1/2.2)  # gamma to linear
    size = lut.shape[0]
    idx = (frame * (size - 1)).astype(np.int32)
    mapped = lut[idx[..., 0], idx[..., 1], idx[..., 2]]
    mapped = np.power(mapped, 2.2)  # linear → gamma
    mapped = np.clip(mapped * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR)

def apply_overlay(frame, overlay_path, alpha=0.3):
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return frame
    overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
    if overlay.shape[2] == 4:
        alpha_mask = overlay[..., 3:] / 255.0
        overlay = overlay[..., :3]
        frame = (1 - alpha_mask) * frame + alpha_mask * overlay
    else:
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame.astype(np.uint8)

def apply_noise(frame, noise_cap):
    ret, noise_frame = noise_cap.read()
    if not ret:
        noise_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, noise_frame = noise_cap.read()
    noise_frame = cv2.resize(noise_frame, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.9, noise_frame, 0.1, 0)

video_file = st.file_uploader("🎥 영상 업로드", type=["mp4", "mov", "avi"])
lut_files = [f for f in os.listdir(LUT_DIR) if f.endswith(".cube")] if os.path.exists(LUT_DIR) else []
overlay_files = [f for f in os.listdir(OVERLAY_DIR) if f.lower().endswith(('.png', '.jpg'))] if os.path.exists(OVERLAY_DIR) else []
noise_files = [f for f in os.listdir(NOISE_DIR) if f.lower().endswith(('.mp4', '.mov'))] if os.path.exists(NOISE_DIR) else []

lut_choice = st.selectbox("🎨 LUT 프리셋", ["None"] + lut_files)
overlay_choice = st.selectbox("📼 오버레이 이미지", ["None"] + overlay_files)
noise_choice = st.selectbox("🌫️ 움직이는 노이즈", ["None"] + noise_files)
fast_mode = st.checkbox("⚡ 빠른 렌더링 (30초 이내)")

process_button = st.button("🚀 필터 적용하기")

if process_button and video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(video_file.read())
        temp_input_path = temp_input.name

    cap = cv2.VideoCapture(temp_input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fast_mode:
        target_fps = min(fps, 15)
    else:
        target_fps = fps

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (width, height))

    if lut_choice != "None":
        lut = load_lut(os.path.join(LUT_DIR, lut_choice))
    else:
        lut = None

    if noise_choice != "None":
        noise_cap = cv2.VideoCapture(os.path.join(NOISE_DIR, noise_choice))
    else:
        noise_cap = None

    progress = st.progress(0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if lut is not None:
            frame = apply_lut(frame, lut)

        if overlay_choice != "None":
            frame = apply_overlay(frame, os.path.join(OVERLAY_DIR, overlay_choice), alpha=0.3)

        if noise_cap is not None:
            frame = apply_noise(frame, noise_cap)

        out.write(frame)
        frame_idx += 1
        progress.progress(frame_idx / total_frames)

    cap.release()
    out.release()
    if noise_cap: noise_cap.release()

    st.success("✅ LUT + Overlay + Noise 적용 완료!")
    st.video(out_path)
    st.download_button("💾 결과 영상 다운로드", open(out_path, "rb").read(),
                       file_name=f"Miromi_v7.7_{lut_choice.replace('.cube','')}.mp4", mime="video/mp4")

elif process_button:
    st.warning("⚠️ 영상을 먼저 업로드하세요.")
