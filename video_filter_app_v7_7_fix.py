# file: video_filter_app.py  (필요하면 기존 파일명으로 저장)
import streamlit as st
import cv2
import numpy as np
import os
import tempfile

# ======================================
# Miromi Retro Filter – Color-LUT Fix (v7.8)
# - 토글 인덱싱 버그 수정
# - 감마 선형화 제거 (Rec.709/sRGB LUT 전제)
# - 기본 BGR 조회(v1과 동일), RGB 옵션 제공
# - LUT_3D_SIZE 기반 reshape
# ======================================

st.set_page_config(page_title="Miromi Retro Filter – LUT Fix", layout="centered")
st.title("🎞 Miromi Retro Filter — LUT Color Pipeline (Fixed)")

# 프로젝트 폴더 내 LUT 디렉터리
LUT_DIR = os.path.join(os.getcwd(), "filters")
os.makedirs(LUT_DIR, exist_ok=True)

# ---------- LUT 로더: LUT_3D_SIZE 기반 ----------
def load_cube_lut(path: str) -> np.ndarray:
    """
    .cube 3D LUT 파일을 읽어서 (size, size, size, 3) float32 [0..1] 로 반환
    - 헤더에서 LUT_3D_SIZE를 찾아 명시적으로 reshape
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    # LUT_3D_SIZE 읽기 (대/소문자 혼용 대비)
    size_line = None
    for ln in lines:
        if ln.upper().startswith("LUT_3D_SIZE"):
            size_line = ln
            break
    if size_line is None:
        raise ValueError("LUT_3D_SIZE not found in cube file: " + path)

    size = int(size_line.split()[-1])

    # 3값(R G B) 라인만 수집
    triplets = []
    for ln in lines:
        # 숫자 3개 라인만
        parts = ln.split()
        if len(parts) == 3:
            try:
                triplets.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except Exception:
                pass

    data = np.asarray(triplets, dtype=np.float32)
    if data.size != size * size * size * 3:
        raise ValueError(
            f"Cube data size mismatch. expected {size**3} rows, got {data.shape[0]} rows"
        )

    lut = data.reshape(size, size, size, 3)
    # 일부 LUT이 0..255 범위인 경우를 대비해 정규화
    if lut.max() > 1.001:
        lut = lut / 255.0

    return np.clip(lut, 0.0, 1.0).astype(np.float32)

# ---------- LUT 적용: 기본 BGR 조회, RGB 옵션 ----------
def apply_lut(frame_bgr: np.ndarray, lut: np.ndarray, order: str = "BGR") -> np.ndarray:
    """
    frame_bgr : OpenCV 프레임(BGR, uint8)
    order     : "BGR" (v1과 동일, 기본) / "RGB" (표준 RGB LUT)
    ※ 감마 선형화/재감마 과정 제거 (Rec.709/sRGB LUT 전제)
    """
    size = lut.shape[0]
    # 0..1 로 정규화
    bgr = frame_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(bgr)

    ib = np.clip((b * (size - 1)).astype(np.int32), 0, size - 1)
    ig = np.clip((g * (size - 1)).astype(np.int32), 0, size - 1)
    ir = np.clip((r * (size - 1)).astype(np.int32), 0, size - 1)

    # 채널 순서에 따라 LUT 인덱싱
    if order == "RGB":
        mapped_rgb = lut[ir, ig, ib]      # LUT이 진짜 RGB 기준일 때
    else:  # "BGR" ← v1과 동일 동작(기본값)
        mapped_rgb = lut[ib, ig, ir]

    # 결과는 RGB → 저장/표시는 BGR
    mapped_rgb = np.clip(mapped_rgb, 0.0, 1.0)
    mapped_u8 = (mapped_rgb * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(mapped_u8, cv2.COLOR_RGB2BGR)

# ---------- UI ----------
video_file = st.file_uploader("🎥 영상 업로드 (MP4/MOV/AVI)", type=["mp4", "mov", "avi"])
lut_files = sorted([f for f in os.listdir(LUT_DIR) if f.lower().endswith(".cube")])

lut_choice = st.selectbox("🎨 LUT 프리셋 선택", ["None"] + lut_files, index=0)

# 토글(라디오) → 기본 BGR(v1과 동일), 필요 시 RGB
order = st.radio(
    "🧭 LUT 채널 순서",
    options=["BGR (v1 호환, 기본)", "RGB (표준)"],
    index=0,
    horizontal=True,
)
order_key = "BGR" if order.startswith("BGR") else "RGB"

process_button = st.button("🚀 필터 적용하기")

# ---------- 처리 ----------
if process_button and video_file is not None:
    # 입력 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_file.read())
        in_path = tmp_in.name

    # 비디오 읽기/쓰기 준비
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("영상을 열 수 없습니다.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # LUT 로드 (선택되었을 때만)
        lut = None
        if lut_choice != "None":
            try:
                lut = load_cube_lut(os.path.join(LUT_DIR, lut_choice))
            except Exception as e:
                st.error(f"LUT 로드 실패: {e}")
                lut = None

        prog = st.progress(0)
        done = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if lut is not None:
                frame = apply_lut(frame, lut, order=order_key)

            out.write(frame)
            done += 1
            if total > 0:
                prog.progress(min(done / total, 1.0))

        cap.release()
        out.release()

        st.success("✅ 처리 완료!")
        st.video(out_path)
        st.download_button(
            "💾 결과 영상 다운로드",
            data=open(out_path, "rb").read(),
            file_name=f"Miromi_LUT_{(lut_choice if lut_choice!='None' else 'none').replace('.cube','')}_{order_key}.mp4",
            mime="video/mp4",
        )

elif process_button and video_file is None:
    st.warning("📁 먼저 영상을 업로드하세요.")
