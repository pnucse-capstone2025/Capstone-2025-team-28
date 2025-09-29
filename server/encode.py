"""
실행순서
python3 encode.py 로 실행. F5 말고, 터미널로 하기
cd server
python3 -m uvicorn main:app --reload
Netflix 폴더에서 npm run dev 실행
"""

import os
import subprocess
import joblib
import numpy as np
import re
from math import ceil, floor
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from mvextractor.videocap import VideoCap
import numpy as np
import os
import subprocess
import joblib
from collections import defaultdict
from math import ceil, floor
import re
import cv2
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
import warnings
warnings.filterwarnings('ignore')

INPUT_VIDEOS = [
    ("lol_encode13", "../input/lol.mp4"),
    ("husky_encode13", "../input/husky.mp4"),
    ("news_encode13", "../input/news.mp4")
]

CRF_MODEL_PATH = "./model/v10_crf_model.pkl"
MAXRATE_MODEL_PATH = "./model/v10_max_model.pkl"
SCALER_X_PATH = "./model/v10_scaler_X.pkl"
SCALER_Y_CRF_PATH = "./model/v10_scaler_y_crf.pkl"
SCALER_Y_MAX_PATH = "./model/v10_scaler_y_max.pkl"

SEG_LEN = 2
LOG_FILE = "./log/encoding_log.txt"
CSV_FILE = "./log/features.csv"

# 해상도 정의
resolutions = {
    "360p":  "640x360",
    "480p":  "854x480",
    "720p":  "1280x720",
    "1080p": "1920x1080"
}
resolution_tags = list(resolutions.keys())


# (1) 세그먼트 분할 - 수정
def split_video(input_path, segment_dir, segment_length=2, drop_short=False, log_path=LOG_FILE):
    """
    -c copy 대신 재인코딩을 통해 정확한 길이의 세그먼트를 생성합니다.
    각 세그먼트의 첫 프레임을 강제로 키프레임으로 만들어 안정성을 높입니다.
    """
    f = open(log_path, "a", encoding="utf-8")
    video_name = os.path.basename(input_path)
    if not os.path.exists(input_path):
        f.write(f"[{video_name}] 입력 파일이 존재하지 않습니다: {input_path}\n")
        f.close()
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")

    try:
        duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_path]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        f.write(f"[{video_name}] 영상 길이: {duration:.2f}초\n")
    except (subprocess.CalledProcessError, ValueError) as e:
        f.write(f"[{video_name}] 영상 정보를 가져올 수 없습니다: {e}\n")
        f.close()
        raise RuntimeError(f"영상 정보를 가져올 수 없습니다: {e}")

    num_segments = ceil(duration / segment_length)
    f.write(f"[{video_name}] 생성할 세그먼트 수: {num_segments}개\n")

    if os.path.exists(segment_dir):
        for file in os.listdir(segment_dir):
            if file.startswith("segment_") and file.endswith(".mp4"):
                os.remove(os.path.join(segment_dir, file))

    seg_paths = []
    for i in range(num_segments):
        out = os.path.join(segment_dir, f"segment_{i}.mp4")
        start_time = i * segment_length
        actual_segment_length = min(segment_length, duration - start_time)

        if actual_segment_length < 0.1: continue # 너무 짧은 마지막 조각은 무시

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(actual_segment_length),
            "-an",
            # '-c:v', 'copy', # 이 부분을 아래 재인코딩 옵션으로 대체
            "-c:v", "libx264",
            "-preset", "ultrafast", # 품질 손실 최소화를 위해 빠른 프리셋 사용
            "-crf", "18", # 시각적으로 거의 무손실에 가까운 CRF 값
            #세그먼트 컷팅을 안정적으로 하기 위한 임시 재인코딩(첫 프레임 I-프레임 강제)
            #최종적으로 플레이어가 보게 될 스트림은 그다음 단계 encode_segments()에서 다시 인코딩된 결과라서,
            #거기서 예측된 값이 최종 품질/비트레이트를 결정
            "-force_key_frames", "expr:eq(n,0)", # 각 세그먼트의 첫 프레임을 I-프레임으로 강제
            out
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            seg_paths.append(out)
            f.write(f"[{video_name}] ✓ segment_{i}.mp4 (재인코딩) 생성완료\n")
        except subprocess.CalledProcessError as e:
            f.write(f"[{video_name}] ✗ segment_{i}.mp4 생성실패: FFmpeg 오류 (코드 {e.returncode})\n")

    f.write(f"[{video_name}] === 분할 완료 ===\n\n")
    f.close()
    return seg_paths

# (2) 세그먼트 인코딩 - 수정
def encode_segments(seg_paths, out_dir, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model):
    """
    AI 인코딩 결과물을 .mp4가 아닌 .ts (MPEG-TS) 포맷으로 저장합니다.
    """
    ai_segs_by_res = {tag: [] for tag in resolution_tags}
    prev_crf, prev_maxrate, prev_gop, prev_features = None, None, None, None

    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}]=== 영상 특성 추출 및 인코딩 파라미터 최적화 ===\n")

    for i, seg_path in enumerate(seg_paths):
        if i == len(seg_paths) - 1 and prev_features is not None:
            crf, maxrate, gop, features = prev_crf, prev_maxrate, prev_gop, prev_features
            f.write(f"[segment_{i}] (last, reuse prev): CRF={crf}, maxrate={maxrate}k\n")
        else:
            features = extract_features(seg_path)
            crf, maxrate = ai_predict(features, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model)
            fps = features['fps']
            gop = int(fps * SEG_LEN)
            prev_crf, prev_maxrate, prev_gop, prev_features = crf, maxrate, gop, features
            f.write(f"[segment_{i} 원본: {features['width']}x{features['height']}] CRF={crf}, maxrate={maxrate}k\n")

        for tag, res in resolutions.items():
            # 출력 파일 확장자를 .ts로 변경
            seg_name = f"ai_seg_{i}_{tag}.ts"
            out_ts = os.path.join(out_dir, seg_name)

            base_pixels = features['width'] * features['height']
            w, h = map(int, res.lower().split("x"))
            target_pixels = w * h
            scaled_maxrate = round(int(maxrate) * (target_pixels / base_pixels)) if base_pixels > 0 else int(maxrate)

            scaled_crf = max(crf - 1, 18) if target_pixels < base_pixels else crf
            f.write(f"[segment_{i} {res}] CRF={scaled_crf}, maxrate={scaled_maxrate}k -> .ts\n")

            cmd = [
                "ffmpeg", "-y", "-i", seg_path,
                "-vf", f"scale={res}",
                "-c:v", "libx264", "-preset", "fast",
                "-crf", str(scaled_crf),
                "-maxrate", f"{scaled_maxrate}k",
                "-bufsize", f"{int(scaled_maxrate) * 2}k",
                "-g", str(gop),
                "-keyint_min", str(gop),
                "-sc_threshold", "0",
                "-an",
                # 출력 포맷을 MPEG-TS로 지정
                "-f", "mpegts",
                out_ts
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ai_segs_by_res[tag].append(out_ts)

    f.write(f"[{video_name}] === AI 인코딩 완료 ===\n\n")
    f.close()
    return ai_segs_by_res


###########################################################################
# 개별 세그먼트 특성 추출
def extract_features(input_path):
    stats = {}
    try:
        # 1. 영상 메타 데이터 + 공간적 복잡도 추출
        video_info = analyze_video(input_path, 10)
        if video_info is None:
            return {"video_path": input_path, "error": "Video open failed"}
        stats.update(video_info)
        
        # 2. 모션 벡터 추출
        mv_stats = extract_motion_vector(input_path)
        stats.update(mv_stats)
        
        # 3. 매크로블록 분석
        mb_stats = get_macroblock_ratios(input_path)
        stats['Intra_macro'] = mb_stats.get("Intra_macro", 0)
        stats['Inter_macro'] = mb_stats.get("Inter_macro", 0)
        stats['Skip_macro'] = mb_stats.get("Skip_macro", 0)
        
        return stats
    
    except Exception as e:
        print(f"Error for {input_path}: {e}")
        return

# 세그먼트별 영상 메타데이터 + 공간적 복잡도 추출 함수
def analyze_video(video_path, sample_rate):
    """
    세그먼트별 영상 정보 추출

    Args:
        video_path (str): 비디오 파일 경로
        sample_rate (int): 프레임 샘플링 비율 (1=모든 프레임, 2=2프레임마다 1개)
        canny_low (int): Canny edge detection 낮은 임계값
        canny_high (int): Canny edge detection 높은 임계값
        gaussian_blur (bool): 전처리시 가우시안 블러 적용 여부

    Returns:
        Dict: 영상 정보 딕셔너리 또는 None (오류시)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return None

    try:
        # 기본 비디오 메타데이터
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_size = get_file_size(video_path)/1024

        processed_frames = 0
        edge_density = []
        pixel_entropy = []

        # 비디오 유효성 검사
        if width <= 0 or height <= 0 or fps <= 0:
            print(f"Invalid video properties: {width}x{height}@{fps}fps")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 샘플링
            if processed_frames % sample_rate != 0:
                processed_frames += 1
                continue

            # 그레이스케일 변환 안전 처리
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Edge Density 계산 (Canny Edge Detection)
            edge_density.append(edge_density_lbp(gray))

            # Texture Complexity 계산 (Laplacian Variance)
            pixel_entropy.append(frame_entropy(gray))

            processed_frames += 1

        info = {
            "video_path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": processed_frames,
            "edge_density": np.mean(edge_density),
            "pixel_entropy": np.mean(pixel_entropy),
            "original_size": original_size
        }

        return info

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None

    finally:
        cap.release()

def frame_entropy(frame, bins=256):
    """픽셀 히스토그램 기반 엔트로피 계산"""
    hist, _ = np.histogram(frame.ravel(), bins=bins, range=(0, 256), density=True)
    hist = hist[hist > 0]  # 0 확률은 제외
    return entropy(hist, base=2)  # Shannon entropy

def edge_density_lbp(frame, P=8, R=1):
    """LBP 기반 엣지 밀도 계산"""
    lbp = local_binary_pattern(frame, P, R, method="uniform")

    # LBP 값이 0 또는 P+1은 균일/평평한 영역, 그 외는 텍스쳐/에지 영역
    edge_pixels = np.sum(lbp <= 4)
    total_pixels = lbp.size
    edge_percent = (edge_pixels / total_pixels) * 100

    return edge_percent

# 파일 크기 측정 함수
def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        return 0
    
# 모션 벡터 추출 함수
def calculate_direction_entropy(motion_x, motion_y, n_bins=16):
    """
    모션 벡터 방향 엔트로피 계산 (올바른 버전)

    Args:
        motion_x, motion_y: 모션 벡터 성분
        n_bins: 히스토그램 bin 개수

    Returns:
        entropy: 방향 엔트로피 (0 ~ log2(n_bins) 범위)
    """
    # 모션 벡터가 없는 경우 (정적인 영상)
    if len(motion_x) == 0 or len(motion_y) == 0:
        return 0.0

    # 영벡터 제거 (움직임이 없는 경우)
    magnitude = np.sqrt(motion_x**2 + motion_y**2)
    valid_mask = magnitude > 1e-6  # 아주 작은 움직임도 제거

    if not np.any(valid_mask):
        return 0.0  # 모든 벡터가 영벡터

    motion_x = motion_x[valid_mask]
    motion_y = motion_y[valid_mask]

    # 각도 계산 (-π ~ π)
    angles = np.arctan2(motion_y, motion_x)

    # 히스토그램 계산 (density=False로 변경!)
    counts, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))

    # 확률로 변환
    total_count = np.sum(counts)
    if total_count == 0:
        return 0.0

    probabilities = counts / total_count

    # 0이 아닌 확률만 사용하여 엔트로피 계산
    probabilities = probabilities[probabilities > 0]

    if len(probabilities) == 0:
        return 0.0

    # 정보 엔트로피 계산
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

def extract_motion_vector(video_path, mv_threshold=1.0):
    """
    개선된 모션 벡터 추출 및 분석

    Args:
        video_path: 비디오 파일 경로
        mv_threshold: 유효한 모션 벡터로 간주할 최소 크기
    """
    # 영상 파일 오픈
    cap = VideoCap()
    cap.open(video_path)

    partition_size_counts = defaultdict(int)
    all_mv_magnitudes = []
    all_mv_x = []
    all_mv_y = []
    total_frames = 0

    # 프레임별 모션 통계 (시간적 변화 분석용)
    frame_motion_stats = []

    try:
        while True:
            success, frame, motion_vectors, frame_type, timestamp = cap.read()

            if not success:
                break

            total_frames += 1

            # I-프레임이거나 모션 벡터가 없는 경우 (강제 키프레임은 무시)
            if motion_vectors.shape[0] == 0:
                frame_motion_stats.append({
                    'frame_type': frame_type,
                    'avg_magnitude': 0.0,
                    'block_count': 0,
                    'significant_motion_ratio': 0.0
                })
                continue

            # 모션 벡터 데이터 추출
            blk_w = motion_vectors[:, 1].astype(int)
            blk_h = motion_vectors[:, 2].astype(int)
            mv_x_raw = motion_vectors[:, 7]
            mv_y_raw = motion_vectors[:, 8]
            mv_scale = motion_vectors[:, 9]

            # motion_x, motion_y 스케일링
            mvx = mv_x_raw / mv_scale
            mvy = mv_y_raw / mv_scale

            # 모션 벡터 크기 계산
            magnitudes = np.sqrt(mvx**2 + mvy**2)

            # 의미 있는 움직임만 필터링 (노이즈 제거)
            significant_motion_mask = magnitudes >= mv_threshold

            # 유효한 모션 벡터 수집
            valid_magnitudes = magnitudes[significant_motion_mask]
            valid_mvx = mvx[significant_motion_mask]
            valid_mvy = mvy[significant_motion_mask]

            if len(valid_magnitudes) > 0:
                all_mv_magnitudes.extend(valid_magnitudes)
                all_mv_x.extend(valid_mvx)
                all_mv_y.extend(valid_mvy)

            # 파티션 크기 통계 (모든 블록)
            for w, h in zip(blk_w, blk_h):
                partition_size_counts[f"mb{w}x{h}"] += 1

            # 프레임별 통계
            frame_motion_stats.append({
                'frame_type': frame_type,
                'avg_magnitude': np.mean(valid_magnitudes) if len(valid_magnitudes) > 0 else 0.0,
                'block_count': len(motion_vectors),
                'significant_motion_ratio': np.sum(significant_motion_mask) / len(magnitudes) if len(magnitudes) > 0 else 0.0
            })

    finally:
        cap.release()

    all_mv_magnitudes = np.array(all_mv_magnitudes)
    all_mv_x = np.array(all_mv_x)
    all_mv_y = np.array(all_mv_y)

    # 결과 분석
    if len(all_mv_magnitudes) == 0:
        print(f"No significant motion vectors found in {video_path}")
        return None

    # 모션 벡터 통계 계산
    mv_stats = calculate_motion_statistics(
        all_mv_magnitudes, all_mv_x, all_mv_y,
        partition_size_counts, frame_motion_stats
    )

    return mv_stats

def calculate_motion_statistics(magnitudes, mvx, mvy, partition_counts, frame_motion_stats):
    """
    모션 벡터 통계 계산 함수 (정리된 버전)

    Parameters:
    - magnitudes: np.array, 각 블록의 모션 벡터 크기
    - mvx, mvy: np.array, 각 블록의 모션 벡터 x, y 성분
    - partition_counts: dict, {'mb16x16': int, 'mb16x8': int, 'mb8x8': int, ...}
    - frame_motion_stats: list of dict, 이전 프레임별 통계

    Returns:
    - motion_stats: dict, 계산된 모션 통계
    """

    # 기본 모션 벡터 통계
    avg_magnitude = np.mean(magnitudes)
    std_magnitude = np.std(magnitudes)
    max_magnitude = np.max(magnitudes)
    median_magnitude = np.median(magnitudes)

    # 모션 에너지
    motion_energy = np.sum(magnitudes**2) / len(magnitudes) if len(magnitudes) > 0 else 0.0

    # 방향 엔트로피
    direction_entropy = calculate_direction_entropy(np.array(mvx), np.array(mvy))

    # 방향성 일관성
    angles = np.arctan2(mvy, mvx)
    angle_std = np.std(angles)
    directional_consistency = max(0.0, 1.0 - (angle_std / np.pi))

    # 파티션 비율 계산
    total_blocks = sum(partition_counts.values())

    # mb16x8과 mb8x16 합치기
    merged_count = partition_counts.get("mb16x8", 0) + partition_counts.get("mb8x16", 0)
    if merged_count > 0:
        partition_counts["mb16x8"] = merged_count
        partition_counts.pop("mb8x16", None)

    partition_ratios = {k: v / total_blocks for k, v in partition_counts.items()} if total_blocks > 0 else {}

    # 표준 파티션 없으면 0으로 채우기
    standard_partitions = ['mb16x16', 'mb16x8', 'mb8x8']
    for partition in standard_partitions:
        if partition not in partition_ratios:
            partition_ratios[partition] = 0.0

    # 파티션 다양성 (엔트로피)
    partition_diversity = np.sum([p * np.log2(p) for p in partition_ratios.values() if p > 0]) * -1

    # 시간적 모션 통계
    frame_magnitudes = [f['avg_magnitude'] for f in frame_motion_stats if f.get('avg_magnitude', 0) > 0]
    temporal_consistency = {
        'motion_variance_across_frames': np.var(frame_magnitudes) if len(frame_magnitudes) > 1 else 0.0,
        'motion_temporal_smoothness': calculate_temporal_smoothness(frame_magnitudes),
        'avg_significant_motion_ratio': np.mean([f.get('significant_motion_ratio', 0) for f in frame_motion_stats]) if frame_motion_stats else 0.0
    }

    # 프레임 수
    frame_count = len(frame_motion_stats)

    # 최종 결과 통합
    motion_stats = {
        'mean_mv_magnitude': avg_magnitude,
        'std_mv_magnitude': std_magnitude,
        'max_mv_magnitude': max_magnitude,
        'median_mv_magnitude': median_magnitude,
        'mv_95_percentile': np.percentile(magnitudes, 95),
        'motion_energy': motion_energy,
        'direction_entropy': direction_entropy,
        'directional_consistency': directional_consistency,
        'partition_diversity': partition_diversity,
        'frame_count': frame_count,
        **partition_ratios,
        **temporal_consistency,
    }

    return motion_stats

def calculate_temporal_smoothness(frame_magnitudes):
    """
    프레임 간 모션 크기의 시간적 일관성 측정
    강제 키프레임 환경에서 중요한 지표
    """
    if len(frame_magnitudes) < 2:
        return 0.0

    # 연속 프레임 간 차이의 표준편차 (낮을수록 일관성 높음)
    differences = np.diff(frame_magnitudes)
    return np.std(differences) if len(differences) > 0 else 0.0

# 매크로블록 추출 함수
def get_macroblock_ratios(video_path):
    """
    FFmpeg x264 인코딩 로그에서 매크로블록 유형(I/Inter/Skip)과 파티션 크기 비율을
    실제 전체 프레임 기준으로 계산하고 CSV용 숫자 딕셔너리 반환
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "null", "-",  # 출력 안 함
        "-x264-params", "log-level=debug"
    ]

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    _, stderr = proc.communicate()

    # print(stderr)

    stats = {
        "Intra_macro": 0.0,
        "Inter_macro": 0.0,
        "Skip_macro": 0.0,
    }
    partition_sizes = {}

    # 1️⃣ 프레임 타입별 개수
    frame_counts = {}
    for line in stderr.splitlines():
        m = re.match(r"\[libx264.*\] frame (\w+):(\d+)", line)
        if m:
            frame_type, count = m.groups()
            frame_counts[frame_type] = int(count)

    # 2️⃣ mb 라인 파싱
    mb_lines = re.findall(r"mb ([IPB])\s+(.*)", stderr)
    for frame_type, mb_info in mb_lines:
        n_frames = frame_counts.get(frame_type, 1)

        # Skip 블록
        skip_match = re.search(r"skip:\s*([\d\.]+)%", mb_info)
        if skip_match:
            stats["Skip_macro"] += float(skip_match.group(1)) * n_frames

        # Intra 블록(I16/I4 등)
        intra_matches = re.findall(r"(I\d+\.\.\d+):\s*([\d\.]+)%", mb_info)
        for key, val in intra_matches:
            stats["Intra_macro"] += float(val) * n_frames
            # partition_sizes[f"{key.replace('..','_')}"] = partition_sizes.get(f"{key.replace('..','_')}", 0) + float(val) * n_frames

        # Inter 블록(P16/B16 등)
        inter_matches = re.findall(r"([PB]\d+\.\.\d+):\s*([\d\.]+)%", mb_info)
        for key, val in inter_matches:
            stats["Inter_macro"] += float(val) * n_frames
            # partition_sizes[f"{key.replace('..','_')}"] = partition_sizes.get(f"{key.replace('..','_')}", 0) + float(val) * n_frames


    # 3️⃣ 전체 프레임 수 합
    total_frames = sum(frame_counts.values())

    # 4️⃣ 전체 프레임 기준 평균 비율 계산
    if total_frames > 0:
        stats["Intra_macro"] /= total_frames * 100
        stats["Inter_macro"] /= total_frames * 100
        stats["Skip_macro"] /= total_frames * 100
        # for key, val in partition_sizes.items():
        #     stats[f"partition_{key}"] = val / total_frames

    return stats

# (3) AI 모델 예측
def ai_predict(feature, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model):
    
    input_features = [
        'width', 'height', 'original_size', 'fps', 'frame_count',
        'edge_density', 'pixel_entropy',
        'mean_mv_magnitude', 'std_mv_magnitude', 'max_mv_magnitude',
        'median_mv_magnitude', 'mv_95_percentile', 'motion_energy',
        'direction_entropy', 'directional_consistency',
        'partition_diversity',
        'mb16x16', 'mb16x8', 'mb8x8',
        'motion_variance_across_frames', 'motion_temporal_smoothness', 'avg_significant_motion_ratio',
        'Intra_macro', 'Inter_macro', 'Skip_macro',
    ]
    
    # 딕셔너리 -> 리스트 변환
    feature_list = [feature[feat] for feat in input_features]
    
    # 입력 전처리
    X_scaled = scaler_X.transform([feature_list])
    
    # 모델 예측
    y_pred_crf = crf_model.predict(X_scaled)
    y_pred_maxrate = max_model.predict(X_scaled)
    
    # 출력 변환
    crf = round(float(scaler_y_crf.inverse_transform(y_pred_crf.reshape(1, -1))))
    maxrate = round(float(scaler_y_max.inverse_transform(y_pred_maxrate.reshape(1, -1))))

    return crf, maxrate

#######################################################################

# (4) 해상도별 concat.txt
def make_concat_txt_multi_res(ai_segs_by_res, out_dir):
    """
    각 해상도별로 인코딩된 세그먼트의 경로를 concat_360p.txt 파일에 저장
    
    Parameters:
    - ai_segs_by_res: {해상도:[세그먼트 경로 리스트]} 딕셔너리
    - out_dir: './static/{video}/concat 폴더 경로
    
    Returns:
    - concat_txts: {해상도:concat_txt경로} 딕셔너리
    """
    log = open(LOG_FILE, "a", encoding="utf-8")
    log.write(f"[{video_name}] === 해상도별 concat.txt 생성 === \n")
    
    concat_txts = {}
    for tag, segs in ai_segs_by_res.items():
        concat_txt = os.path.join(out_dir, f"concat_{tag}.txt")
        with open(concat_txt, "w") as f:
            for p in segs:
                f.write(f"file '{os.path.abspath(p)}'\n")
        concat_txts[tag] = concat_txt
        log.write(f"[{tag}] concat.txt: {concat_txt}\n")
    
    log.write(f"[{video_name}] === 해상도별 concat.txt 생성 완료 === \n\n")    
    log.close()
    
    return concat_txts

# (5) 해상도별 병합 mp4 - 수정
def concat_segments_multi_res(concat_txts, out_dir):
    """
    .ts 세그먼트들을 concat demuxer로 병합하고,
    결과물을 스트림 복사를 통해 .mp4 컨테이너로 재포장(remuxing)합니다.
    이 과정에서 타임스탬프가 재정렬됩니다.
    """
    merged_mp4s = {}
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === 해상도별 TS 세그먼트 병합 및 MP4로 재포장 === \n")

    for tag, concat_txt in concat_txts.items():
        out_mp4 = os.path.join(out_dir, f"merged_ai_{tag}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c", "copy", # 인코딩 없이 스트림만 복사 (TS -> MP4)
            "-movflags", "+faststart", # 스트리밍을 위한 mov atom 재배치
            out_mp4
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        merged_mp4s[tag] = out_mp4
        f.write(f"[{tag}] .ts 병합 -> .mp4 재포장 완료: {out_mp4}\n")

    f.write(f"[{video_name}] === 해상도별 세그먼트 병합 mp4 생성 완료 === \n\n")
    f.close()
    return merged_mp4s


# (7) **MP4Box로 여러 해상도 mp4를 한 번에 dash 분할 (한 개의 mpd 생성)** - 수정 (fixed_mp4s → merged_mp4s)
def mp4box_dash_multi_res(merged_mp4s, out_dir, segment_ms=2000):
    """
    MP4Box로 MPEG_DASH 스트리밍용 세그먼트와 MPD(manifest) 파일 생성
    
    Parameters:
    - merged_mp4s: {해상도:병합된 mp4 경로} 딕셔너리
    - out_dir: './static/{video}/mp4bos' 폴더 경로
    - segment_ms: 세그먼트 분할 간격(2초=2000ms)
    
    Returns:
    - None
    """
    
    mp4_args = []
    
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === DASH MPD/m4s 생성 시작=== \n")
    
    for tag in resolution_tags:
        mp4_args.append(f"{merged_mp4s[tag]}#video:name={tag}")
    mpd_path = os.path.join(out_dir, "manifest.mpd")
    cmd = [
        "MP4Box", 
        "-dash", str(segment_ms), 
        "-frag", str(segment_ms), 
        "-rap",
        "-profile", "dashavc264:live",
        "-out", mpd_path
    ] + mp4_args
    subprocess.run(cmd, check=True)
    f.write(f"[{tag}] MP4Box 실행: {cmd}\n")
    
    f.write(f"[{video_name}] === DASH MPD/m4s 생성 완료 === \n\n")    
    f.close()





# 여기서 부터는 초단위 비트레이트 추출 및 CSV 저장하는 함수
# [CHANGED] 세그먼트 파일명 정규식: ai_seg_{idx}_{res}.(ts|mp4)
# [ADD] 세그먼트 파일명 정규식: ai_seg_{idx}_{res}.(ts|mp4)
_SEG_RE = re.compile(r"^ai_seg_(\d+)_([A-Za-z0-9]+)\.(ts|mp4)$")


def _normalize_seg_name(filename: str) -> tuple[int, str, str]:
    name = os.path.basename(filename)
    m = _SEG_RE.match(name)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    fixed = name.replace(" ", "_")
    m2 = _SEG_RE.match(fixed)
    if m2:
        return int(m2.group(1)), m2.group(2), m2.group(3)
    raise ValueError(f"Unrecognized segment file name: {name}")

def _probe_kbps_per_second(media_path: str) -> dict[int, float]:
    """
    ffprobe로 패킷 단위 size를 읽어 1초 버킷 합산 (세그먼트 첫 타임스탬프를 0으로 정규화)
    return: {sec(int: 0..SEG_LEN-1), kbps(float)}
    """
    # [CHANGED] -show_packets 추가, dts_time도 함께 요청
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_packets",                                     # [CHANGED]
        "-show_entries", "packet=pts_time,dts_time,size",    # [CHANGED]
        "-of", "csv=p=0",
        media_path
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    kbps_per_sec: dict[int, float] = {}
    first_ts = None

    for line in r.stdout.strip().splitlines():
        # 기대 포맷: pts_time,dts_time,size
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue

        pts_str, dts_str, size_str = parts[0], parts[1], parts[2]

        # [CHANGED] pts_time이 N/A이면 dts_time으로 대체
        ts_str = pts_str if pts_str and pts_str != "N/A" else dts_str
        if not ts_str or ts_str == "N/A":
            continue

        try:
            ts = float(ts_str)
            size_bytes = int(size_str)
        except Exception:
            continue

        if first_ts is None:
            first_ts = ts

        # 세그먼트 기준 0초부터 정규화
        rel = ts - first_ts
        # 간헐적 음수/NaN 방지
        if not (rel == rel):  # NaN 체크
            continue
        if rel < 0:
            rel = 0.0

        sec = int(floor(rel))  # 0,1,...
        if sec >= SEG_LEN:
            sec = SEG_LEN - 1  # [KEEP] 2초 세그먼트면 0/1로 캡

        kbps_per_sec.setdefault(sec, 0.0)
        kbps_per_sec[sec] += (size_bytes * 8) / 1000.0  # bytes→kbps 누적

    # [ADD] 안전장치: 전부 0초만 채워졌다면 2초로 분배
    if len(kbps_per_sec) == 1 and 0 in kbps_per_sec and SEG_LEN >= 2:
        total = kbps_per_sec[0]
        half = round(total / SEG_LEN, 2)
        kbps_per_sec = {0: half, 1: half}  # 균등 분배(간단/안정)

    return kbps_per_sec


def extract_bitrate_per_segment(segment_dir: str, out_csv_path: str, add_one_index: bool = True):
    """
    각 세그먼트(ts/mp4)의 1초 단위 비트레이트 측정 → CSV 저장
    columns: segment_name,time_second,bitrate_kbps,resolution
    - segment_dir: ./static/{video}/ai_segment  (권장)
    - out_csv_path: ./static/{video}/bitrate/{video}_bitrate_per_second.csv
    - add_one_index: 세그먼트 인덱스를 +1 해서 프론트 표기와 맞출지 여부
    """
    import csv
    from glob import glob

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    # [CHANGED] ts/mp4 모두 대상
    files = sorted(
        glob(os.path.join(segment_dir, "ai_seg_*_*.ts")) +
        glob(os.path.join(segment_dir, "ai_seg_*_*.mp4")),
        key=lambda p: os.path.basename(p)
    )

    rows = []
    for seg_path in files:
        try:
            idx, res, ext = _normalize_seg_name(seg_path)
        except ValueError:
            # 이름 규칙 벗어나면 스킵
            continue

        disp_idx = idx + 1 if add_one_index else idx
        # [CHANGED] 프론트에서 보기 좋은 표기 고정: ai_seg_{n}_{res}.mp4
        segment_name_for_csv = f"ai_seg_{disp_idx}_{res}.mp4"

        # 1초 버킷별 kbps 집계
        sec_kbps = _probe_kbps_per_second(seg_path)
        for sec in sorted(sec_kbps.keys()):
            rows.append([segment_name_for_csv, sec, round(sec_kbps[sec], 2), res])

    # 시간/세그먼트 순 정렬
    def _sort_key(row):
        # row = [segment_name, sec, kbps, res]
        # segment_name: ai_seg_{n}_{res}.mp4
        try:
            n = int(row[0].split("_")[2])  # n
        except Exception:
            n = 0
        return (n, row[1])

    rows.sort(key=_sort_key)

    # CSV 저장
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(["segment_name", "time_second", "bitrate_kbps", "resolution"])  # [CHANGED]
        writer.writerows(rows)

    # [CHANGED] 파일 로그
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{video_name}] Bitrate CSV saved: {out_csv_path} (rows={len(rows)})\n")
    print(f"[비트레이트 분석 완료] → {out_csv_path}")


# (8) 전체 실행
if __name__ == "__main__":
    CRF_MODEL = joblib.load(CRF_MODEL_PATH)
    MAXRATE_MODEL = joblib.load(MAXRATE_MODEL_PATH)
    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y_CRF = joblib.load(SCALER_Y_CRF_PATH)
    SCALER_Y_MAX = joblib.load(SCALER_Y_MAX_PATH)

    for video_name, input_path in INPUT_VIDEOS:        
        with open(LOG_FILE, "a") as f:  # 'a'는 append 모드 (기존 내용 뒤에 추가)
            f.write(f"=== [{video_name}] 영상 인코딩 시작 ===\n")
            
        # 각 영상별 출력 폴더 지정
        OUT_DIR = f"./static/{video_name}"
        os.makedirs(OUT_DIR, exist_ok=True)

        print("1. 세그먼트 분할")
        SEG_DIR = f"./static/{video_name}/segments"
        os.makedirs(SEG_DIR, exist_ok=True)
        seg_paths = split_video(
            input_path,
            SEG_DIR,
            segment_length=SEG_LEN,
            drop_short=False,
            log_path=LOG_FILE)
        
        print("2. 세그먼트별 AI 인코딩 (해상도 4개)")
        ai_segment = f"./static/{video_name}/ai_segment"
        os.makedirs(ai_segment, exist_ok=True)
        ai_segs_by_res = encode_segments(
            seg_paths, 
            ai_segment, 
            SCALER_X, 
            SCALER_Y_CRF, 
            SCALER_Y_MAX, 
            CRF_MODEL, 
            MAXRATE_MODEL)
        
        # [CHANGED] 세그먼트별 초단위 비트레이트 추출 + CSV 저장
        bitrate_dir = f"./static/{video_name}/bitrate"
        os.makedirs(bitrate_dir, exist_ok=True)
        bitrate_csv = os.path.join(bitrate_dir, f"{video_name}_bitrate_per_second.csv")  # 예: husky_encode13_bitrate_per_second.csv
        extract_bitrate_per_segment(ai_segment, bitrate_csv, add_one_index=True)  # [CHANGED]


        print("3. concat.txt 생성")
        concat_dir = f"./static/{video_name}/concat_dir"
        os.makedirs(concat_dir, exist_ok=True)
        concat_txts = make_concat_txt_multi_res(
            ai_segs_by_res, 
            concat_dir)
        
        print("4. 병합 mp4 생성")
        merged_dir = f"./static/{video_name}/merged"
        os.makedirs(merged_dir, exist_ok=True)
        merged_mp4s = concat_segments_multi_res(
            concat_txts, 
            merged_dir)
    
        print("6. mp4box로 dash 분할 + mpd 생성 (1개)")
        mp4box_dir = f"./static/{video_name}/mp4box"
        os.makedirs(mp4box_dir, exist_ok=True)
        mp4box_dash_multi_res(
            merged_mp4s,  # fixed_mp4s → merged_mp4s 수정
            mp4box_dir, 
            segment_ms=SEG_LEN*1000)
        
        print(f"=== [{video_name}] DASH 인코딩 완료 ===\n")
