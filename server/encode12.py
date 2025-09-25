"""
실행순서
python3 encode.py 로 실행. F5 말고, 터미널로 하기
cd server
python3 -m uvicorn main:app --reload
Netflix 폴더에서 npm run dev 실행
"""

"""
pip install numpy scikit-image tqdm
"""

import os
import subprocess
import joblib
import numpy as np
import pandas as pd
from collections import Counter
import glob
import re
from math import ceil, floor
from scipy.stats import entropy
from skimage.feature import local_binary_pattern

INPUT_VIDEOS = [
    ("husky", "../input/husky.mp4")
]

CRF_MODEL_PATH = "./model/v10_crf_model.pkl"
MAXRATE_MODEL_PATH = "./model/v10_max_model.pkl"
SCALER_X_PATH = "./model/v10_scaler_X.pkl"
SCALER_Y_CRF_PATH = "./model/v10_scaler_y_crf.pkl"
SCALER_Y_MAX_PATH = "./model/v10_scaler_y_max.pkl"

SEG_LEN = 2
LOG_FILE = "./log/v1_log.txt"

# 해상도 정의
resolutions = {
    "360p":  "640x360",
    "480p":  "854x480",
    "720p":  "1280x720",
    "1080p": "1920x1080"
}
resolution_tags = list(resolutions.keys())


# (1) 세그먼트 분할
def split_video(input_path, segment_dir, segment_length=2, drop_short=False, log_path=LOG_FILE):
    """
    Args:
        input_path: 입력 비디오 파일 경로
        segment_dir: 세그먼트 저장 디렉토리
        segment_length: 세그먼트 길이 (초)
        drop_short: True면 짧은 마지막 세그먼트 제외, False면 포함
        log_path: 로그 저장용 파일 경로
    """

    # 로그 파일 열기
    f = open(log_path, "a", encoding="utf-8")

    video_name = os.path.basename(input_path)

    # 1. 입력 검증
    if not os.path.exists(input_path):
        f.write(f"[{video_name}] 입력 파일이 존재하지 않습니다: {input_path}\n")
        f.close()
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_path}")
        
    # 2. 영상 길이 가져오기
    try:
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", input_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        f.write(f"[{video_name}] 영상 길이: {duration:.2f}초\n")
    except (subprocess.CalledProcessError, ValueError) as e:
        f.write(f"[{video_name}] 영상 정보를 가져올 수 없습니다: {e}\n")
        f.close()
        raise RuntimeError(f"영상 정보를 가져올 수 없습니다: {e}")
    
    # 3. 세그먼트 수 계산
    if drop_short:
        num_segments = floor(duration / segment_length)
        f.write(f"[{video_name}] 생성할 세그먼트 수: {num_segments}개 (각 {segment_length}초, 짧은 세그먼트 제외)\n")
        excluded_length = duration - (num_segments * segment_length)
        if excluded_length > 0:
            f.write(f"[{video_name}] 제외될 마지막 부분: {excluded_length:.2f}초\n")
    else:
        num_segments = ceil(duration / segment_length)
        last_segment_length = duration - ((num_segments - 1) * segment_length)
        f.write(f"[{video_name}] 생성할 세그먼트 수: {num_segments}개 (마지막 세그먼트: {last_segment_length:.2f}초)\n")
    
    if num_segments == 0:
        f.write(f"[{video_name}] 생성할 세그먼트가 없습니다 (영상이 너무 짧음)\n")
        f.close()
        return []
    
    # 4. 기존 세그먼트 파일 정리
    if os.path.exists(segment_dir):
        removed_count = 0
        for file in os.listdir(segment_dir):
            if file.startswith("segment_") and file.endswith(".mp4"):
                try:
                    os.remove(os.path.join(segment_dir, file))
                    removed_count += 1
                except OSError as e:
                    f.write(f"[{video_name}] [경고] 파일 삭제 실패: {file} ({e})\n")
        
        if removed_count > 0:
            f.write(f"[{video_name}] 기존 세그먼트 파일 {removed_count}개 삭제됨\n")
    
    # 5. 세그먼트 생성
    seg_paths = []
    failed_segments = []
    
    for i in range(num_segments):
        out = os.path.join(segment_dir, f"segment_{i}.mp4")
        start_time = i * segment_length
        
        if drop_short:
            actual_segment_length = segment_length
        else:
            actual_segment_length = min(segment_length, duration - start_time)
        
        # 최소 길이 체크 (추가 안전장치)
        if drop_short:
            if actual_segment_length < segment_length:
                print(f"⚠️  segment_{i}.mp4 스킵됨 (길이 {actual_segment_length:.2f}초 < 최소 {segment_length:.2f}초)\n")
                continue
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(actual_segment_length),
            "-an",
            "-c:v", "copy",
            out
        ]
        
        try:
            result = subprocess.run(cmd, check=True, 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            
            if os.path.exists(out) and os.path.getsize(out) > 0:
                seg_paths.append(out)
                f.write(f"[{video_name}] ✓ segment_{i}.mp4 생성완료 ({start_time:.1f}s ~ {start_time + actual_segment_length:.1f}s, {actual_segment_length:.1f}초)\n")
            else:
                failed_segments.append(i)
                f.write(f"[{video_name}] ✗ segment_{i}.mp4 생성실패 (파일 크기 0)\n")
                
        except subprocess.CalledProcessError as e:
            failed_segments.append(i)
            f.write(f"[{video_name}] ✗ segment_{i}.mp4 생성실패: FFmpeg 오류 (코드 {e.returncode})\n")
            continue
    
    # 6. 결과 요약
    success_count = len(seg_paths)
    f.write(f"\n[{video_name}] === 분할 완료 ===\n")
    f.write(f"[{video_name}] 성공: {success_count}개 세그먼트\n\n")
    
    if failed_segments:
        f.write(f"[{video_name}] 실패한 세그먼트: {failed_segments}\n")
    
    if success_count == 0:
        f.write(f"[{video_name}] 모든 세그먼트 생성에 실패했습니다\n")
        f.close()
        raise RuntimeError("모든 세그먼트 생성에 실패했습니다")
    
    f.close()
    return seg_paths

# (2) 모션벡터/AI 특성 추출 함수 (변경 없음, 생략)
def ai_predict(feature_list, scaler_X, scaler_y, model):
    """
    feature_list : [mean_mag, max_mag, std_mag, count, I, P, B, Intra, Inter, Skip]
    """
    # feature_list는 이미 10개 특성이 정해진 순서로 담긴 리스트
    X_scaled = scaler_X.transform([feature_list])
    pred_scaled = model.predict(X_scaled)
    print(pred_scaled)
    pred_scaled = pred_scaled.reshape(1, -1)
    pred_y = scaler_y.inverse_transform(pred_scaled)
    crf = round(pred_y[0, 0])
    maxrate = pred_y[0, 1]
    return crf, maxrate

# 세그먼트 특성 추출 함수
def extract_features(input_path):
    stats = {}

    try:
        # 1. 영상 기본 정보 + 공간적 복잡도 추출
        video_info = analyze_video(input_path, 10)
        
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
        processed_frames = 0
        edge_density = []
        pixel_entropy = []

        # 비디오 유효성 검사
        if width <= 0 or height <= 0 or fps <= 0:
            print(f"Invalid video properties: {width}x{height}@{fps}fps")
            return None

        print(f"Processing video: {width}x{height}, {fps:.2f}fps")

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
        }

        print(f"Analysis complete: {processed_frames} frames processed")
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

# (3) AI 인코딩 + 해상도별 저장
def encode_segments(seg_paths, out_dir, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model):
    ai_segs_by_res = {tag: [] for tag in resolution_tags}
    prev_crf, prev_maxrate = None, None  # 직전 세그먼트 값 저장
    
    for i, seg_path in enumerate(seg_paths):
        
        # 마지막 세그먼트는 특징 추출 건너뛰고 이전 값 재사용
        if i == len(seg_paths) - 1 and prev_crf is not None and prev_maxrate is not None:
            crf, maxrate = prev_crf, prev_maxrate
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[AI인코딩] segment_{i} (last, reuse prev): CRF={crf}, maxrate={maxrate}\n")
        else:
            features = extract_features(seg_path)
            crf, maxrate = ai_predict(features, scaler_X, scaler_y, model)
            prev_crf, prev_maxrate = crf, maxrate  # 직전 세그먼트 값 저장
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[AI인코딩] segment_{i}: CRF={crf}, maxrate={maxrate}\n")
            print(f"[AI인코딩] segment_{i}: CRF={crf}, maxrate={maxrate}")
        
        for tag, res in resolutions.items():
            seg_name = f"ai_seg_{i}_{tag}.mp4"
            out_mp4 = os.path.join(out_dir, seg_name)
                    
            # --- 인코딩 (해상도별)
            cmd = [
                "ffmpeg", "-y", "-i", seg_path,
                "-vf", f"scale={res}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", str(crf),
                "-maxrate", str(int(maxrate)),
                "-bufsize", str(int(maxrate * 2)),
                "-an", out_mp4
            ]
            subprocess.run(cmd, check=True)
            ai_segs_by_res[tag].append(out_mp4)
    return ai_segs_by_res

# (4) 해상도별 concat.txt
def make_concat_txt_multi_res(ai_segs_by_res, out_dir):
    concat_txts = {}
    for tag, segs in ai_segs_by_res.items():
        concat_txt = os.path.join(out_dir, f"concat_{tag}.txt")
        with open(concat_txt, "w") as f:
            for p in segs:
                f.write(f"file '{os.path.abspath(p)}'\n")
        concat_txts[tag] = concat_txt
    return concat_txts

# (5) 해상도별 병합 mp4
def concat_segments_multi_res(concat_txts, out_dir):
    merged_mp4s = {}
    for tag, concat_txt in concat_txts.items():
        out_mp4 = os.path.join(out_dir, f"merged_ai_{tag}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            out_mp4
        ]
        subprocess.run(cmd, check=True)
        merged_mp4s[tag] = out_mp4
    return merged_mp4s

# (6) 해상도별 병합 mp4 재인코딩
def reencode_mp4_multi_res(merged_mp4s, out_dir):
    fixed_mp4s = {}
    for tag, merged_mp4 in merged_mp4s.items():
        fixed_mp4 = os.path.join(out_dir, f"merged_ai_fixed_{tag}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", merged_mp4,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-movflags", "+faststart",
            "-an", fixed_mp4
        ]
        subprocess.run(cmd, check=True)
        fixed_mp4s[tag] = fixed_mp4
    return fixed_mp4s

# (7) **MP4Box로 여러 해상도 mp4를 한 번에 dash 분할 (한 개의 mpd 생성)**
def mp4box_dash_multi_res(fixed_mp4s, out_dir, segment_ms=10000):
    mp4_args = []
    for tag in resolution_tags:
        mp4_args.append(f"{fixed_mp4s[tag]}#video:name={tag}")
    mpd_path = os.path.join(out_dir, "manifest.mpd")
    cmd = [
        "MP4Box", "-dash", str(segment_ms), "-frag", str(segment_ms), "-rap",
        "-profile", "dashavc264:live",
        "-out", mpd_path
    ] + mp4_args
    print("[MP4Box 실행]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[완료] DASH MPD/m4s 생성: {mpd_path}")

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
        
        print("2. 세그먼트별 특성 추출")
        extract_features()
        
        # print("2. 세그먼트별 AI 인코딩 (해상도 4개)")
        ai_segs_by_res = encode_segments(
            seg_paths, 
            OUT_DIR, 
            SCALER_X, 
            SCALER_Y_CRF, 
            SCALER_Y_MAX, 
            CRF_MODEL, 
            MAXRATE_MODEL)
        # print("3. concat.txt 생성")
        # concat_txts = make_concat_txt_multi_res(ai_segs_by_res, OUT_DIR)
        # print("4. 병합 mp4 생성")
        # merged_mp4s = concat_segments_multi_res(concat_txts, OUT_DIR)
        # print("5. 병합 mp4 재인코딩")
        # fixed_mp4s = reencode_mp4_multi_res(merged_mp4s, OUT_DIR)
        # print("6. mp4box로 dash 분할 + mpd 생성 (1개)")
        # mp4box_dash_multi_res(fixed_mp4s, OUT_DIR, segment_ms=SEG_LEN*1000)
        # print(f"=== [{video_name}] DASH 인코딩 완료 ===\n")
