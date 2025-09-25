"""
기존과 유사하게 1.99 단위로 세그먼트를 재인코딩한 후,
MP4Box에서 -segment-timeline을 사용해 타임라인 정렬
--> MPD 파일이 다소 길어지나 비교적 버벅거림이 덜함
"""

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
    ("husky", "../input/husky.mp4")
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
    f.write(f"[{video_name}] === 분할 완료 ===\n")
    f.write(f"[{video_name}] 성공: {success_count}개 세그먼트\n\n")
    
    if failed_segments:
        f.write(f"[{video_name}] 실패한 세그먼트: {failed_segments}\n")
    
    if success_count == 0:
        f.write(f"[{video_name}] 모든 세그먼트 생성에 실패했습니다\n")
        f.close()
        raise RuntimeError("모든 세그먼트 생성에 실패했습니다")
    
    f.close()
    return seg_paths

# (2) 세그먼트 특성 추출 및 인코딩
def encode_segments(seg_paths, out_dir, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model):
    ai_segs_by_res = {tag: [] for tag in resolution_tags}
    prev_crf, prev_maxrate = None, None  # 직전 세그먼트 값 저장
    
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}]=== 영상 특성 추출 및 인코딩 파라미터 최적화 ===\n")
        
    for i, seg_path in enumerate(seg_paths):
        
        # 마지막 세그먼트는 특징 추출 건너뛰고 이전 값 재사용
        if i == len(seg_paths) - 1 and prev_crf is not None and prev_maxrate is not None:
            crf, maxrate, gop = prev_crf, prev_maxrate, prev_gop
            f.write(f"[segment_{i}] (last, reuse prev): CRF={crf}, maxrate={maxrate}k\n")
        else:
            features = extract_features(seg_path)
            crf, maxrate = ai_predict(features, scaler_X, scaler_y_crf, scaler_y_max, crf_model, max_model)
            fps = features['fps']
            gop = int(fps * SEG_LEN)
            
            prev_crf, prev_maxrate, prev_gop = crf, maxrate, gop  # 직전 세그먼트 값 저장
            
            f.write(f"[segment_{i} 원본: {features['width']}x{features['height']}] CRF={crf}, maxrate={maxrate}k\n")
        
        for tag, res in resolutions.items():
            seg_name = f"ai_seg_{i}_{tag}.mp4"
            out_mp4 = os.path.join(out_dir, seg_name)
                    
            # Max rate 해상도 비례 스케일링
            base_pixels  = features['width'] * features['height']
            w, h = map(int, res.lower().split("x"))
            target_pixels  = w * h
            scaled_maxrate = round(int(maxrate) * (target_pixels  / base_pixels ))
            
            # CRF 보정 (저해상도일수록 블록 깨짐 방지 위해 낮춰줌)
            if target_pixels < base_pixels:
                scaled_crf = max(crf-1, 18)
            else:
                scaled_crf = crf
            
            f.write(f"[segment_{i} {res}] CRF={scaled_crf}, maxrate={scaled_maxrate}k\n")
            
            # --- 인코딩 (해상도별)
            # 정확히 2초 단위로 분할하기 위해 GOP 구조 변경
            cmd = [
                "ffmpeg", "-y", "-i", seg_path,
                "-vf", f"scale={res}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", str(scaled_crf),
                "-maxrate", f"{scaled_maxrate}k",
                "-bufsize", f"{int(scaled_maxrate)*2}k",
                # "-g", str(gop),
                # "-keyint_min", str(gop),
                # "-sc_threshold", "0",
                "-an", out_mp4
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ai_segs_by_res[tag].append(out_mp4)
            
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
        original_size = get_file_size(input_path)/1024
        
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

# (5) 해상도별 병합 mp4
def concat_segments_multi_res(concat_txts, out_dir):
    """
    각 해상도별로 저장된 세그먼트를 하나로 병합
    여러 개의 mp4 세그먼트를 하나로 합쳐서 최종 mp4 파일을 만드는 부분
    
    Parameters:
    - concat_txts: {해상도:concat_txt경로} 딕셔너리
    - out_dir: './static/{video}/merged' 폴더 경로
    
    Returns:
    - merged_mp4s: {해상도: 병합된 영상 경로} 딕셔너리
    """
    
    merged_mp4s = {}
    
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === 해상도별 세그먼트 병합 === \n")
    
    for tag, concat_txt in concat_txts.items():
        out_mp4 = os.path.join(out_dir, f"merged_ai_{tag}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            out_mp4
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        merged_mp4s[tag] = out_mp4
        f.write(f"[{tag}] mp4 병합: {out_mp4}\n")
    
    f.write(f"[{video_name}] === 해상도별 세그먼트 병합 mp4 생성 완료 === \n\n")    
    f.close()
    
    return merged_mp4s

# (6) 해상도별 병합 mp4 재인코딩
def reencode_mp4_multi_res(merged_mp4s, out_dir):
    """
    각 해상도별로 병합된 mp4를 다시 인코딩하는 과정
    병합된 mp4 파일을 재인코딩+스트리밍 최적화하여 플레이어 호환성을 높임
    
    Parameters:
    - merged_mp4s: {해상도:병합된 mp4 경로} 딕셔너리
    - out_dir: './static/{video}/fixed' 폴더 경로
    
    Returns:
    - fixed_mp4s: {해상도: 재인코딩 영상 경로} 딕셔너리
    """
    
    fixed_mp4s = {}
    
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === 해상도별 mp4 재인코딩 === \n")
    
    for tag, merged_mp4 in merged_mp4s.items():
        fixed_mp4 = os.path.join(out_dir, f"merged_ai_fixed_{tag}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", merged_mp4,
            "-c:v", "copy",
            # "-force_key_frames", f"expr:gte(t,n_forced*{SEG_LEN})",  # 2초마다 강제 I-frame
            "-movflags", "+faststart",
            fixed_mp4
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fixed_mp4s[tag] = fixed_mp4
        f.write(f"[{tag}] mp4 재인코딩: {fixed_mp4}\n")
        
    f.write(f"[{video_name}] === 해상도별 MP4 재인코딩 완료 === \n\n")    
    f.close()
    
    return fixed_mp4s

# (7) **MP4Box로 여러 해상도 mp4를 한 번에 dash 분할 (한 개의 mpd 생성)**
def mp4box_dash_multi_res(fixed_mp4s, out_dir, segment_ms=2000):
    """
    MP4Box로 MPEG_DASH 스트리밍용 세그먼트와 MPD(manifest) 파일 생성
    
    Parameters:
    - fixed_mp4s: {해상도:재인코딩 mp4 경로} 딕셔너리
    - out_dir: './static/{video}/mp4bos' 폴더 경로
    - segment_ms: 세그먼트 분할 간격(2초=2000ms)
    
    Returns:
    - None
    """
    
    mp4_args = []
    
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === DASH MPD/m4s 생성 시작=== \n")
    
    for tag in resolution_tags:
        mp4_args.append(f"{fixed_mp4s[tag]}#video:name={tag}")
    mpd_path = os.path.join(out_dir, "manifest.mpd")
    cmd = [
        "MP4Box", 
        "-dash", str(segment_ms), 
        "-frag", str(segment_ms), 
        "-rap",
        # "-bs-switching", "no",
        # "-profile", "dashavc264:onDemand",
        "-profile", "dashavc264:live",
        "-segment-timeline",
        "-out", mpd_path
    ] + mp4_args
    subprocess.run(cmd, check=True)
    f.write(f"[{tag}] MP4Box 실행: {cmd}\n")
    
    f.write(f"[{video_name}] === DASH MPD/m4s 생성 완료 === \n\n")    
    f.close()

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
        
        print("5. 병합 mp4 재인코딩")
        fixed_dir = f"./static/{video_name}/fixed"
        os.makedirs(fixed_dir, exist_ok=True)
        fixed_mp4s = reencode_mp4_multi_res(
            merged_mp4s, 
            fixed_dir)
        
        print("6. mp4box로 dash 분할 + mpd 생성 (1개)")
        mp4box_dir = f"./static/{video_name}/mp4box"
        os.makedirs(mp4box_dir, exist_ok=True)
        mp4box_dash_multi_res(
            fixed_mp4s, 
            mp4box_dir, 
            segment_ms=SEG_LEN*1000)
        
        print(f"=== [{video_name}] DASH 인코딩 완료 ===\n")
