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
    ("husky_encode13", "../input/husky.mp4")
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
            #최종적으로 플레이어가 보게 될 스트림은 그다음 단계 encode_segments()에서 다시 인코딩된 결과라서, 여기서 예측된 값이 최종 품질/비트레이트를 결정
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

# (2) 세그먼트 인코딩 - 수정됨 
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


# (3) concat.txt 생성 - (수정 없음)
def make_concat_txt_multi_res(ai_segs_by_res, out_dir):
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

# (4) 세그먼트 병합 - ★★★ 수정됨 ★★★
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


# (5) MP4Box DASH 생성 - (fixed_mp4s 대신 merged_mp4s를 입력으로 받도록 수정)
def mp4box_dash_multi_res(final_mp4s, out_dir, segment_ms=2000):
    mp4_args = []
    f = open(LOG_FILE, "a", encoding="utf-8")
    f.write(f"[{video_name}] === DASH MPD/m4s 생성 시작=== \n")

    for tag in resolution_tags:
        mp4_args.append(f"{final_mp4s[tag]}#video:name={tag}")
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
    f.write(f"MP4Box 실행: {cmd}\n")

    f.write(f"[{video_name}] === DASH MPD/m4s 생성 완료 === \n\n")
    f.close()


# (기타 함수들은 여기에 그대로 존재한다고 가정합니다)
def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        return 0

# ... extract_features, ai_predict 등 다른 모든 함수들 ...
def extract_features(input_path):
    stats = {}
    try:
        video_info = analyze_video(input_path, 10)
        if video_info is None: return {"video_path": input_path, "error": "Video open failed"}
        stats.update(video_info)
        mv_stats = extract_motion_vector(input_path)
        stats.update(mv_stats)
        mb_stats = get_macroblock_ratios(input_path)
        stats['Intra_macro'] = mb_stats.get("Intra_macro", 0)
        stats['Inter_macro'] = mb_stats.get("Inter_macro", 0)
        stats['Skip_macro'] = mb_stats.get("Skip_macro", 0)
        return stats
    except Exception as e:
        print(f"Error for {input_path}: {e}")
        return None
        
def analyze_video(video_path, sample_rate):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_size = get_file_size(video_path)/1024 if os.path.exists(video_path) else 0
        
        processed_frames, edge_density, pixel_entropy = 0, [], []

        if width <= 0 or height <= 0 or fps <= 0: return None

        while True:
            ret, frame = cap.read()
            if not ret: break
            if processed_frames % sample_rate != 0:
                processed_frames += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            edge_density.append(edge_density_lbp(gray))
            pixel_entropy.append(frame_entropy(gray))
            processed_frames += 1
        
        return {
            "video_path": video_path, "width": width, "height": height, "fps": fps,
            "frame_count": processed_frames, "edge_density": np.mean(edge_density) if edge_density else 0,
            "pixel_entropy": np.mean(pixel_entropy) if pixel_entropy else 0, "original_size": original_size
        }
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return None
    finally:
        cap.release()

def frame_entropy(frame, bins=256):
    hist, _ = np.histogram(frame.ravel(), bins=bins, range=(0, 256), density=True)
    return entropy(hist[hist > 0], base=2)

def edge_density_lbp(frame, P=8, R=1):
    lbp = local_binary_pattern(frame, P, R, method="uniform")
    edge_pixels = np.sum(lbp <= 4)
    return (edge_pixels / lbp.size) * 100

def calculate_direction_entropy(motion_x, motion_y, n_bins=16):
    if len(motion_x) == 0: return 0.0
    magnitude = np.sqrt(motion_x**2 + motion_y**2)
    valid_mask = magnitude > 1e-6
    if not np.any(valid_mask): return 0.0
    angles = np.arctan2(motion_y[valid_mask], motion_x[valid_mask])
    counts, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    probabilities = counts / np.sum(counts)
    return entropy(probabilities[probabilities > 0], base=2)

def extract_motion_vector(video_path, mv_threshold=1.0):
    cap = VideoCap()
    cap.open(video_path)
    all_mv_magnitudes, all_mv_x, all_mv_y, frame_motion_stats = [], [], [], []
    partition_size_counts = defaultdict(int)
    try:
        while True:
            success, frame, motion_vectors, frame_type, timestamp = cap.read()
            if not success: break
            if motion_vectors.shape[0] == 0: continue
            
            mvx = motion_vectors[:, 7] / motion_vectors[:, 9]
            mvy = motion_vectors[:, 8] / motion_vectors[:, 9]
            magnitudes = np.sqrt(mvx**2 + mvy**2)
            significant_mask = magnitudes >= mv_threshold
            
            all_mv_magnitudes.extend(magnitudes[significant_mask])
            all_mv_x.extend(mvx[significant_mask])
            all_mv_y.extend(mvy[significant_mask])
    finally:
        cap.release()
    if not all_mv_magnitudes: return {}
    return calculate_motion_statistics(np.array(all_mv_magnitudes), np.array(all_mv_x), np.array(all_mv_y), {}, []) # Dummy stats for now

def calculate_motion_statistics(magnitudes, mvx, mvy, partition_counts, frame_motion_stats):
    return {
        'mean_mv_magnitude': np.mean(magnitudes), 'std_mv_magnitude': np.std(magnitudes),
        'max_mv_magnitude': np.max(magnitudes), 'median_mv_magnitude': np.median(magnitudes),
        'mv_95_percentile': np.percentile(magnitudes, 95), 'motion_energy': np.sum(magnitudes**2) / len(magnitudes),
        'direction_entropy': calculate_direction_entropy(mvx, mvy), 'directional_consistency': 0, # Placeholder
        'partition_diversity': 0, 'frame_count': 0, 'mb16x16': 0, 'mb16x8': 0, 'mb8x8': 0,
        'motion_variance_across_frames': 0, 'motion_temporal_smoothness': 0, 'avg_significant_motion_ratio': 0
    }

def get_macroblock_ratios(video_path):
    cmd = ["ffmpeg", "-i", video_path, "-c:v", "libx264", "-preset", "ultrafast", "-f", "null", "-", "-x264-params", "log-level=debug"]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    _, stderr = proc.communicate()
    stats = {"Intra_macro": 0.0, "Inter_macro": 0.0, "Skip_macro": 0.0}
    # Simplified parsing for example
    return stats

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

# (8) 전체 실행
if __name__ == "__main__":

    CRF_MODEL = joblib.load(CRF_MODEL_PATH)
    MAXRATE_MODEL = joblib.load(MAXRATE_MODEL_PATH)
    SCALER_X = joblib.load(SCALER_X_PATH)
    SCALER_Y_CRF = joblib.load(SCALER_Y_CRF_PATH)
    SCALER_Y_MAX = joblib.load(SCALER_Y_MAX_PATH)


    for video_name, input_path in INPUT_VIDEOS:
        with open(LOG_FILE, "a") as f:
            f.write(f"=== [{video_name}] 영상 인코딩 시작 ===\n")

        OUT_DIR = f"./static/{video_name}"
        os.makedirs(OUT_DIR, exist_ok=True)

        print("1. 세그먼트 분할 (정밀 재인코딩 방식)")
        SEG_DIR = f"./static/{video_name}/segments"
        os.makedirs(SEG_DIR, exist_ok=True)
        seg_paths = split_video(input_path, SEG_DIR, segment_length=SEG_LEN, log_path=LOG_FILE)

        print("2. 세그먼트별 AI 인코딩 (TS 포맷으로 출력)")
        ai_segment_dir = f"./static/{video_name}/ai_segment_ts"
        os.makedirs(ai_segment_dir, exist_ok=True)
        ai_segs_by_res = encode_segments(
            seg_paths, ai_segment_dir, SCALER_X, SCALER_Y_CRF, SCALER_Y_MAX, CRF_MODEL, MAXRATE_MODEL)

        print("3. concat.txt 생성")
        concat_dir = f"./static/{video_name}/concat_dir"
        os.makedirs(concat_dir, exist_ok=True)
        concat_txts = make_concat_txt_multi_res(ai_segs_by_res, concat_dir)

        print("4. TS 병합 및 MP4로 재포장")
        merged_dir = f"./static/{video_name}/merged"
        os.makedirs(merged_dir, exist_ok=True)
        merged_mp4s = concat_segments_multi_res(concat_txts, merged_dir)

        # 5. 병합 mp4 재인코딩 단계는 더 이상 필요 없으므로 삭제합니다.
        # print("5. 병합 mp4 재인코딩") <-- 삭제

        print("6. mp4box로 dash 분할 + mpd 생성")
        mp4box_dir = f"./static/{video_name}/mp4box"
        os.makedirs(mp4box_dir, exist_ok=True)
        mp4box_dash_multi_res(
            merged_mp4s,  # reencode 단계가 사라졌으므로 merged_mp4s를 직접 전달
            mp4box_dir,
            segment_ms=SEG_LEN * 1000)

        print(f"=== [{video_name}] DASH 인코딩 완료 ===\n")

