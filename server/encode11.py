# 3개 영상 각각 4가지 해상도로 인코딩 (이건 실행 못해봄. 실행하면 wsl이 튕긴다.. 컴터 이슈)
#encode9.py에서 영상 1개로는 잘 동작하는 거 확인함.

#실행순서
#python3 encode.py 로 실행. F5 말고, 터미널로 하기
#cd server
#python3 -m uvicorn main:app --reload
#Netflix 폴더에서 npm run dev 실행


''' 인코딩 했는데 mpd파일 내용이 텅 비었을 때 다음 코드를 터미널(server 폴더)에서 실행 하면 mpd파일 생성됨
wsl 서버 문제 인듯 

MP4Box -dash 10000 -frag 10000 -rap -profile dashavc264:live \
 -out ./static/husky/manifest.mpd \
 ./static/husky/merged_ai_fixed_360p.mp4 \
 ./static/husky/merged_ai_fixed_480p.mp4 \
 ./static/husky/merged_ai_fixed_720p.mp4 \
 ./static/husky/merged_ai_fixed_1080p.mp4

'''

import os
import subprocess
import joblib
import numpy as np
import pandas as pd
from collections import Counter
import glob
import re


INPUT_VIDEOS = [
    ("lol", "../input/lol.mp4")
]
#INPUT_MP4 = "../input/husky.mp4"
TMP_SEG_DIR = "./static/husky/segments"
OUT_DIR = "./static/husky"
MODEL_PATH = "./large_kinetics_rf_model.pkl"
SCALER_X_PATH = "./large_kinetics_rf_scaler_X.pkl"
SCALER_Y_PATH = "./large_kinetics_rf_scaler_y.pkl"
MV_EXTRACT_PY = "./mv_extractor/extract_mvs.py"
SEG_LEN = 10

# 해상도 정의
resolutions = {
    "360p":  "640x360",
    "480p":  "854x480",
    "720p":  "1280x720",
    "1080p": "1920x1080"
}
resolution_tags = list(resolutions.keys())

# 인코딩 파라미터 확인용 로그
parameter_log_file = "./large_parameter_log.txt"

os.makedirs(TMP_SEG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

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

def extract_features(input_path, segment_motion_vector_dir):
    # 1. 모션벡터 추출
    extract_script = "../mv_extractor/extract_mvs.py"
    cmd = [
        "python3", extract_script,
        input_path,
        "-d", segment_motion_vector_dir
    ]
    subprocess.run(cmd)
    print("===================")

    # 2. 모션 벡터 통계
    csv_dir = os.path.join(segment_motion_vector_dir, "motion_vectors")
    npy_files = glob.glob(os.path.join(csv_dir, '*.npy'))
    for npy_file in npy_files:
        data = np.load(npy_file)
        csv_file = npy_file.replace('.npy', '.csv')
        np.savetxt(csv_file, data, delimiter=",")
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    all_magnitudes = []
    for csv_file in csv_files:
        if os.path.getsize(csv_file) == 0:
            continue
        df = pd.read_csv(csv_file)
        df.columns = ['ref_offset', 'block_w', 'block_h', 'ref_x', 'ref_y', 'cur_x', 'cur_y', 'mv_x', 'mv_y', 'mv_scale']
        df['motion_x'] = df['mv_x'] / df['mv_scale']
        df['motion_y'] = df['mv_y'] / df['mv_scale']
        df['magnitude'] = np.sqrt(df['motion_x']**2 + df['motion_y']**2)
        all_magnitudes.extend(df['magnitude'])
    if len(all_magnitudes) == 0:
        # 세그먼트가 너무 짧아서 벡터 없음
        return [0.0]*7

    motion_x_arr = np.array(all_magnitudes)
    
    # # 3. 프레임 비율 분석 (I/P/B)
    # frame_types_file = os.path.join(segment_motion_vector_dir, "frame_types.txt")
    # I_ratio = P_ratio = B_ratio = 0.0
    # if os.path.exists(frame_types_file):
    #     with open(frame_types_file, 'r') as f:
    #         frame_types = [line.strip() for line in f if line.strip()]
    #     counts = Counter(frame_types)
    #     total = len(frame_types)
    #     if total > 0:
    #         I_ratio = round(counts.get('I', 0)/total, 3)
    #         P_ratio = round(counts.get('P', 0)/total, 3)
    #         B_ratio = round(counts.get('B', 0)/total, 3)
    
    # 4. 매크로블록 비율 분석 (Intra/Inter/Skip)
    cmd = [
        'ffmpeg',
        '-debug', 'mb_type',
        '-i', input_path,
        '-f', 'null',
        '-'
    ]
    
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = process.stderr
    symbol_map = {'Intra': ['I', 'i'], 'Inter': ['d', '<', '>', 'X'], 'Skip': ['S']}
    mb_lines = [line for line in stderr.splitlines() if re.search(r'([dI<SX>iS]\s*){10,}', line)]
    total_counts = {'Intra': 0, 'Inter': 0, 'Skip': 0}
    for line in mb_lines:
        symbols = re.findall(r'[dI<SX>iS]', line)
        for k, v in symbol_map.items():
            total_counts[k] += sum(symbols.count(s) for s in v)
    total_blocks = sum(total_counts.values())
    Intra_ratio = Inter_ratio = Skip_ratio = 0.0
    if total_blocks > 0:
        Intra_ratio = round(total_counts['Intra'] / total_blocks, 3)
        Inter_ratio = round(total_counts['Inter'] / total_blocks, 3)
        Skip_ratio = round(total_counts['Skip'] / total_blocks, 3)

    # 5. 반환 (항상 10개)
    features = [
        float(np.mean(motion_x_arr)),   # mean_magnitude
        float(np.max(motion_x_arr)),    # max_magnitude
        float(np.std(motion_x_arr)),    # std_magnitude
        float(len(motion_x_arr)),       # motion_vector_count
        Intra_ratio,                    # Intra
        Inter_ratio,                    # Inter
        Skip_ratio                      # Skip
    ]
    print(features)
    
    return features


# (1) 세그먼트 분할 (한 번만, 예: 720p 기준)
def split_video(input_path, segment_dir, segment_length=10):
    from math import ceil
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", input_path
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())
    num_segments = ceil(duration / segment_length)
    seg_paths = []
    for f in os.listdir(segment_dir):
        if f.startswith("segment_") and f.endswith(".mp4"):
            os.remove(os.path.join(segment_dir, f))
    for i in range(num_segments):
        out = os.path.join(segment_dir, f"segment_{i}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(i * segment_length),
            "-i", input_path,
            "-t", str(segment_length),
            "-an",
            "-c:v", "copy",
            out
        ]
        subprocess.run(cmd, check=True)
        seg_paths.append(out)
    print("세그먼트 파일 생성:", seg_paths)
    return seg_paths

# (2) 모션벡터/AI 특성 추출 함수 (변경 없음, 생략)

# (3) AI 인코딩 + 해상도별 저장
def ai_encode_segments_multi_res(seg_paths, out_dir, scaler_X, scaler_y, model):
    ai_segs_by_res = {tag: [] for tag in resolution_tags}
    
    for i, seg_path in enumerate(seg_paths):
        mv_dir = os.path.join(out_dir, f"mv_{seg_path}")
        os.makedirs(mv_dir, exist_ok=True)
        features = extract_features(seg_path, mv_dir)
        
        for tag, res in resolutions.items():
            seg_name = f"ai_seg_{i}_{tag}.mp4"
            out_mp4 = os.path.join(out_dir, seg_name)
            
            # --- AI 특징 추출
            crf, maxrate = ai_predict(features, scaler_X, scaler_y, model)
            
            with open(parameter_log_file, "a") as f:  # 'a'는 append 모드 (기존 내용 뒤에 추가)
                f.write(f"[AI인코딩] segment_{i} {tag}: CRF={crf}, maxrate={maxrate}\n")
            print(f"[AI인코딩] segment_{i} {tag}: CRF={crf}, maxrate={maxrate}")
            
            # --- 인코딩 (해상도별)
            cmd = [
                "ffmpeg", "-y", "-i", seg_path,
                "-vf", f"scale={res}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", str(crf),
                "-maxrate", str(int(maxrate)),
                "-bufsize", "4000k",
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
    model = joblib.load(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    for video_name, input_path in INPUT_VIDEOS:
        
        print(f"\n=== [{video_name}] 영상 인코딩 시작 ===")
        with open(parameter_log_file, "a") as f:  # 'a'는 append 모드 (기존 내용 뒤에 추가)
            f.write(f"=== [{video_name}] 영상 인코딩 시작 ===")
            
        # 각 영상별 출력 폴더 지정
        TMP_SEG_DIR = f"./static/{video_name}/segments"
        OUT_DIR = f"./static/{video_name}"
        os.makedirs(TMP_SEG_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)

        print("1. 세그먼트 분할")
        seg_paths = split_video(input_path, TMP_SEG_DIR, SEG_LEN)
        print("2. 세그먼트별 AI 인코딩 (해상도 4개)")
        ai_segs_by_res = ai_encode_segments_multi_res(seg_paths, OUT_DIR, scaler_X, scaler_y, model)
        print("3. concat.txt 생성")
        concat_txts = make_concat_txt_multi_res(ai_segs_by_res, OUT_DIR)
        print("4. 병합 mp4 생성")
        merged_mp4s = concat_segments_multi_res(concat_txts, OUT_DIR)
        print("5. 병합 mp4 재인코딩")
        fixed_mp4s = reencode_mp4_multi_res(merged_mp4s, OUT_DIR)
        print("6. mp4box로 dash 분할 + mpd 생성 (1개)")
        mp4box_dash_multi_res(fixed_mp4s, OUT_DIR, segment_ms=SEG_LEN*1000)
        print(f"=== [{video_name}] DASH 인코딩 완료 ===\n")
