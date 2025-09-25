"""
목적:
- CRF: 세그먼트/해상도별 1초 단위 비트레이트 CSV를 '전체 타임라인(0초 시작)'으로 재배치
- CBR: ffprobe로 mp4에서 packet 단위 size를 읽어 1초 단위 비트레이트 계산
- 두 시계열(CRF vs CBR)을 같은 해상도에서 비교 그래프 출력 (길이 130초로 고정)

필요:
- pip install pandas matplotlib
- ffprobe 설치(FFmpeg 포함)
"""

import os, json, subprocess, math , shutil
import pandas as pd
import matplotlib.pyplot as plt

# ================== 경로/설정 ==================
CRF_CSV_PATH = "../Netflix/public/husky/bitrate_logs/husky_bitrate_per_second.csv"
CBR_FILES = {
    "360p":  "output_cbr_360p.mp4",
    "480p":  "output_cbr_480p.mp4",
    "720p":  "output_cbr_720p.mp4",
    "1080p": "output_cbr_1080p.mp4",
}

# 해상도별 **목표 평균 비트레이트(kbps)** 와 스케일
CBR_TARGETS = { 
    "360p":  {"kbps": 800,  "scale": "640:360"}, 
    "480p":  {"kbps": 1200, "scale": "854:480"},  
    "720p":  {"kbps": 2500, "scale": "1280:720"},  
    "1080p": {"kbps": 5000, "scale": "1920:1080"},
}

GENERATE_CBR_ONCE = True  # CBR MP4 처음 한 번만 만들기
SOURCE_VIDEO = "../input/husky.mp4"

VIDEO_FPS = 24
SAVE_DIR = "./plots"
TITLE_PREFIX = "1s Averaged Bitrate over Time (CBR vs CRF)"

# **영상 실제 길이(초)**: 요청대로 130초로 고정
VIDEO_DURATION_SEC = 130  # [ADDED] 0~129초까지만 표시
# =================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# === (1) CRF CSV 읽기 & 0초 기준으로 정렬 ===
def load_crf_bitrate_csv(csv_path: str) -> dict:
    """
    반환: { "1080p": pd.Series(kbps, index=seconds[0..]) , ... }
    CSV 헤더: segment_name,time_second,bitrate_kbps
    segment_name: ai_seg_{N}_{RES}.mp4 (예: ai_seg_3_1080p.mp4)
    time_second: 0~9 (세그먼트 내부 초)
    """
    df = pd.read_csv(csv_path)

    # 해상도/세그먼트 번호 추출
    df["resolution"] = df["segment_name"].str.extract(r"_(360p|480p|720p|1080p)\.mp4$")
    df["seg_idx"] = df["segment_name"].str.extract(r"ai_seg_(\d+)_").astype(int)

    # **세그먼트 1부터 시작하는 데이터 → 0초 기준으로 오프셋 제거**
    min_seg = int(df["seg_idx"].min())           # [ADDED]
    base_offset = min_seg * 10                    # [ADDED] 10초 단위 세그먼트 가정
    df["global_sec"] = (df["seg_idx"] * 10 + df["time_second"]) - base_offset  # [CHANGED]

    # 0 이상만 남기고, 영상 길이(130초)로 자르기
    df = df[(df["global_sec"] >= 0) & (df["global_sec"] < VIDEO_DURATION_SEC)]  # [ADDED]

    series_by_res = {}
    for res, g in df.groupby("resolution"):
        g = g.sort_values(["global_sec"])
        s = pd.Series(g["bitrate_kbps"].values, index=g["global_sec"].values)

        # 같은 초에 중복이 있으면 평균으로 정리
        s = s.groupby(level=0).mean()  # [ADDED]

        # 0~129초 범위로 리샘플(결측은 내부만 선형 보간)
        idx = range(0, VIDEO_DURATION_SEC)  # [ADDED]
        s = s.reindex(idx)                  # [ADDED]
        s = s.interpolate(limit_direction="both")  # [CHANGED] 앞/뒤 소량 보간만
        series_by_res[res] = s
    return series_by_res

# === (2) CBR mp4에서 1s 단위 비트레이트(kbps) 측정 ===
def ffprobe_packets(video_path: str):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,size",
        "-of", "json", video_path
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    data = json.loads(out)
    return data.get("packets", [])

def measure_per_second_bitrate_kbps(video_path: str) -> pd.Series:
    """
    패킷 단위 크기를 1초 버킷으로 합산 → 초당 kbps로 변환
    결과는 0~(VIDEO_DURATION_SEC-1)로 슬라이스
    """
    packets = ffprobe_packets(video_path)
    if not packets:
        raise RuntimeError(f"No packets from ffprobe for {video_path}")

    # 전체 길이(초) 파악 (정보 출력용)
    dur_sec = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]).decode("utf-8").strip())

    # 1초 버킷 합산
    bucket = {}
    for p in packets:
        t = float(p.get("pts_time", 0.0))
        sz = int(p.get("size", 0))
        sec = int(math.floor(t))
        if 0 <= sec < VIDEO_DURATION_SEC:             # [ADDED] 130초로 제한
            bucket[sec] = bucket.get(sec, 0) + sz

    idx = list(range(0, VIDEO_DURATION_SEC))          # [ADDED]
    kbps_vals = []
    for s in idx:
        b = bucket.get(s, 0)
        kbps_vals.append((b * 8) / 1000.0)            # byte→bit→kbps

    s = pd.Series(kbps_vals, index=idx)
    s.index.name = "second"

    # 평균(실측) 출력
    file_bytes = os.path.getsize(video_path)
    avg_kbps = (file_bytes * 8 / 1000.0) / max(dur_sec, 1e-6)
    print(f"[INFO] {os.path.basename(video_path)} duration≈{dur_sec:.2f}s, "
          f"avg_bitrate≈{avg_kbps:.1f} kbps (file_size/duration)")
    return s

# === (3) 플롯 ===
def plot_compare(crf_s: pd.Series, cbr_s: pd.Series, res: str, save_dir: str):
    # **시간축을 0~129초로 고정**하고 그대로 사용 (불필요한 앞 채움 삭제)
    idx = range(0, VIDEO_DURATION_SEC)   # [CHANGED]

    crf_aligned = crf_s.reindex(idx)
    cbr_aligned = cbr_s.reindex(idx)

    # 내부 결측만 부드럽게 - 앞뒤는 이미 0~129로 맞춰짐
    crf_aligned = crf_aligned.interpolate(limit_direction="both")  # [CHANGED]
    cbr_aligned = cbr_aligned.interpolate(limit_direction="both")  # [CHANGED]

    plt.figure(figsize=(11.5, 3.8))
    plt.plot(idx, crf_aligned.values, label="CRF (measured kbps)")
    plt.plot(idx, cbr_aligned.values, label="CBR (measured kbps)")
    plt.title(f"{TITLE_PREFIX} — {res}")
    plt.xlabel("Time (s)")
    plt.ylabel("Avg bitrate per second (kbps)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    out = os.path.join(save_dir, f"cbr_vs_crf_{res}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"[SAVED] {out}")
    plt.close()

def encode_cbr_once(): 
    """SOURCE_VIDEO를 해상도별/목표 평균 비트레이트로 CBR 인코딩해 CBR_FILES에 저장"""
    if not os.path.exists(SOURCE_VIDEO):
        print(f"[WARN] SOURCE_VIDEO 미존재: {SOURCE_VIDEO} → CBR 생성 스킵")
        return
    for res, out_mp4 in CBR_FILES.items():
        if os.path.exists(out_mp4):  # 이미 있으면 건너뜀
            print(f"[SKIP] {res}: {out_mp4} 이미 존재")
            continue
        target = CBR_TARGETS[res]
        # FFmpeg: 평균 비트레이트(-b:v), 최대(-maxrate), 버퍼(-bufsize) 세트로 'CBR에 가까운' 스트림
        avg = f"{target['kbps']}k"
        maxr = f"{int(target['kbps']*1.1)}k"     # 살짝 여유 (10%)  [ADDED]
        buf  = f"{int(target['kbps']*2)}k"       # 버퍼 크기         [ADDED]
        scale = target["scale"]
        cmd = [
            "ffmpeg","-y","-i",SOURCE_VIDEO,
            "-vf", f"scale={scale}",
            "-c:v","libx264","-preset","veryfast","-tune","zerolatency",
            "-b:v", avg, "-maxrate", maxr, "-bufsize", buf,
            "-x264-params","nal-hrd=cbr:force-cfr=1",
            "-r","30",  # 필요 시 조정
            "-an", out_mp4
        ]
        print("[ENCODE]", " ".join(cmd))
        subprocess.check_call(cmd)

def main():
    # 1) CRF 읽기
    crf_dict = load_crf_bitrate_csv(CRF_CSV_PATH)

    if GENERATE_CBR_ONCE:
        encode_cbr_once()  

    # 2) 해상도별 CBR 측정 & 비교
    for res, cbr_file in CBR_FILES.items():
        if not os.path.exists(cbr_file):
            print(f"[SKIP] {res}: {cbr_file} 없음")
            continue

        cbr_series = measure_per_second_bitrate_kbps(cbr_file)

        if res not in crf_dict:
            print(f"[WARN] CRF CSV에 {res} 데이터가 없음")
            continue

        plot_compare(crf_dict[res], cbr_series, res, SAVE_DIR)

if __name__ == "__main__":
    main()