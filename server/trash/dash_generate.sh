#!/bin/bash

# static/ 폴더 아래 모든 비디오 폴더 대상
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_NAME="husky"
BASE_DIR="${SCRIPT_DIR}/static/${VIDEO_NAME}"

RES_LABELS=("360p" "480p" "720p" "1080p")
REP_IDS=(0 1 2 3)

for VIDEO_DIR in "$BASE_DIR"/*/; do
    SEGMENT_DIR="${VIDEO_DIR}"  # m4s가 저장된 위치
    echo "[▶️] 처리 중: $VIDEO_DIR"

    for i in "${!RES_LABELS[@]}"; do
        label="${RES_LABELS[$i]}"
        rep_id="${REP_IDS[$i]}"

        # chunk-stream${rep_id}-0000.m4s를 이용해 init 생성
        FIRST_SEG="${SEGMENT_DIR}/chunk-stream${rep_id}-0000.m4s"
        INIT_PATH="${SEGMENT_DIR}/init-stream${rep_id}.mp4"

        if [ -f "$FIRST_SEG" ]; then
            ffmpeg -y -i "$FIRST_SEG" -c copy \
              -movflags +frag_keyframe+empty_moov+default_base_moof \
              -f mp4 "$INIT_PATH"

            echo "  [✅] init-stream${rep_id}.mp4 생성 완료"
        else
            echo "  [⚠️] $FIRST_SEG 없음 - 건너뜀"
        fi
    done
done

echo "[✅] 모든 해상도의 init-stream 생성 완료"
