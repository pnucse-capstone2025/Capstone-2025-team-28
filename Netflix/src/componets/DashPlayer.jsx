// DashPlayer.jsx — 전체 교체본 (CSV 로딩 + 비트레이트 차트 + 네트워크 스로틀 버튼 + SW 연동 + 'slow' 강제 저화질)
// =============================================================================
import React, { useRef, useEffect, useState } from 'react';
import Chart from 'chart.js/auto';
import Papa from 'papaparse';
import './DashPlayer.css';

// ========================[ 상수 ]==============================================
// encode.py 의 SEG_LEN 과 반드시 일치해야 전역 시간축이 맞음
const SEG_LEN = 2; // [KEEP]

// [KEEP] UI에서 선택한 네트워크 프로파일 → dash.js 강제 품질 제한에 매핑
//   - 'slow'일 때 대략 1.5Mbps 이하만 허용되도록 제한(없으면 가능한 최저 품질로 강제)
//   - 'fast' / 'off' 는 자동 ABR 재개
const PROFILE_CAP_KBPS = {
  slow: 1500,  // 느림 모드 상한 (kbps)
  fast: Infinity,
  off:  Infinity,
};

// ==================[ 유틸: 비디오 이름/세그먼트 파싱 ]===========================
// /stream/{video}/manifest.mpd 형태에서 video_name 추출
function extractVideoName(manifestUrl) {
  try {
    const u = new URL(manifestUrl, window.location.href);
    const parts = u.pathname.split('/').filter(Boolean);
    const idx = parts.indexOf('stream');
    if (idx !== -1 && parts[idx + 1]) return parts[idx + 1];
    if (parts.length >= 2) return parts[parts.length - 2];
  } catch (e) {
    console.warn('[extractVideoName] failed:', e);
  }
  return '';
}

// MP4Box가 만든 m4s 파일명에서 해상도/세그먼트 인덱스 추출
// 예: merged_ai_720p_dash12.m4s, 720p_dash12.m4s, anything_720p_dash12.m4s 지원
function parseMp4boxM4s(url) {
  const name = url.split('?')[0].split('/').pop();
  let m = name.match(/merged_ai_(\d+p)_dash(\d+)\.m4s$/);
  if (m) return { resolution: m[1], segIdx: parseInt(m[2], 10) };
  m = name.match(/(\d+p)_dash(\d+)\.m4s$/);
  if (m) return { resolution: m[1], segIdx: parseInt(m[2], 10) };
  m = name.match(/_(\d+p)_dash(\d+)\.m4s$/);
  if (m) return { resolution: m[1], segIdx: parseInt(m[2], 10) };
  console.warn('[parseMp4boxM4s] Unmatched m4s name:', name);
  return null;
}

// =========================[ 컴포넌트 ]=========================================
export default function DashPlayer({ manifestUrl }) {
  // ---------- refs ----------
  const videoRef = useRef(null);
  const chartRef = useRef(null);
  const playerRef = useRef(null);
  const chartInstanceRef = useRef(null);

  // ---------- state ----------
  const [bitrateLog, setBitrateLog] = useState([]);
  const [currentSegment, setCurrentSegment] = useState(null);

  const [uiProfile, setUiProfile] = useState('off');     // [KEEP] 네트워크 버튼 active 표시
  const currentProfileRef = useRef('off');               // [KEEP] SW로 보낼 현재 프로파일 저장

  // ==================[ CSV 로딩: manifest 오리진 기준 ]=========================
  useEffect(() => {
    if (!manifestUrl) return;
    const videoName = extractVideoName(manifestUrl);
    if (!videoName) {
      console.warn('⚠️ video_name 추출 실패 – CSV 로딩 중단');
      return;
    }

    // [FIX] manifestUrl의 origin(예: http://localhost:8000) 기준으로 CSV 요청
    const base = new URL(manifestUrl, window.location.href);
    const csvUrl = new URL(
      `/stream/${videoName}/bitrate/${videoName}_bitrate_per_second.csv`,
      base.origin
    ).toString();

    console.log('[CSV fetch] url:', csvUrl);

    fetch(csvUrl, { mode: 'cors' })
      .then((res) => {
        console.log('[CSV fetch] status:', res.status);
        if (!res.ok) throw new Error(`CSV 요청 실패: ${res.status}`);
        return res.text();
      })
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          dynamicTyping: true,
          transformHeader: (h) => h.trim(),
          complete: (result) => {
            console.log('✅ CSV 샘플:', result.data.slice(0, 3));
            setBitrateLog(result.data);
          },
        });
      })
      .catch((e) => console.error('❌ CSV 로딩 오류:', e.message));
  }, [manifestUrl]);

  // =====================[ dash.js 초기화 + 차트 생성 ]==========================
  useEffect(() => {
    const dashjs = window.dashjs;
    if (!dashjs || typeof dashjs.MediaPlayer !== 'function') {
      console.error('❌ dash.js 로딩 실패');
      return;
    }

    console.log('[BUILD_MARK] 20250914D DashPlayer loaded'); // [ADDED] 배포 확인용

    const player = dashjs.MediaPlayer().create();
    playerRef.current = player;

    // [CHANGED] 미지원 옵션 제거(경고 방지) + 안전한 기본치만 설정
    player.updateSettings({
      streaming: {
        abr: {
          autoSwitchBitrate: { video: true },
          initialBitrate: { video: 800 }, // kbps (초반 과상향 방지)
        },
        buffer: {
          bufferTimeAtTopQuality: 8,
          bufferToKeep: 20,
        },
      },
    });

    player.initialize(videoRef.current, manifestUrl, true);

    // 차트 생성(심플)
    chartInstanceRef.current = new Chart(chartRef.current, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Bitrate (kbps)', data: [] }] },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { title: { display: true, text: 'Time (s)' }, beginAtZero: true, min: 0 },
          y: { title: { display: true, text: 'Bitrate (kbps)' } },
        },
      },
    });

    return () => {
      try { player.reset(); } catch {}
      chartInstanceRef.current?.destroy();
    };
  }, [manifestUrl]);

  // ============[ 세그먼트 로딩 완료 이벤트에서 차트/로그 갱신 ]===============
  useEffect(() => {
    if (!bitrateLog.length || !playerRef.current) return;
    const player = playerRef.current;

    const handleFragment = (e) => {
      const url = e?.request?.url;
      const isVideo = e?.request?.mediaType === 'video';
      if (!url || !isVideo) return;

      const parsed = parseMp4boxM4s(url);
      if (!parsed) return;

      const { resolution, segIdx } = parsed; // segIdx = 1,2,3...
      const segmentName = `ai_seg_${segIdx}_${resolution}.mp4`; // CSV의 segment_name 규칙과 일치
      setCurrentSegment(segmentName);

      const rows = bitrateLog.filter(r => String(r.segment_name).trim() === segmentName);
      if (!rows.length) {
        console.warn(`[CSV 미존재] ${segmentName}`);
        return;
      }

      const chart = chartInstanceRef.current;

      // [KEEP] SEG_LEN(=2초) 기준으로 전역 시간축 채우기
      for (let localSec = 0; localSec < SEG_LEN; localSec++) {
        const row = rows.find(r => parseInt(r.time_second) === localSec);
        const globalSec = (segIdx - 1) * SEG_LEN + localSec;
        const globalSecStr = String(globalSec);

        if (!row || isNaN(parseFloat(row.bitrate_kbps))) continue;

        const bitrate = parseFloat(row.bitrate_kbps);
        const idx = chart.data.labels.indexOf(globalSecStr);

        if (idx !== -1) chart.data.datasets[0].data[idx] = bitrate;
        else {
          chart.data.labels.push(globalSecStr);
          chart.data.datasets[0].data.push(bitrate);
        }
      }

      chart.update();
      console.log(`🔍 세그먼트: ${segmentName} (SEG_LEN=${SEG_LEN}s)`);
    };

    player.on(window.dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, handleFragment);
    return () => {
      player.off(window.dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, handleFragment);
    };
  }, [bitrateLog]);

  // =====================[ 1초 주기 콘솔 로그 (중복 방지) ]======================
  const printedRef = useRef(new Set());
  useEffect(() => { printedRef.current = new Set(); }, [currentSegment]);

  useEffect(() => {
    if (!currentSegment || bitrateLog.length === 0) return;
    const rows = bitrateLog.filter(r => String(r.segment_name).trim() === currentSegment);
    const timer = setInterval(() => {
      for (let sec = 0; sec < SEG_LEN; sec++) {
        const key = `${currentSegment}-${sec}`;
        if (printedRef.current.has(key)) continue;
        const row = rows.find(r => parseInt(r.time_second) === sec);
        if (row) {
          console.log(`📦 [${currentSegment}] ${sec}s → ${row.bitrate_kbps} kbps`);
          printedRef.current.add(key);
        }
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [currentSegment, bitrateLog]);

  // ===================[ SW 등록 + 초기 프로파일 전송 ]=========================
  useEffect(() => {
    const onMsg = (e) => {
      // [KEEP] 프로파일 변경 시 dash.js 품질 제한/해제
      if (e.data?.type === 'PROFILE_CHANGED') {
        const p = e.data.profile;
        console.log('[SW] PROFILE_CHANGED →', p);
        applyProfileToDash(p);
      }
      if (e.data?.type === 'LOG') console.log(e.data.msg);
    };
    navigator.serviceWorker?.addEventListener('message', onMsg);

    (async () => {
      if ('serviceWorker' in navigator) {
        try {
          // [KEEP] 네트워크 스로틀 Service Worker 등록
          await navigator.serviceWorker.register('/throttle-sw.js', { scope: '/' });
          await navigator.serviceWorker.ready;

          if (!navigator.serviceWorker.controller) {
            navigator.serviceWorker.addEventListener('controllerchange', () => {
              sendProfile(currentProfileRef.current); // 컨트롤러 생기면 현재 프로파일 재전송
            });
          } else {
            sendProfile('off'); // 초기값 전송
          }
        } catch (err) {
          console.warn('SW 등록 실패:', err);
        }
      }
    })();

    return () => navigator.serviceWorker?.removeEventListener('message', onMsg);
  }, []);

  // ===================[ 네트워크 프로파일 전송 함수 ]===========================
  function sendProfile(p) {
    currentProfileRef.current = p;
    setUiProfile(p);

    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.controller.postMessage({
        type: 'SET_PROFILE',
        profile: p,
      });
      console.log(`[UI] Network profile → ${p}`);
      applyProfileToDash(p); // [KEEP] 즉시 dash에도 반영
    } else {
      console.warn('[SW] 아직 제어권 없음');
    }
  }

  // [KEEP] 프로파일을 dash.js 에 강제로 반영 (느림이면 낮은 품질 고정, 그 외 자동 복귀)
  function applyProfileToDash(profile) {
    const player = playerRef.current;
    if (!player) return;

    const cap = PROFILE_CAP_KBPS[profile] ?? Infinity;

    try {
      // [KEEP] v4 settings 업데이트
      const settings = player.getSettings?.() || {};
      const next = {
        ...settings,
        streaming: {
          ...(settings.streaming || {}),
          abr: {
            ...((settings.streaming && settings.streaming.abr) || {}),
            maxBitrate: { video: isFinite(cap) ? cap : Infinity },
          },
        },
      };
      player.updateSettings?.(next);
    } catch {}

    if (profile === 'slow') {
      try {
        let repList = null;
        if (typeof player.getBitrateListFor === 'function') {
          const arr = player.getBitrateListFor('video');
          if (Array.isArray(arr)) {
            repList = arr.map((v, i) => {
              const kbps = v > 100000 ? Math.round(v / 1000) : Math.round(v);
              return { kbps, q: i };
            });
          }
        } else if (typeof player.getBitrateInfoListFor === 'function') {
          const arr = player.getBitrateInfoListFor('video');
          if (Array.isArray(arr)) {
            repList = arr.map(x => ({ kbps: Math.round(x.bitrate / 1000), q: x.qualityIndex }));
          }
        }

        if (repList && repList.length) {
          const eligible = repList.filter(x => x.kbps <= cap).sort((a, b) => b.kbps - a.kbps);
          const targetQ = (eligible[0]?.q ?? repList[repList.length - 1].q);
          player.setAutoSwitchQualityFor('video', false);
          player.setQualityFor('video', targetQ, true);
          console.log(`[ABR] force qualityIndex=${targetQ} (cap=${cap}kbps)`);
        } else {
          console.warn('[ABR] bitrate list unavailable');
        }
      } catch (e) {
        console.warn('[ABR] force quality failed:', e);
      }
    } else {
      try {
        player.setAutoSwitchQualityFor('video', true);
        console.log('[ABR] autoSwitch 재개');
      } catch {}
    }
  }

  // =============================[ UI ]=========================================
  return (
    <div className="dash-container">{/* [ADDED] 상단 여백/가운데 정렬 래퍼 */} 
      <div className="dash-video-wrapper">{/* [ADDED] 16:9 고정 비율 박스 */}
        <video ref={videoRef} className="dash-video" controls />
      </div>

      {/* 네트워크 전환 버튼 (Fast/Slow/Off) */}
      <div className="throttle-controls">
        <span className="throttle-label">Network:</span>
        <button
          className={`throttle-btn ${uiProfile === 'fast' ? 'active' : ''}`}
          onClick={() => sendProfile('fast')}
        >
          Fast
        </button>
        <button
          className={`throttle-btn ${uiProfile === 'slow' ? 'active' : ''}`}
          onClick={() => sendProfile('slow')}
        >
          Slow
        </button>
        <button
          className={`throttle-btn ${uiProfile === 'off' ? 'active' : ''}`}
          onClick={() => sendProfile('off')}
        >
          Off
        </button>
      </div>

      <div className="dash-chart">
        <canvas ref={chartRef} />
      </div>
    </div>
  );
}
// =============================================================================
