/* throttle-sw.js */
/// <reference lib="webworker" />
// 프로필 정의
const PROFILES = {
  off:  { bps: Infinity, latency: 0 },           // 제한 없음
  slow: { bps: 700_000, latency: 120 },        // 700kbps, 120ms RTT
  fast: { bps: 10_000_000, latency: 40 },        // 10Mbps, 40ms
};

let current = PROFILES.off;

// 클라이언트가 보낸 프로필 변경 수신
self.addEventListener('message', (ev) => {
  const { type, profile } = ev.data || {};
  if (type === 'SET_PROFILE' && PROFILES[profile]) {
    current = PROFILES[profile];
    self.clients.matchAll({ includeUncontrolled: true, type: 'window' })
      .then(clients => clients.forEach(c => c.postMessage({
        type: 'PROFILE_CHANGED',
        profile,
        bps: current.bps,
        latency: current.latency,
      })));
  }
});

// install/activate
self.addEventListener('install', (e) => e.waitUntil(self.skipWaiting()));
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

// throttle 대상: mpd, m4s, init, mp4(세그먼트) 등
const THROTTLE_MATCH = /\.(mpd|m4s|mp4|m4v|cmfv?)($|\?)/i;
const BYPASS_METHODS = new Set(['HEAD', 'OPTIONS']);

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // 캐시 우회 & 비대상/메소드 우회
  if (!THROTTLE_MATCH.test(new URL(req.url).pathname) ||
      BYPASS_METHODS.has(req.method) ||
      current.bps === Infinity) {
    event.respondWith(fetch(req, { cache: 'no-store' }));
    return;
  }

  event.respondWith(handleThrottled(req));
});

async function handleThrottled(request) {
  // 원본 응답 가져오기 (Range 포함)
  const upstream = await fetch(request, { cache: 'no-store' });

  // 바이트 스트리밍 가능한 경우에만 스로틀
  const reader = upstream.body?.getReader?.();
  if (!reader) return upstream;

  const { bps, latency } = current;

  // 최초 지연(왕복 지연 감안)
  if (latency > 0) {
    await delay(latency);
  }

  // bps 기준으로 청크 크기 계산 (50ms 당 전송량)
  // 너무 작으면 overhead 커지니 하한/상한 둠.
  const sliceIntervalMs = 50;
  const bytesPerTick = Math.max(8 * 1024, Math.min((bps * sliceIntervalMs) / 1000 / 8, 256 * 1024));

  const stream = new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        controller.close();
        return;
      }

      // value를 bytesPerTick 단위로 나눠서 천천히 push
      let offset = 0;
      while (offset < value.byteLength) {
        const end = Math.min(offset + bytesPerTick, value.byteLength);
        controller.enqueue(value.slice(offset, end));
        offset = end;
        // 전송 간격
        if (offset < value.byteLength) {
          await delay(sliceIntervalMs);
        }
      }
    },
    cancel(reason) {
      try { reader.cancel(); } catch {}
    }
  });

  // 헤더 복사 & 캐시 무효화
  const headers = new Headers(upstream.headers);
  headers.set('cache-control', 'no-store');

  return new Response(stream, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers
  });
}

function delay(ms) {
  return new Promise((res) => setTimeout(res, ms));
}
