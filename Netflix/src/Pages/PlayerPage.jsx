//Adaptive Streaming Player Page

import React from 'react'
import { useParams } from 'react-router-dom'
import vids from '../assets/videos.js'
import DashPlayer from '../componets/DashPlayer'

// PlayerPage 컴포넌트는 URL 파라미터에서 videoId를 추출하고, 해당 ID에 맞는 manifest를 찾아 DashPlayer에 전달합니다.
export default function PlayerPage() {
  const { videoId } = useParams() 

  // videoId는 문자열이니까 숫자로 바꿔서 조회
  const video = vids.find(v => v.id === parseInt(videoId, 10))

  if (!video) {
    return <div className="player-page--error">⚠️ 해당 영상을 찾을 수 없습니다.</div>
  }

  return (
    <div className="player-page">
      {/* DashPlayer에 manifest URL만 넘겨주면 dash.js가 adaptive streaming 해준다. */}
      <div className="player-page__player">
        <DashPlayer manifestUrl={video.manifest} />
      </div>
    </div>
  )
}


