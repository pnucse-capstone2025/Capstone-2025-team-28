[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nRcUn8vA)

# 1. 프로젝트 배경
## 1.1. 국내외 시장 현황 및 문제점
### 국내외 시장 현황

  현대 사회의 미디어 소비 패러다임은 OTT(Over-The-Top) 플랫폼을 중심으로 급격하게 재편되고 있다. YouTube, Netflix와 같은 스트리밍 서비스는 더 이상 단순한 선택지를 넘어, 현대인의 일상에 깊숙이 자리 잡은 필수적인 여가 활동으로 부상했다.

  2024년 문화체육관광부에서 발표한 「국민문화예술 여가활동 조사 결과 발표」에 의하면, **TV 시청과 온라인/모바일 동영상 시청**이 가장 많이 참여하는 세부 여가 활동이라고 집계되었다. <br><br>
  <img width="600" height="336" alt="Image" src="https://github.com/user-attachments/assets/6c25e31c-1f0d-4a1f-9a13-1eef8a7cd455" />

  그리고 Nielsen의 「2025년 5월 The Gauge™ 보고서」에 따르면 2025년 5월 스트리밍 서비스의 TV 시청률이 44.8%를 기록하며 사상 최고치를 달성했고, 이는 **스트리밍이 미디어 시장의 주류**가 되었음을 증명한다. <br>
  <img width="600" height="528" alt="Image" src="https://github.com/user-attachments/assets/56eb545f-69b2-4b5d-a619-5c0d81f48e9a" />


### 기존 적응형 스트리밍의 문제점

  기존의 스트리밍 시스템(DASH)은 다음과 같은 세 가지 주요 문제점을 가지고 있다.
#### 1. 비효율적인 비트레이트(Bitrate) 관리
영상 콘텐츠는 장면마다 복잡도가 다르다. 액션 영화의 화려한 전투씬과 정적인 인터뷰 영상이 동일한 데이터 처리량을 요구하지 않는 것은 당연하다. 하지만 기존 스트리밍 시스템은 영상의 특성을 무시하고 네트워크 상황에만 의존하여 모든 장면에 일괄적인 품질 규칙을 적용한다. 그 결과, 복잡한 장면에서는 **품질이 깨지고**, 단순한 장면에서는 **데이터가 낭비**되는 비효율이 발생한다.

#### 2. 서버 스토리지 부담 증가
다양한 네트워크 환경과 디바이스에 대응하기 위해, 스트리밍 서비스는 수많은 버전(해상도, 비트레이터별)의 영상을 미리 인코딩하여 저장해야 한다. 이는 **서버 스토리지 비용의 기하급수적인 증가**로 이루어진다.

#### 3. 잦은 품질 변경으로 인한 QoE 저하
네트워크가 불안정한 환경(예: 이동 중인 대중교통)에서는 영상의 화질이 수시로 변경된다. 이러한 잦은 변화는 시청자의 몰입을 방해하고, 이는 곧 **사용자 경험(QoE)의 심각한 저하**로 이어진다.

## 1.2. 필요성과 기대효과
장면의 복잡도에 따라 비트레이트를 동적으로 할당하여, 불안정한 네트워크에서도 사용자가 체감하는 **화질 저하를 최소화하고 안정적인 시청 경험을 제공**한다. 또한 단순한 장면의 데이터 사용량은 줄이고 복잡한 장면에 집중하여 전체 파일 크기를 최적화하는데, 이는 **서버 스토리지와 CDN 비용 절감**으로 직접 이어진다.

# 2. 개발 목표
## 2.1. 목표 및 세부 내용
#### 1. 영상 특성 기반 최적화 모델 개발
영상의 장면 복잡도와 같은 특성을 자동으로 추출하고, 이를 기반으로 **최적의 인코딩 파라미터(CRF, Max Rate)를 예측**하는 AI 모델을 구현한다.

#### 2. 효율적인 적응형 스트리밍 시스템 구축
AI 모델이 예측한 파라미터를 적용하여 최적화된 비디오 세그먼트를 생성한다. 이 세그먼트를 MPEG-DASH 표준에 맞춰 패키징하여, **모든 환경에서 안정적으로 재생되는 스트리밍 시스템을 구축**한다. 또한, 기존 방식보다 적은 수의 인코딩 조합을 사용하여 **서버 스토리지 효율**을 높인다. 

#### 3. 사용자용 웹 서비스 개발 및 배포
사용자가 영상을 시청하고 관리할 수 있는 **웹 어플리케이션을 개발하고 배포**한다.
* 주요 기능: 회원 관리(가입, 로그인), 시청 기록 관리, 즐겨찾기 등

### 2.2. 기존 서비스 대비 차별성 
**MPEG-DASH(Dynamic Adaptive Streaming over HTTP)** 는 적응형 HTTP 스트리밍 기술의 대표적인 구현 표준으로, 서버가 여러 해상도와 비트레이트 조합으로 인코딩 된 영상 세그먼트를 미리 생성하여 저장해두고 클라이언트가 현재 네트워크 상황을 보고 예측하여, 어떤 세그먼트를 받을지 실시간으로 결정해서 재생하는 방식이다.<br>
<img width="600" height="460" alt="Image" src="https://github.com/user-attachments/assets/b9b8596c-91c6-44c9-b802-3911ca8b295b" />

기존 MPEG-DASH 기술은 네트워크 상태만 고려하지만, 본 프로젝트는 **영상 콘텐츠의 고유한 특성까지 분석**하여 비효율을 개선한다.

| 장면 유형 (Scene Type) | 기존 DASH의 문제점 | 본 프로젝트의 해결 방식 | 핵심 목표 |
| :--- | :--- | :--- | :--- |
| **복잡한 장면**<br>(액션, 스포츠 등) | 네트워크 저하 시, 비트레이트가 부족해져 **품질 붕괴 (아티팩트) 발생** | **인간 시각 시스템(HVS) 특성**을 활용한 지각적 압축 (VMAF 기반) | **체감 품질 저하 최소화** |
| **단순한 장면**<br>(풍경, 인터뷰 등) | 네트워크 양호 시, 필요 이상의 비트레이트를 할당하여 **데이터 낭비** 및 **품질 역설** 발생 | 시각적 손실이 없는 임계점을 찾아 **정교하게 비트레이트 최적화** | **낭비 방지 및 완벽한 품질 유지** |

## 2.3. 사회적 가치 도입 계획 
### 1. 공공성
본 프로젝트는 **제한된 네트워크 환경에서도 안정적인 영상 시청을 지원하여 공공 이익에 기여**한다.
* 상대적으로 네트워크 인프라가 열악한 지역의 정보 접근성을 높여 **디지털 격차를 해소**한다.
* 재난과 같은 위기 상황에서 네트워크 자원을 효율적으로 사용하여, **안정적인 통신을 확보**하고 중요한 정보가 더 많은 사람에게 전달되도록 돕는다.

### 2. 지속 가능성
불필요한 데이터 전송을 줄여 IT 자원의 효율적인 사용을 촉진하고, 이는 **경제적 지속 가능성**으로 이어진다.
* 영상 특성에 맞춰 데이터 전송량을 최적화하여 불필요한 트래픽을 줄인다. 이는 **스트리밍 서비스의 운영 비용을 절감**시켜 경제적 지속 가능성을 높인다.

### 3. 환경 보호

스트리밍으로 발생하는 **막대한 전력 소비를 줄여 'Green IT' 실현**에 기여한다.
* 전체 데이터 전송량을 줄이면 데이터를 처리하는 데이터 센터의 전력 소비가 감소한다. 이는 곧 **디지털 탄소 발자국을 저감**하는 효과로 이어져 환경 보호에 기여한다.

# 3. 시스템 설계
## 3.1. 시스템 구성도

<img width="600" height="393" alt="Image" src="https://github.com/user-attachments/assets/4ea5a4eb-2c2b-465e-a6f3-5af04ae77c6b" /><br>

* 구성도는 서버 측과 클라이언트 측으로 구분
* 기존 DASH에 **영상 인코딩 최적화 모듈인 하늘색 부분을 추가**해 영상 특성과 네트워크를 함께 고려하는 구조
* 서버는 **AWS EC2 인스턴스 상에 구축**되고, 클라이언트는 **사용자의 웹 브라우저**에서 동작
* 좌측의 영상은 사전에 서버에 업로드 된 콘텐츠
* 사용자가 특정 영상을 시청하길 원할 때, 서버는 해당 영상을 처리하여 스트리밍

## 3.2. 사용 기술

| 구분 | 기술 스택 | 주요 역할 |
| :--- | :--- | :--- |
| **Cloud & <br>Deployment** | **AWS EC2 (Ubuntu)** | 클라우드 가상 서버를 활용하여 애플리케이션의 안정적인 호스팅 환경 제공 |
| | **Docker, Docker Compose** | 서비스 환경을 컨테이너화하여 개발, 테스트, 배포 환경의 일관성 유지 및 관리 용이성 확보 |
| | **Nginx** | 고성능 웹 서버 및 리버스 프록시(Reverse Proxy)로 정적 파일 처리 및 백엔드 서버로의 요청 분배 담당 |
| | **DuckDNS** | 동적 IP 주소를 가진 EC2 인스턴스에 고정 도메인 이름을 부여하여 쉽게 접근할 수 있도록 지원 |
| **Backend** | **Python 3.11, FastAPI** | 현대적이고 빠른 고성능 웹 API 서버 구축 및 비즈니스 로직 처리 |
| | **Uvicorn** | FastAPI를 위한 초고속 ASGI(Asynchronous Server Gateway Interface) 서버 |
| **Frontend** | **React.js (Vite)** | 사용자 인터페이스(UI) 구축 및 동적 기능 구현 (Vite를 사용하여 빠른 개발 환경 구성) |
| | **dash.js** | MPEG-DASH 표준을 지원하는 비디오 플레이어로, 적응형 스트리밍 클라이언트 기능 수행 |
| | **Firebase** | **Authentication**: 소셜 로그인 등 간편하고 안전한 사용자 인증 기능 구현<br>**Firestore**: 실시간 데이터베이스를 활용한 사용자 데이터 및 영상 메타데이터 관리 |
| **Video Processing <br>& AI Pipeline** | **FFmpeg, MP4Box** | **FFmpeg**: 영상 인코딩/디코딩, 특성 추출 등 핵심 영상 처리<br>**MPBox**: 인코딩된 영상을 MPEG-DASH 형식에 맞게 세그먼트(Segment)로 분할 및 패키징 |
| | **NumPy, Pandas** | 추출된 영상 특성 데이터를 정제하고 머신러닝 모델 학습을 위한 데이터셋 구성 |
| | **Scikit-learn <br>(MLPRegressor)** | 영상 특성 기반 최적 인코딩 파라미터(CRF, Maxrate) 예측을 위한 **회귀 모델(MLPRegressor) 개발 및 학습** |

# 4. 개발 결과
## 4.1. 전체 시스템 흐름도

### 서버 측 시스템
<img width="600" height="436" alt="Image" src="https://github.com/user-attachments/assets/9e39e46a-f1b6-446f-843b-b52ccdba110f" /><br>

서버 측은 다음과 같은 과정을 통해 인코딩 및 영상 전송 과정을 거친다.
1.	**세그먼트 분할**
    * 원본 영상을 2초 단위로 재인코딩하여 세그먼트 생성
    * 각 세그먼트의 첫 프레임을 I-frame으로 강제해 경계 안정화

2.	**세그먼트의 영상 특성 추출**
    * 모션 벡터, 매크로블록 분포, 영상 메타 데이터, 공간적 복잡도 추출

3.	**영상 인코딩 최적화 모델**
    * 세그먼트의 영상 특성을 입력하여 최적화된 CRF, Max rate 예측
    * CRF는 유동적으로 비트레이트를 조절하여 영상의 일정한 품질을 유지하도록 하며, Max rate는 갑작스러운 비트 전송량의 증가에도 버퍼링이 발생하지 않도록 최대 비트 전송량 제한

4.	**최적화 인코딩 파라미터를 적용하여 세그먼트 인코딩**
    * 각 세그먼트를 최적화된 CRF, Max rate로 인코딩
    * 해상도별로 각 세그먼트 인코딩 – 360p, 480p, 720p, 1080p
    * 인코딩 된 파일은 .ts(MPEG-TS) 포맷으로 저장

5.	**MP4 파일 병합**
    * 해상도별 .ts 파일들을 이어 붙여 mp4 파일로 재포장

6.	**DASH 패키징**
    * MP4Box를 사용하여 manifest.mpd와 .m4s 생성
    * -rap 옵션으로 세그먼트가 I-frame 경계에서 시작되게 보장
    * Representation별 init segment를 MP4Box가 자동 생성

7.	**MPD 파일과 세그먼트가 서버에 저장**
    * MPD 파일은 세그먼트들의 정보를 담고 있으며, 클라이언트가 어떤 세그먼트를 요청할 지 판단하는 데 활용
    * 세그먼트는 미리 최적화 파라미터로 인코딩 되어 클라이언트의 요청에 따라 FastAPI를 통해 HTTP 서버로 제공

### 클라이언트 측 시스템
<img width="365" height="487" alt="Image" src="https://github.com/user-attachments/assets/a64365d9-f316-4a04-b569-d09c03996a17" /><br>


클라이언트 측은 **dash.js 라이브러리**를 기반으로 영상 스트리밍을 수행한다.
1. 서버에서 **MPD 파일** 다운로드
2. MPD 파일을 파싱하여, 현재 **사용 가능한 representation 목록**과 세그먼트 위치 등의 정보를 추출
3. dash.js 내부의 **ABR 알고리즘**이 네트워크 처리량과 버퍼 상태를 고려하여, **가장 적절한 품질의 세그먼트 선택**
4. 선택된 세그먼트는 URL을 통해 다운로드되며, **Media Source Extensions(MSE)** 를 통해 해당 세그먼트를 video buffer에 추가
5. 세그먼트는 H.264로 압축된 인코딩 영상이므로, 브라우저는 이를 자동으로 디코딩
6. 디코딩된 영상은 브라우저에서 재생

## 4.2. 기능 설명 및 주요 기능 명세서

### 1. 영상 특성 추출 및 최적 파라미터 예측 기능
* 기능 설명: 업로드된 영상 세그먼트의 **시각적 특성**을 정량적으로 분석하고, 이를 기반으로 AI 모델을 통해 **최적의 인코딩 파라미터(CRF, Max rate)를 도출**하는 핵심 기능
* 기능 명세:
  | 항목 | 내용 |
  | :--- | :--- |
  | 입력 (Input) | 원본 영상 세그먼트 파일 (2초 단위) |
  | 주요 처리 (Process) | 1. FFmpeg을 이용해 세그먼트의 모션 벡터, 매크로블록 분포, 공간적 복잡도 등 특성 데이터 추출 <br> 2. 추출된 특성 데이터를 AI 모델(MLPRegressor)의 입력값으로 전달 <br> 3. AI 모델이 학습된 가중치를 기반으로 최적의 CRF 및 Maxrate 값을 예측 |
  | 출력 (Output) | 예측된 최적 인코딩 파라미터 (예: { "crf": 23, "maxrate": "4M" }) |
    
### 2. 멀티-해상도 최적화 인코딩 기능
* 기능 설명: 예측된 최적 파라미터를 적용하여 각 영상 세그먼트를 **다양한 해상도의 스트리밍용 파일로 변환**하는 기능
* 기능 명세:
  | 항목 | 내용 |
  | :--- | :--- |
  | 입력 (Input) | 1. 원본 영상 세그먼트 파일 <br> 2. 예측된 최적 인코딩 파라미터 <br> 3. 목표 해상도 목록 (360p, 480p, 720p, 1080p) |
  | 주요 처리 (Process) | 1. 각 목표 해상도에 맞춰 영상 스케일링 <br> 2. 예측된 CRF, Maxrate 값을 인코딩 옵션으로 설정하여 FFmpeg으로 세그먼트 재인코딩 <br> 3. I-frame을 세그먼트 시작점에 강제하여 스트리밍 안정성 확보 |
  | 출력 (Output) | 해상도별로 최적화 인코딩된 영상 세그먼트 파일들 (.ts 포맷) |

### 3. MPEG-DASH 콘텐츠 패키징 기능
  * 기능 설명: 인코딩된 여러 해상도의 영상 세그먼트들을 MPEG-DASH 표준에 맞게 **스트리밍 가능한 최종 콘텐츠(mpd, m4s)로 생성**하는 기능
  * 기능 명세:
    | 항목 | 내용 |
    | :--- | :--- |
    | 입력 (Input) | 모든 해상도로 인코딩된 영상 세그먼트 파일들 |
    | 주요 처리 (Process) | 1. 해상도별 .ts 파일들을 하나의 .mp4 파일로 병합(Remuxing) <br> 2. MP4Box를 사용하여 병합된 파일을 DASH 포맷으로 패키징 <br> 3. I-frame 기준으로 세그먼트를 분할하고, 각 해상도(Representation)별 초기화 세그먼트 생성 |
    | 출력 (Output) | 1. 스트리밍 정보 파일 (manifest.mpd) <br> 2. 미디어 세그먼트 파일들 (.m4s) |

### 4. 적응형 스트리밍 재생 기능
* 기능 설명: 클라이언트(브라우저) 환경에서 사용자의 **네트워크 상태를 실시간으로 감지**하여 가장 적절한 화질의 영상을 끊김 없이 재생하는 기능
* 기능 명세: 
  | 항목 | 내용 |
  | :--- | :--- |
  | 입력 (Input) | 1. 서버에 저장된 manifest.mpd 파일 URL <br>  2. 사용자의 현재 네트워크 대역폭 및 버퍼 상태 |
  | 주요 처리 (Process) | 1. dash.js가 MPD 파일을 파싱하여 사용 가능한 화질 목록과 세그먼트 정보 확인 <br>  2. 내장된 ABR(Adaptive Bitrate) 알고리즘이 네트워크 상태를 분석하여 최적의 화질(Representation) 선택 <br>  3. 선택된 화질의 미디어 세그먼트를 서버에 요청 및 다운로드 <br>  4. Media Source Extensions(MSE) API를 통해 다운로드된 세그먼트를 비디오 버퍼에 추가 및 디코딩/재생 |
  | 출력 (Output) | 사용자의 시청 환경에 최적화된 비디오 스트림 재생 |

## 4.3. AI 모델 성능 및 학습 결과
### 1. 모델 성능 지표
CRF와 Max Rate를 예측하는 두 개의 MLPRegressor 모델에 대한 평가 지표는 다음과 같다.
| Metric | CRF Score | Max Rate Score |
| :--- | :--- | :--- |
| **MSE** | 3.42 | 179137.37 |
| **RMSE** | 1.85 | 423.24 |
| **MAE** | 1.42 | 300.13 |
| **R²** | 0.42 | 0.60 |

### 2. 주요 하이퍼 파라미터
하이퍼파라미터 튜닝을 통해 결정된 각 모델의 최적값은 다음과 같다.
| Hyperparameter | CRF Model | Max Rate Model | 설명 |
| :--- | :--- | :--- | :--- |
| **Solver** | Adam | Adam | 가중치 최적화 옵티마이저 |
| **Learning_rate** | Constant | Constant | 고정 학습률 |
| **Hidden_layer_sizes** | (80,) | (100,) | 은닉층 구조 |
| **Alpha** | 0.1 | 0.001 | L2 정규화 강도 |
| **Activation** | relu | relu | 활성화 함수 |

### 3. 모델 훈련 결과
CRF와 Max Rate 모델 모두 훈련 과정에서 손실(RMSE)가 안정적으로 수렴하는 것을 확인했다.

#### 손실 곡선 그래프 (RMSE)
<img width="413" height="267" alt="Image" src="https://github.com/user-attachments/assets/f2fc32aa-5099-42ff-9d6a-28775e69b0a7" />
<img width="413" height="265" alt="Image" src="https://github.com/user-attachments/assets/12faa623-d41e-4f2c-a02d-fa6ef214befe" />

#### Max Rate 산점도 그래프
<img width="413" height="412" alt="Image" src="https://github.com/user-attachments/assets/cf308a56-e674-4cc6-a8db-cae59d536767" />

## 4.4. 시스템 적용 결과 및 성능 비교
### 1. 인코딩 파라미터 예측 결과 출력
<img width="600" height="464" alt="Image" src="https://github.com/user-attachments/assets/ecf594ed-edab-43f0-b0a4-b9552a6f5867" />

### 2. 비트레이트 절감 효과
제안하는 최적화 모델을 기존 인코딩 방식과 비교했을 때, 평균 비트레이트 및 데이터 절감률은 다음과 같다.
| 인코딩 방식 | 평균 비트레이트 (kbps) | 절감률 (Savings) |
| :--- | :--- | :--- |
| 원본 | 1254.96 | N/A |
| CBR | 761.59 | 47.66% |
| CRF | 656.98 | 54.85% |
| **최적화 모델** | **584.16** | **59.85%** |

결과적으로, 최적화 모델은 원본 대비 약 60%의 비트레이트 절감률을, CRF 방식 대비 약 10%의 비트레이트 절감률을 보인다.
<br><br>
<img width="600" height="430" alt="Image" src="https://github.com/user-attachments/assets/a586a5e3-1fdb-47d9-8eec-bda180b0143d" />

위 그래프와 같이, 모델은 **저모션(Low-motion) 구간**에서는 원본보다는 비트레이트가 감소하나, CRF와 비교하면 화질 저하를 방지하기 위해 CRF와 비슷한 수준의 비트레이트를 유지하고 있다. 반면 **고모션(High-motion) 구간**에서는 원본보다 비트레이트가 감소했으며, 시각적 마스킹 효과로 인해 품질 저하가 눈에 띄지 않으므로 CRF보다 비트레이트 절감률이 뛰어난 것을 확인할 수 있다.

### 3. 시각적 품질 평가 (VMAF)
<img width="600" height="471" alt="Image" src="https://github.com/user-attachments/assets/a6115330-0f81-4008-abef-2c8c7b496624" />

최적화 모델을 적용한 영상의 VMAF 점수를 측정한 결과, **평균 89.5점**으로 목표 기준치(90점)에 근접한 높은 품질을 유지했다. 그러나 일부 구간에서는 Max Rate 제약으로 인한 순간적인 품질 저하가 관찰되었으나, 전반적으로 우수한 품질을 보였다.

## 4.5. 디렉토리 구조
>

### 5. 설치 및 실행 방법
>
#### 5.1. 설치절차 및 실행 방법
> 설치 명령어 및 준비 사항, 실행 명령어, 포트 정보 등
#### 5.2. 오류 발생 시 해결 방법
> 선택 사항, 자주 발생하는 오류 및 해결책 등

### 6. 소개 자료 및 시연 영상
#### 6.1. 프로젝트 소개 자료
> PPT 등
#### 6.2. 시연 영상
> 영상 링크 또는 주요 장면 설명

### 7. 팀 구성
#### 7.1. 팀원별 소개 및 역할 분담
>
#### 7.2. 팀원 별 참여 후기
> 개별적으로 느낀 점, 협업, 기술적 어려움 극복 사례 등

### 8. 참고 문헌 및 출처

```

## 5. README.md 작성팁 
* 마크다운 언어를 이용해 README.md 파일을 작성할 때 참고할 수 있는 마크다운 언어 문법을 공유합니다.  
* 다양한 예제와 보다 자세한 문법은 [이 문서](https://www.markdownguide.org/basic-syntax/)를 참고하세요.

### 5.1. 헤더 Header
```
# This is a Header 1
## This is a Header 2
### This is a Header 3
#### This is a Header 4
##### This is a Header 5
###### This is a Header 6
####### This is a Header 7 은 지원되지 않습니다.
```
<br />

### 5.2. 인용문 BlockQuote
```
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
```
> This is a first blockqute.
>	> This is a second blockqute.
>	>	> This is a third blockqute.
<br />

### 5.3. 목록 List
* **Ordered List**
```
1. first
2. second
3. third  
```
1. first
2. second
3. third
<br />

* **Unordered List**
```
* 하나
  * 둘

+ 하나
  + 둘

- 하나
  - 둘
```
* 하나
  * 둘

+ 하나
  + 둘

- 하나
  - 둘
<br />

### 5.4. 코드 CodeBlock
* 코드 블럭 이용 '``'
```
여러줄 주석 "```" 이용
"```
#include <stdio.h>
int main(void){
  printf("Hello world!");
  return 0;
}
```"

단어 주석 "`" 이용
"`Hello world`"

* 큰 따움표(") 없이 사용하세요.
``` 
<br />

### 5.5. 링크 Link
```
[Title](link)
[부산대학교 정보컴퓨터공학부](https://cse.pusan.ac.kr/cse/index..do)

<link>
<https://cse.pusan.ac.kr/cse/index..do>
``` 
[부산대학교 정보컴퓨터공학부](https://cse.pusan.ac.kr/cse/index..do)

<https://cse.pusan.ac.kr/cse/index..do>
<br />

### 5.6. 강조 Highlighting
```
*single asterisks*
_single underscores_
**double asterisks**
__double underscores__
~~cancelline~~
```
*single asterisks* <br />
_single underscores_ <br />
**double asterisks** <br />
__double underscores__ <br />
~~cancelline~~  <br />
<br />

### 5.7. 이미지 Image
```
<img src="image URL" width="600px" title="Title" alt="Alt text"></img>
![Alt text](image URL "Optional title")
```
- 웹에서 작성한다면 README.md 내용 안으로 이미지를 드래그 앤 드롭하면 이미지가 생성됩니다.
- 웹이 아닌 로컬에서 작성한다면, github issue에 이미지를 드래그 앤 드롭하여 image url 을 얻을 수 있습니다. (URL만 복사하고 issue는 제출 안 함.)
  <img src="https://github.com/user-attachments/assets/0fe3bff1-7a2b-4df3-b230-cac4ef5f6d0b" alt="이슈에 image 올림" width="600" />
  <img src="https://github.com/user-attachments/assets/251c6d42-b36b-4ad4-9cfa-fa2cc67a9a50" alt="image url 복사" width="600" />


### 5.8. 유튜브 영상 추가
```markdown
[![영상 이름](유튜브 영상 썸네일 URL)](유튜브 영상 URL)
[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://www.youtube.com/watch?v=zh_gQ_lmLqE)    
```
[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://www.youtube.com/watch?v=zh_gQ_lmLqE)    

- 이때 유튜브 영상 썸네일 URL은 유투브 영상 URL로부터 다음과 같이 얻을 수 있습니다.

- `Youtube URL`: https://www.youtube.com/watch?v={동영상 ID}
- `Youtube Thumbnail URL`: http://img.youtube.com/vi/{동영상 ID}/0.jpg 
- 예를 들어, https://www.youtube.com/watch?v=zh_gQ_lmLqE 라고 하면 썸네일의 주소는 http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg 이다.

