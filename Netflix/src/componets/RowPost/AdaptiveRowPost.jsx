import React from "react";
import { useNavigate } from "react-router-dom";
import { Swiper, SwiperSlide } from "swiper/react";
import { Navigation } from "swiper";
import "swiper/css";
import "swiper/css/navigation";
import "../../componets/RowPost/RowPostStyles.scss";

function AdaptiveRowPost({ title, videos }) {
  const nav = useNavigate();

  return (
    <div className="ml-2 lg:ml-11 mb-32 RowContainer">
      {/* 제목 */}
      <div className="flex justify-between items-center pb-4 xl:pb-0">
        <h1 className="text-white font-normal text-base sm:text-2xl md:text-4xl">
          {title}
        </h1>
      </div>

      <Swiper
        breakpoints={{
          1800: { slidesPerView: 6.1, slidesPerGroup: 5 },
          1536: { slidesPerView: 5, slidesPerGroup: 5 },
          1280: { slidesPerView: 4.3, slidesPerGroup: 4 },
          768: { slidesPerView: 3.3, slidesPerGroup: 3 },
          330: { slidesPerView: 2.1, slidesPerGroup: 2 },
          0: { slidesPerView: 2, slidesPerGroup: 2 },
        }}
        modules={[Navigation]}
        spaceBetween={8}
        navigation
        className="SwiperStyle"
      >
        {videos.map((video) => (
          <SwiperSlide key={video.id} className="bg-cover">
            <div
              className="group relative cursor-pointer"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const route = `/adaptive-play/${video.id}`;
                console.log("🖱️ 썸네일 클릭됨:", video.title);
                console.log("➡️ 이동 경로:", route);
                console.log("🎯 manifest URL:", video.manifest);
                nav(route);
              }}
            >
              <img
                src={video.thumb || "/assets/default.jpg"}
                alt={video.title}
                className="rounded-sm w-full h-full"
                onError={(e) => {
                  e.target.src = "/assets/default.jpg";
                }}
              />
              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition duration-200 bg-black bg-opacity-40 flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="white"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-10 h-10 text-white"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z"
                  />
                </svg>
              </div>
            </div>
          </SwiperSlide>
        ))}
      </Swiper>
    </div>
  );
}

export default AdaptiveRowPost;
