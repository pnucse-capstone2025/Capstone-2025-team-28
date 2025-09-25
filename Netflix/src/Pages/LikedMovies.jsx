import React from "react";
import UserMovieSection from "../componets/UserMovieSection/UserMovieSection";

//props.from 값에 따라 Mylist, WatchedMovies, LikedMovies 중 하나의 리스트를 보여준다.
function LikedMovies() {
  return <UserMovieSection from="LikedMovies"></UserMovieSection>;
}

export default LikedMovies;
