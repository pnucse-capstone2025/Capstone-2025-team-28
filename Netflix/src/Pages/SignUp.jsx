import React from "react";
import { useState, useContext } from "react";

import { Link, useNavigate } from "react-router-dom";
import { Fade } from "react-awesome-reveal";
import {
  getAuth,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
} from "firebase/auth";
import { setDoc, doc } from "firebase/firestore";
import { db } from "../Firebase/FirebaseConfig";
import { AuthContext } from "../Context/UserContext";
import { ClipLoader } from "react-spinners";
import WelcomePageBanner from "../images/WelcomePageBanner.jpg";

function SignUp() {
  const { User, setUser } = useContext(AuthContext);
  //사용자가 입력한 이메일과 비밀번호를 저장할 상태 변수
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [ErrorMessage, setErrorMessage] = useState("");
  const [loader, setLoader] = useState(false);

  const navigate = useNavigate();

  //회원가입 버튼을 클릭했을 때 실행되는 함수
  const handleSubmit = (e) => {
    e.preventDefault(); //폼 기본 제출 막기
    setLoader(true); //로딩 상태 활성화

    // Firebase Authentication을 사용하여 회원가입 처리
    const auth = getAuth();
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => { //성공하면 userCredential 객체 반환
        // Signed in
        //회원가입 이후, 사용자가 실제 로그인된 상태인지 확인하기 위해 onAuthStateChanged 사용
        onAuthStateChanged(auth, (user) => {
          const EmptyArray = [];
          // Firestore에 사용자의 정보(email, uid)를 저장
          setDoc(doc(db, "Users", user.uid), {
            email: email,
            Uid: user.uid,
          }).then(() => {
            // MyList, WatchedMovies, LikedMovies 컬렉션에 빈 배열로 초기화
            // 사용자 전용 영화 목록을 초기화
            // 각 문서에는 movies: [] 형식으로 빈 배열이 들어가며,
            // 나중에 사용자가 영화를 추가하거나 삭제할 수 있도록 설정
            setDoc(
              doc(db, "MyList", user.uid),
              {
                movies: EmptyArray,
              },
              { merge: true }
            ).then(() => {
              setDoc(
                doc(db, "WatchedMovies", user.uid),
                {
                  movies: EmptyArray,
                },
                { merge: true }
              );
              setDoc(
                doc(db, "LikedMovies", user.uid),
                {
                  movies: EmptyArray,
                },
                { merge: true }
              );
            });
          });
        });

        const user = userCredential.user;
        if (user != null) {
          navigate("/"); //회원가입 성공 후 홈으로 이동
        }
      })

      //createUserWithEmailAndPassword() 실패 시 에러 처리
      //에러 예시 : 이미 가입된 이메일, 비밀번호 6자 미만,
      //이메일 형식 아님, 이메일 입력 안함
      .catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;
        setLoader(false);
        setErrorMessage(errorMessage);
        console.log(errorCode);
        console.log(errorMessage);
      });
  };

  return (
    <section
      className="h-[100vh] bg-gray-500"
      style={{
        background: `linear-gradient(0deg, hsl(0deg 0% 0% / 73%) 0%, hsl(0deg 0% 0% / 73%) 35%),url(${WelcomePageBanner})`,
      }}
    >
      <div className="h-[100vh] flex flex-col items-center justify-center px-6 py-8 mx-auto md:h-screen lg:py-0">
        <div className="w-full bg-[#000000a2] rounded-lg shadow sm:my-0 md:mt-0 sm:max-w-lg xl:p-0 border-2 border-stone-800 lg:border-0">
          <Fade>
            <div>
              <div className="p-6 space-y-4 md:space-y-6 sm:p-12">
                <h1 className="text-xl font-bold leading-tight tracking-tight text-white md:text-2xl dark:text-white">
                  Create a new account
                </h1>
                <h1 className="text-white text-2xl p-3 text-center border-2 border-blue-700 rounded-sm">
                  PNUPLAY
                </h1>
                <form
                  onSubmit={handleSubmit}
                  className="space-y-4 md:space-y-6"
                  action="#"
                >
                  <div>
                    <label
                      for="email"
                      className="block mb-2 text-sm font-medium text-white dark:text-white"
                    >
                      Your email
                    </label>
                    <input
                      onChange={(e) => setEmail(e.target.value)}
                      type="email"
                      name="email"
                      id="email"
                      className={
                        ErrorMessage
                          ? "bg-stone-700 text-white sm:text-sm rounded-sm border-2 border-blue-700 focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:text-white "
                          : "bg-stone-700 text-white sm:text-sm rounded-sm focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:text-white "
                      }
                      placeholder="name@emil.com"
                      required=""
                    ></input>
                  </div>
                  <div>
                    <label
                      for="password"
                      className="block mb-2 text-sm font-medium text-white dark:text-white"
                    >
                      Password
                    </label>
                    <input
                      onChange={(e) => setPassword(e.target.value)}
                      type="password"
                      name="password"
                      id="password"
                      placeholder="••••••••"
                      className={
                        ErrorMessage
                          ? "bg-stone-700 text-white sm:text-sm rounded-sm border-2 border-blue-700 focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5"
                          : "bg-stone-700 text-white sm:text-sm rounded-sm focus:ring-primary-600 focus:border-primary-600 block w-full p-2.5 dark:text-white"
                      }
                      required=""
                    ></input>
                  </div>
                  <div>
                    {ErrorMessage && (
                      <h1 className="flex text-white font-bold p-4 bg-blue-700 rounded text-center">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={1.5}
                          stroke="currentColor"
                          className="w-6 h-6 mr-1"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
                          />
                        </svg>

                        {ErrorMessage}
                      </h1>
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-start">
                      <div className="flex items-center h-5">
                        <input
                          id="remember"
                          aria-describedby="remember"
                          type="checkbox"
                          className="w-4 h-4 border border-gray-300 rounded bg-gray-50 focus:ring-3 focus:ring-primary-300 "
                          required=""
                        ></input>
                      </div>
                      <div className="ml-3 text-sm">
                        <label for="remember" className="text-gray-500">
                          Remember me
                        </label>
                      </div>
                    </div>
                  </div>
                  <button
                    type="submit"
                    className={`w-full text-white ${
                      loader
                        ? `bg-stone-700`
                        : `bg-blue-800 focus:ring-4 focus:outline-none focus:ring-primary-300`
                    } font-medium rounded-sm text-sm px-5 py-2.5 text-center`}
                  >
                    {loader ? <ClipLoader color="#ff0000" /> : "Create now"}
                  </button>
                  <p className="text-sm font-light text-gray-500">
                    Already have one?{" "}
                    <Link
                      className="font-medium text-white hover:underline"
                      to={"/signin"}
                    >
                      Sign in
                    </Link>
                  </p>
                </form>
              </div>
            </div>
          </Fade>
        </div>
      </div>
    </section>
  );
}

export default SignUp;
