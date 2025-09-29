import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";

//은재 정보

const firebaseConfig = {
  apiKey: "AIzaSyCqNwC1ytNe5F4s0tbmXAg9Vf9IpMgbc3A",
  authDomain: "netflix-e3fed.firebaseapp.com",
  projectId: "netflix-e3fed",
  storageBucket: "netflix-e3fed.firebasestorage.app",
  messagingSenderId: "408206225379",
  appId: "1:408206225379:web:6fbf0d110b3f2c291afad6"
};

/*
const firebaseConfig = {
  apiKey: "AIzaSyBIhBiO3flFpAcL2Fm_Ef22QQo6udFp5b4",
  authDomain: "react-netflix-eb4f0.firebaseapp.com",
  projectId: "react-netflix-eb4f0",
  storageBucket: "react-netflix-eb4f0.appspot.com",
  messagingSenderId: "29045190704",
  appId: "1:29045190704:web:a7c74bd778aa5f993c7df5",
  measurementId: "G-9TB7LL3YPM",
};
*/
// Initialize Firebase
export const FirebaseApp = initializeApp(firebaseConfig);
export const db = getFirestore(FirebaseApp);
const analytics = getAnalytics(FirebaseApp);