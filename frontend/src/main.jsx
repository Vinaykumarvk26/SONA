import React from "react";
import ReactDOM from "react-dom/client";
import ReactGA from "react-ga4";
import App from "./App";
import "./index.css";

// Initialize Google Analytics
ReactGA.initialize("G-SPV55P81ZL");

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
