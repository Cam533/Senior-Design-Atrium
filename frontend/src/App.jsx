import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useLocation,
  Navigate,
} from "react-router-dom";
import Map from "../components/Map";
import LikedLots from "../components/LikedLots";
import Projects from "../components/Projects";
import Login from "../components/Auth/Login";
import ProfileSetup from "../components/Auth/ProfileSetup";
import Profile from "../components/Auth/Profile";
import atriumIcon from "../pics/atrium_icon.png";
import { useAuth } from "./context/AuthContext";

function Topbar() {
  const location = useLocation();
  const { user, logout } = useAuth();

  return (
    <div className="topbar">
      <div className="brand">
        <img src={atriumIcon} alt="Atrium icon" className="brand-icon" />
        <div className="brand-text">atrium</div>
      </div>
      <nav className="topbar-nav">
        <Link
          to="/"
          className={
            location.pathname === "/"
              ? "nav-link nav-button active"
              : "nav-link nav-button"
          }
        >
          Map
        </Link>
        <Link
          to="/chat"
          className={
            location.pathname === "/chat"
              ? "nav-link nav-button active"
              : "nav-link nav-button"
          }
        >
          Liked Lots
        </Link>
        <Link
          to="/projects"
          className={
            location.pathname === "/projects"
              ? "nav-link nav-button active"
              : "nav-link nav-button"
          }
        >
          Projects
        </Link>
        {!user ? (
          <>
            <Link
              to="/login"
              className={
                location.pathname === "/login"
                  ? "nav-link nav-button active"
                  : "nav-link nav-button"
              }
            >
              Login
            </Link>
            <Link
              to="/signup"
              className={
                location.pathname === "/signup"
                  ? "nav-link nav-button active"
                  : "nav-link nav-button"
              }
            >
              Sign Up
            </Link>
          </>
        ) : (
          <Link
            to="/profile"
            className={
              location.pathname === "/profile"
                ? "nav-link nav-button active"
                : "nav-link nav-button"
            }
          >
            Profile
          </Link>
        )}
      </nav>
    </div>
  );
}

function MapPage() {
  return (
    <main className="main-content">
      <div className="map-container">
        <Map />
      </div>
    </main>
  );
}

function ChatPage() {
  return (
    <main className="main-content">
      <div className="projects-container">
        <LikedLots />
      </div>
    </main>
  );
}

function ProjectsPage() {
  return (
    <main className="main-content">
      <div className="projects-container">
        <Projects />
      </div>
    </main>
  );
}
export default function App() {
  return (
    <BrowserRouter>
      <div className="app-root">
        <Topbar />
        <Routes>
          <Route path="/" element={<MapPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/signup"
            element={<Navigate to="/profile-setup" replace />}
          />
          <Route path="/profile-setup" element={<ProfileSetup />} />
          <Route path="/profile" element={<Profile />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
