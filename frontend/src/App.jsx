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
import About from "../components/About";
import NotificationBell, { useUnreadCount } from "../components/Notifications";
import Login from "../components/Auth/Login";
import ProfileSetup from "../components/Auth/ProfileSetup";
import Profile from "../components/Auth/Profile";
import atriumIcon from "../pics/atrium_icon.png";
import { useAuth } from "./context/AuthContext";

function Topbar({ unreadCount, onRead }) {
  const location = useLocation();
  const { user } = useAuth();

  return (
    <div className="topbar">
      <Link to="/about" className="brand" style={{ textDecoration: 'none', color: 'white' }}>
        <img src={atriumIcon} alt="Atrium icon" className="brand-icon" />
        <div className="brand-text">atrium</div>
      </Link>
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
          to="/about"
          className={
            location.pathname === "/about"
              ? "nav-link nav-button active"
              : "nav-link nav-button"
          }
        >
          About
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
        {user && (
          <NotificationBell unreadCount={unreadCount} onRead={onRead} />
        )}
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

function AboutPage() {
  return (
    <main style={{ flex: '1 1 auto', overflowY: 'auto' }}>
      <About />
    </main>
  );
}

function AppShell() {
  const { count, refresh } = useUnreadCount();

  return (
    <div className="app-root">
      <Topbar unreadCount={count} onRead={refresh} />
      <Routes>
        <Route path="/" element={<MapPage />} />
        <Route path="/about" element={<AboutPage />} />
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
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
