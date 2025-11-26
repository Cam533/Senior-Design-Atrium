# How to Run Frontend and Backend

## Prerequisites
- Python virtual environment activated (for backend)
- Node.js and npm installed (for frontend)
- `.env` file configured with database credentials and API keys

## Running the Backend (FastAPI)

1. **Activate your virtual environment** (if not already activated):
   ```bash
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   
   # On Windows CMD:
   venv\Scripts\activate.bat
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

2. **Navigate to the project root** (if not already there):
   ```bash
   cd "Path\To\SENIOR-DESIGN-ATRIUM\backend"
   ```

3. **Start the FastAPI server**:
   ```bash
   fastapi dev main.py
   ```

   The backend will be available at: `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Root endpoint: `http://localhost:8000/`

## Running the Frontend (React + Vite)

1. **Open a new terminal** (keep the backend running in the first terminal)

2. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

3. **Install dependencies** (if you haven't already):
   ```bash
   npm install
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```

   The frontend will typically be available at: `http://localhost:4000`
   (Vite will show you the exact URL in the terminal)

