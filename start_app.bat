@echo off
echo Starting Virtual Try-On Application...

echo.
echo Starting Backend Server...
cd backend
start "Backend" cmd /c "C:\Users\yunnn\Desktop\cv\venv\Scripts\python.exe main.py"

echo.
echo Waiting for backend to start...
timeout /t 5

echo.
echo Starting Frontend Server...
cd ..\frontend
start "Frontend" cmd /c "npm install && npm start"

echo.
echo Application is starting...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.

pause