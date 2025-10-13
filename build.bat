@echo off
echo [SwapFace GTX 1650] Building Docker Image...
echo Make sure Docker Desktop is running and NVIDIA Container Toolkit is installed.
echo.
docker build -t swapface-gtx1650 .
echo.
if %errorlevel% equ 0 (
    echo [SUCCESS] Build complete! Use run.bat to start the container.
) else (
    echo [ERROR] Build failed! Check Docker installation and GPU drivers.
    pause
    exit /b 1
)
echo.
pause
