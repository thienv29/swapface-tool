@echo off
echo [SwapFace GTX 1650] Starting Container...
echo Make sure NVIDIA Docker is installed and GPU drivers are up to date.
echo If container already exists, it will be removed first.
echo.
echo Checking for existing container...
docker ps -a --filter name=swapface-gtx1650-app --format "{{.Names}}" | findstr swapface-gtx1650-app >nul
if %errorlevel% equ 0 (
    echo Removing existing container...
    docker rm -f swapface-gtx1650-app
)
echo.
echo Starting new container...
docker run -d --gpus all --name swapface-gtx1650-app -p 8000:8000 ^
    -v "%CD%/output:/app/output" ^
    -v "%CD%/models:/app/models" ^
    -v "%CD%/static:/app/static" ^
    -v "%CD%/templates:/app/templates" ^
    -e CUDA_VISIBLE_DEVICES=0 ^
    swapface-gtx1650
echo.
if %errorlevel% equ 0 (
    echo [SUCCESS] Container started successfully!
    echo.
    echo Web interface: http://localhost:8000
    echo.
    echo To view logs: docker logs -f swapface-gtx1650-app
    echo To stop: docker stop swapface-gtx1650-app
) else (
    echo [ERROR] Failed to start container! Check Docker and GPU setup.
)
echo.
pause
