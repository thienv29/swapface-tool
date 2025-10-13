@echo off
echo Starting SwapFace GTX 1650 Container...
echo Make sure NVIDIA Docker is installed and GPU drivers are up to date.
echo.
docker run --gpus all --name swapface-gtx1650-app -p 8000:8000 -v "%CD%/output:/app/output" -v "%CD%/models:/app/models" -v "%CD%/static:/app/static" -v "%CD%/templates:/app/templates" -e CUDA_VISIBLE_DEVICES=0 swapface-gtx1650
echo.
pause
