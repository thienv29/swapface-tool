@echo off
echo Building SwapFace GTX 1650 Docker Image...
docker build -t swapface-gtx1650 .
echo.
echo Build complete! Use run.bat to start the container.
