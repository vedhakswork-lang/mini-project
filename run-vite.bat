@echo off
set "ROOT=C:\Users\akash\OneDrive\Desktop\MINI PROJECT"
set "NODE=%ROOT%\node\node-v24.20.0-win-x64"
set "PATH=%NODE%;%PATH%"
cd /d "%ROOT%"
npx.cmd create-vite@latest frontend --template react-ts --no-interactive
