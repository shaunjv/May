@echo off
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Install uv first, then run this launcher again.
  pause
  exit /b 1
)
if not exist "web\dist\index.html" (
  pushd web
  call npm ci
  if errorlevel 1 goto :failed
  call npm run build
  if errorlevel 1 goto :failed
  popd
)
uv run --extra web agent-web
if errorlevel 1 pause
exit /b
:failed
echo Frontend setup failed. Check the error above.
pause
exit /b 1
