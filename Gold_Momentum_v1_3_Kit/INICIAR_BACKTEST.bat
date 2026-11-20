@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creando entorno de Python...
  python -m venv .venv
  if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto error
python -m pip install yfinance pandas numpy
if errorlevel 1 goto error
echo.
echo Ejecutando Gold Momentum v1.3 en modo historico/paper-signal...
echo.
python gold_momentum_v13.py --refresh
if errorlevel 1 goto error
echo.
echo Listo. Revisa la carpeta gold_reports.
pause
exit /b 0
:error
echo.
echo Hubo un error. Copia el mensaje y mandamelo para corregirlo.
pause
exit /b 1
