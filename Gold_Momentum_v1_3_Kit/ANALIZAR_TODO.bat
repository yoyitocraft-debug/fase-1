@echo off
setlocal
cd /d "%~dp0"
python -m pip install --upgrade pandas
if errorlevel 1 goto error
echo.
echo Analizando las carpetas y ZIPs de resultados...
python analizador_maestro.py .
if errorlevel 1 (
  echo.
  echo No se pudo completar. Revisa el mensaje anterior.
  pause
  exit /b 1
)
echo.
echo Abre resultado_maestro\INFORME_MAESTRO.html
start "" "%CD%\resultado_maestro\INFORME_MAESTRO.html"
pause
exit /b 0
:error
echo No se pudo instalar pandas. Comprueba Python y tu conexion.
pause
exit /b 1
