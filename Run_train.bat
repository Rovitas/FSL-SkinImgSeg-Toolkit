@echo off
rem call "D:\ProgramData\Miniconda\Scripts\activate.bat" 
cd /d "%~dp0code\train"
python train.py
if exist "%~dp0no_shutdown" (
    echo ## File 'no_shutdown' exists, skip shutdown
) else (
    echo ## Training is over, your PC will shutdown in 2 mins.
    shutdown -s -t 120
)
pause