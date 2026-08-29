@echo off
REM set battery screen time
powercfg /setacvalueindex scheme_current sub_video videoidle 600
powercfg /setactive scheme_current
echo The screen will shutdown after 10min.
rem pause