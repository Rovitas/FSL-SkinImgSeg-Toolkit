@echo off
REM set battery screen time
powercfg /setacvalueindex scheme_current sub_video videoidle 60
powercfg /setactive scheme_current
echo The screen will shutdown after 1min.
rem pause