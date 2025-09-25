@echo off
REM Fix encoding issues for Chinese characters
chcp 65001
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8
echo Encoding set to UTF-8 for Chinese character support
echo PYTHONIOENCODING=%PYTHONIOENCODING%
echo LANG=%LANG%
echo.
echo Now run your Python scripts normally
echo Example: python run_hierarchical_analysis.py
cmd