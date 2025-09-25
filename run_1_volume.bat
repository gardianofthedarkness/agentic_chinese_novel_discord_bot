@echo off
echo ================================================================================
echo 🚀 STARTING LIMITLESS VOLUME 1 PROCESSING - TEST VERSION
echo ================================================================================
echo This will process Volume 1 completely with detailed progress updates
echo Results will be saved to database: limitless_1_volume_results.db
echo Log file: limitless_1_volume.log
echo ================================================================================
echo.
echo You will see detailed progress for each chunk:
echo - Progress percentage and ETA
echo - Tokens used per chunk
echo - Processing time per chunk  
echo - Characters and events found
echo - Running totals
echo.
echo Press Ctrl+C to stop processing at any time
echo.
pause

REM Fix encoding for Chinese characters
chcp 65001
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8

REM Run the limitless 1-volume processor
python limitless_1_volume_processor.py

echo.
echo ================================================================================
echo 🏁 VOLUME 1 PROCESSING COMPLETED
echo ================================================================================
echo Check the following files for results:
echo - limitless_1_volume_results.db (SQLite database with all analysis)
echo - limitless_1_volume.log (detailed processing log)
echo - limitless_volume_1_report_*.json (final summary report)
echo ================================================================================
pause