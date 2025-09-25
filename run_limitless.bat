@echo off
echo ================================================================================
echo 🚀 STARTING LIMITLESS 5-VOLUME PROCESSING
echo ================================================================================
echo This will process all 5 volumes with complete character/storyline/timeline analysis
echo Results will be saved to database: limitless_processing_results.db
echo Log file: limitless_processing.log
echo ================================================================================
echo.
echo Press Ctrl+C to stop processing at any time
echo.
pause

REM Fix encoding for Chinese characters
chcp 65001
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8

REM Run the limitless processor
python limitless_5_volume_processor.py

echo.
echo ================================================================================
echo 🏁 PROCESSING COMPLETED
echo ================================================================================
echo Check the following files for results:
echo - limitless_processing_results.db (SQLite database with all analysis)
echo - limitless_processing.log (detailed processing log)
echo - limitless_5_volume_report_*.json (final summary report)
echo ================================================================================
pause