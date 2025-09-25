@echo off
echo ================================================================================
echo 🚀 ENHANCED VOLUME 1 BATCH PROCESSOR
echo ================================================================================
echo This script processes Volume 1 with detailed batch-by-batch reporting:
echo.
echo 📊 What you'll see for each batch:
echo    - Number of retrospections DeepSeek makes before satisfaction
echo    - Decision types: continue, adjust_database, function_call, satisfied
echo    - Database changes and function calls made by DeepSeek
echo    - Satisfaction progression through iterations
echo    - Processing time and token usage
echo.
echo 💾 Enhanced database tracking:
echo    - batch_progress: Overall batch statistics
echo    - batch_decisions: Every decision DeepSeek makes
echo    - database_changes: All database modifications
echo    - enhanced_analysis: Detailed analysis results
echo.
echo 🎯 Key metrics tracked:
echo    - How many times DeepSeek retrospects per batch
echo    - Types of decisions and confidence levels
echo    - Database operations and function calls
echo    - Satisfaction improvement over iterations
echo.
echo Press any key to start enhanced processing...
pause

REM Fix encoding for Chinese characters
chcp 65001
set PYTHONIOENCODING=utf-8
set LANG=en_US.UTF-8

REM Run the enhanced volume 1 processor
python volume_1_batch_processor.py

echo.
echo ================================================================================
echo 🏁 ENHANCED PROCESSING COMPLETED
echo ================================================================================
echo Check the following files for detailed results:
echo - volume_1_enhanced_batch.db (Enhanced SQLite database)
echo - volume_1_enhanced_report_*.json (Comprehensive report)
echo - volume_1_batch_processing.log (Detailed processing log)
echo.
echo Database contains:
echo - Batch progress and decision tracking
echo - Every DeepSeek decision with reasoning
echo - Database changes and function calls
echo - Satisfaction progression analysis
echo ================================================================================
pause