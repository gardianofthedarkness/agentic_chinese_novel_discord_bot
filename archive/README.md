# Archive - Old/Unused Files

This directory contains old versions and testing files that are no longer used in the main workflow.

## Archived Files

- **`quick_unified_run.py`** - SQLite fallback version (replaced by final unified)
- **`run_unified_clean.py`** - Separate cleaning version (integrated into final)  
- **`debug_volumes.py`** - Volume detection debugging script (functionality integrated)

## Current Active Files

**Use only these files:**
- `processors/run_unified_final.py` - **MAIN ENTRY POINT**
- `processors/unified_novel_processor.py` - Core processing logic
- `processors/migrate_volume_1.py` - Volume 1 migration utilities

## Command

```bash
cd processors
python run_unified_final.py
```

Do not use files in this archive directory - they are kept for reference only.