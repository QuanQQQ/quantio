@echo off
echo Starting A-share Data System...
echo 1. Updating Data (Optional - press Ctrl+C to skip if already updated)
python -c "from src.fetcher import update_all; update_all(limit=20)"
echo.
echo 2. Launching Dashboard...
streamlit run src/app.py
pause
