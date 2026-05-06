@echo off
echo ==================================================
echo Welcome to the FP-Growth Project!
echo ==================================================
echo.
echo Step 1: Installing required libraries...
pip install -r requirements.txt
echo.
echo Step 2: Running the Application...
streamlit run app.py
pause
