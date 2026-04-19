@echo off
echo ==========================================
echo  Intent-Aware Hybrid Retrieval System
echo ==========================================
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Copying data file...
if not exist data mkdir data
if exist ..\profiles.csv copy ..\profiles.csv data\profiles.csv
if exist profiles.csv copy profiles.csv data\profiles.csv
echo.
echo Starting server at http://localhost:8000 ...
python main.py
pause
