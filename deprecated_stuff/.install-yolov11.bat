@REM python -m venv .venv-yolov11
@REM .\.venv-yolov11\Scripts\activate.bat
call pip uninstall ultralytics roboflow -y
call pip install ultralytics roboflow
@REM call pip install inference supervision