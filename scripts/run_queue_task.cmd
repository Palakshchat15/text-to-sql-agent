@echo off
rem Launcher for the Text-to-SQL evaluation queue, for Windows Task Scheduler (task "T2SQL_eval_queue")
rem or a double-click, so it runs independently of any editor. Resumes from the per-question caches.
title Text-to-SQL evaluation queue - do not close
cd /d "%~dp0.."
if not exist "results\logs" mkdir "results\logs"
".venv\Scripts\python.exe" "scripts\run_eval_queue.py" %* >> "results\logs\queue_task.out" 2>&1
echo finished %date% %time% >> "results\logs\queue_task.out"
