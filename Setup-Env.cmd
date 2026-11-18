@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "NO_PAUSE=0"
set "USE_MIRROR=0"
set "WITH_BUILD=0"
for %%A in (%*) do (
    if /I "%%~A"=="--no-pause" set "NO_PAUSE=1"
    if /I "%%~A"=="--mirror" set "USE_MIRROR=1"
    if /I "%%~A"=="--build" set "WITH_BUILD=1"
)

echo ==================================================
echo   scsp_mmd_tools 动作转换环境一键安装 (Python)
echo ==================================================
echo.

REM 1. 查找 Python (>= 3.10 且包含 venv 与 tkinter)
set "PY_CMD="
for %%C in ("py -3" "python") do (
    if not defined PY_CMD (
        %%~C -c "import sys, tkinter, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if !errorlevel! equ 0 set "PY_CMD=%%~C"
    )
)

if not defined PY_CMD (
    echo [提示] 未检测到可用的 Python ^>= 3.10（需自带 tkinter 与 venv）。
    where winget >nul 2>nul
    if !errorlevel! equ 0 (
        echo [1/3] 尝试通过 winget 安装 Python 3.13 ...
        winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
        if !errorlevel! equ 0 (
            for %%P in ("%LocalAppData%\Programs\Python\Python313\python.exe" "C:\Python313\python.exe") do (
                if exist "%%~P" (
                    "%%~P" -c "import sys, tkinter, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
                    if !errorlevel! equ 0 set "PY_CMD="%%~P""
                )
            )
        )
    )
)

if not defined PY_CMD (
    echo [错误] 仍未找到可用的 Python ^>= 3.10（含 tkinter）。
    echo        请先从 https://www.python.org/downloads/ 安装 Python 3.13（勾选 tcl/tk 与 PATH），然后重试。
    goto :fail
)

echo [1/3] 使用 Python: %PY_CMD%

REM 2. 创建虚拟环境 .venv 并安装动作依赖 (numpy, UnityPy)
if not exist ".venv\Scripts\python.exe" (
    echo [2/3] 正在创建虚拟环境 .venv ...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [错误] 创建 .venv 失败。
        goto :fail
    )
) else (
    echo [2/3] 检测到已有 .venv，跳过创建。
)

set "VENV_PY=.venv\Scripts\python.exe"
"%VENV_PY%" -m pip --version >nul 2>nul
if errorlevel 1 (
    "%VENV_PY%" -m ensurepip --upgrade
    if errorlevel 1 goto :fail
)

echo [3/3] 正在安装动作转换依赖 (requirements.txt: numpy, UnityPy) ...
set "PIP_EXTRA="
if "%USE_MIRROR%"=="1" set "PIP_EXTRA=-i https://pypi.tuna.tsinghua.edu.cn/simple"

"%VENV_PY%" -m pip install -r requirements.txt %PIP_EXTRA%
if errorlevel 1 (
    echo [警告] 默认源安装失败，尝试使用清华镜像重试 ...
    "%VENV_PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络连接后重试。
        goto :fail
    )
)

if "%WITH_BUILD%"=="1" (
    echo [可选] 正在安装打包工具 PyInstaller ...
    "%VENV_PY%" -m pip install pyinstaller %PIP_EXTRA%
    if errorlevel 1 goto :fail
)

"%VENV_PY%" -B -c "import tkinter, numpy, UnityPy, studio_gui, motion_pipeline; from scsp_motion.backend import BuiltinPoseBackend; BuiltinPoseBackend(); print('[OK] 动作转换依赖 (numpy / UnityPy / BuiltinPoseBackend) 与 Tkinter 全部就绪！')"
if errorlevel 1 (
    echo [错误] 动作转换环境自检未通过。
    goto :fail
)

echo.
echo ==================================================
echo   动作转换环境安装完成！
echo   启动界面: .venv\Scripts\pythonw.exe studio_gui.py
echo   命令行转换: .venv\Scripts\python.exe motion_pipeline.py build ^<任务目录^> --output ^<输出目录^>
echo ==================================================
call :maybe_pause
exit /b 0

:fail
echo.
echo [失败] 动作转换环境配置未完成，请根据上方提示处理后重试。
call :maybe_pause
exit /b 1

:maybe_pause
if "%NO_PAUSE%"=="1" exit /b 0
echo %cmdcmdline% | find /i "/c" >nul
if not errorlevel 1 pause
exit /b 0
