@echo off
rem Build GIOVA as a folder app: dist\GIOVA\GIOVA.exe
rem One folder, not one file: a one-file exe unpacks ~300 MB to a temp folder on
rem every start (slow, and antivirus tends to flag it).
rem Recordings are written to the project's Data folder while the exe sits in dist\GIOVA,
rem and to a Data folder next to the exe once it is copied anywhere else.
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

rem Tests first: do not package a build that fails them.
%PY% -m unittest discover -s tests -p "test_*.py" || goto :fail

rem The eeg folder travels into the exe whole (--add-data below). GIOVA 1.0.0 run from source
rem recorded into eeg\Data when the project had no Data folder: never package recordings.
if exist eeg\Data (
  echo eeg\Data holds recordings: move it to the project folder ^(GIOVA\Data^) before building.
  goto :fail
)

rem Exe icon: GIOVA's .ico (eeg\ui\assets\giova.ico).
if not exist build_icon mkdir build_icon
%PY% eeg_studio.py --icon build_icon\giova.ico || goto :fail
rem the logo, the .ico and the fonts travel with the code
set ASSETS=--add-data "eeg\ui\assets;eeg\ui\assets"

%PY% -m PyInstaller --noconfirm --clean --windowed ^
  --name GIOVA --icon build_icon\giova.ico %ASSETS% ^
  --add-data "eeg;python\eeg" ^
  --collect-binaries brainflow ^
  --collect-submodules pyqtgraph ^
  --collect-submodules mne --collect-data mne ^
  --collect-all pygame ^
  --collect-submodules sklearn --collect-submodules pyriemann ^
  --collect-submodules array_api_compat --collect-submodules array_api_extra ^
  --collect-submodules pywt ^
  --collect-submodules h5py --collect-submodules pymatreader ^
  --exclude-module tkinter ^
  eeg_studio.py || goto :fail

rem The licences travel with the program: GIOVA's (GPL-3.0) and those of everything it carries.
%PY% tools\third_party.py --copy dist\GIOVA\licenses || goto :fail
copy /y LICENSE dist\GIOVA\LICENSE.txt >nul
copy /y THIRD_PARTY_NOTICES.md dist\GIOVA\THIRD_PARTY_NOTICES.md >nul
copy /y README.md dist\GIOVA\README.md >nul

rem The packaged exe must pass its own check (BrainFlow DLLs, SciPy, Qt).
start /wait "" dist\GIOVA\GIOVA.exe --selftest
type dist\GIOVA\selftest.txt
findstr /c:"SELFTEST OK" dist\GIOVA\selftest.txt >nul || goto :fail

rem The download: the folder as it is, without the self-test's report or any recordings.
del /q dist\GIOVA\selftest.txt
if exist dist\GIOVA\Data rmdir /s /q dist\GIOVA\Data
for /f %%v in ('%PY% -c "import eeg; print(eeg.VERSION)"') do set VERSION=%%v
set ZIP=dist\GIOVA-%VERSION%-windows-x64.zip
if exist %ZIP% del /q %ZIP%
powershell -NoProfile -Command "Compress-Archive -Path dist\GIOVA -DestinationPath %ZIP%" || goto :fail
echo.
echo Built dist\GIOVA\GIOVA.exe and %ZIP%
exit /b 0

:fail
echo Build FAILED
exit /b 1
