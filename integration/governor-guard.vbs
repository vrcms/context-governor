' governor-guard.vbs — Truly invisible launcher for the Context Governor.
' Scheduled Task runs THIS via wscript.exe (no console, no taskbar).
' The True flag makes wscript.exe wait for pythonw.exe to exit,
' so the Task Scheduler can track whether the guard is alive.
' Paths are derived from this script's own location (repo root is its
' grandparent), so the guard works from any checkout — same posture as
' run-governor.ps1's $PSScriptRoot.
Dim shell, fso, pythonw, workDir
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
workDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
pythonw = fso.BuildPath(workDir, ".venv\Scripts\pythonw.exe")
shell.CurrentDirectory = workDir
shell.Run """" & pythonw & """ -m contextmanager.launcher", 0, True
