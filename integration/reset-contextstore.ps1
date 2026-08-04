# reset-contextstore.ps1 — find whatever actually has contextstore/ open (via
# the Windows Restart Manager API, not a guess at command-line patterns), stop
# it, then reset the store. No reboot needed.
#
# Root cause: contextstore/ is SQLite in WAL mode (hotness.db, index.db +
# their -shm/-wal companions). Windows keeps those files locked as long as ANY
# process has an open connection to them. `python -m contextmanager.launcher`
# (the governor) is what opens them.
#
# v2 note: an earlier version of this script matched processes by command
# line ("contextmanager.launcher"). That failed the first time it was used —
# the actual lock holder was a python.exe running ELEVATED (as Administrator),
# and a non-elevated query can see that such a process exists but cannot read
# its command line, so it silently didn't match. This version (a) asks
# Windows directly which process(es) hold the folder's files open via
# RmGetList, so it doesn't depend on guessing a command-line pattern, and
# (b) self-elevates via UAC if not already running as Administrator, since a
# standard-integrity process can't stop a higher-integrity one regardless of
# how it's identified.
#
# Usage:
#   .\integration\reset-contextstore.ps1              # backup + fresh empty store (default, reversible)
#   .\integration\reset-contextstore.ps1 -Permanent   # skip backup, delete outright
#   .\integration\reset-contextstore.ps1 -WhatIf      # show what would happen, change nothing

param(
    [switch]$Permanent,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

# ---- 0. self-elevate if needed (stopping another process's handles requires it) ----
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Not elevated -- relaunching as Administrator (UAC prompt incoming)..." -ForegroundColor Yellow
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    if ($Permanent) { $argList += "-Permanent" }
    if ($WhatIf)     { $argList += "-WhatIf" }
    Start-Process pwsh -Verb RunAs -ArgumentList $argList -Wait
    exit $LASTEXITCODE
}

$storeDir = Join-Path $PSScriptRoot "contextstore"
if (-not (Test-Path $storeDir)) {
    Write-Host "No contextstore folder at $storeDir -- nothing to do." -ForegroundColor Yellow
    exit 0
}

# ---- 1. ask Windows (Restart Manager) exactly which process(es) hold these files open ----
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class RmHandle {
    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    public static extern int RmStartSession(out uint pSessionHandle, int dwSessionFlags, string strSessionKey);
    [DllImport("rstrtmgr.dll")]
    public static extern int RmEndSession(uint pSessionHandle);
    [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
    public static extern int RmRegisterResources(uint pSessionHandle, uint nFiles, string[] rgsFilenames,
        uint nApplications, IntPtr rgApplications, uint nServices, string[] rgsServiceNames);
    [DllImport("rstrtmgr.dll")]
    public static extern int RmGetList(uint dwSessionHandle, out uint pnProcInfoNeeded, ref uint pnProcInfo,
        [In, Out] RM_PROCESS_INFO[] rgAffectedApps, ref uint lpdwRebootReasons);

    [StructLayout(LayoutKind.Sequential)]
    public struct RM_UNIQUE_PROCESS {
        public int dwProcessId;
        public System.Runtime.InteropServices.ComTypes.FILETIME ProcessStartTime;
    }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct RM_PROCESS_INFO {
        public RM_UNIQUE_PROCESS Process;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string strAppName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)] public string strServiceShortName;
        public int ApplicationType;
        public uint AppStatus;
        public uint TSSessionId;
        [MarshalAs(UnmanagedType.Bool)] public bool bRestartable;
    }
}
"@

function Get-LockingPids([string]$path) {
    $files = @(Get-ChildItem -Path $path -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    if (-not $files) { $files = @($path) }
    [uint32]$session = 0
    $key = [Guid]::NewGuid().ToString()
    if ([RmHandle]::RmStartSession([ref]$session, 0, $key) -ne 0) { return @() }
    try {
        if ([RmHandle]::RmRegisterResources($session, [uint32]$files.Count, $files, 0, [IntPtr]::Zero, 0, $null) -ne 0) { return @() }
        [uint32]$needed = 0; [uint32]$info = 0; [uint32]$reasons = 0
        [RmHandle]::RmGetList($session, [ref]$needed, [ref]$info, $null, [ref]$reasons) | Out-Null
        if ($needed -eq 0) { return @() }
        $arr = New-Object 'RmHandle+RM_PROCESS_INFO[]' $needed
        $info = $needed
        [RmHandle]::RmGetList($session, [ref]$needed, [ref]$info, $arr, [ref]$reasons) | Out-Null
        return $arr | ForEach-Object { $_.Process.dwProcessId }
    } finally {
        [RmHandle]::RmEndSession($session) | Out-Null
    }
}

$pids = @(Get-LockingPids $storeDir | Select-Object -Unique)

if ($pids) {
    Write-Host "Found $($pids.Count) process(es) holding contextstore open:" -ForegroundColor Yellow
    foreach ($procId in $pids) {
        $info = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
        $desc = if ($info) { "$($info.Name) -- $($info.CommandLine)" } else { "(process already gone)" }
        Write-Host "  PID ${procId}: $desc"
    }

    if ($WhatIf) {
        Write-Host "`n-WhatIf: would stop the above, then reset contextstore. Nothing changed." -ForegroundColor Cyan
        exit 0
    }

    foreach ($procId in $pids) {
        Write-Host "Stopping PID $procId..." -ForegroundColor Yellow
        Stop-Process -Id $procId -ErrorAction SilentlyContinue
    }

    $deadline = (Get-Date).AddSeconds(5)
    foreach ($procId in $pids) {
        while ((Get-Date) -lt $deadline -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
            Start-Sleep -Milliseconds 200
        }
        if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            Write-Host "PID $procId still alive after 5s, forcing..." -ForegroundColor Yellow
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Milliseconds 300   # brief grace period for Windows to actually release the file locks
} else {
    Write-Host "Restart Manager reports nothing holding contextstore open -- should already be unlocked." -ForegroundColor Green
    if ($WhatIf) {
        Write-Host "-WhatIf: would reset contextstore now. Nothing changed." -ForegroundColor Cyan
        exit 0
    }
}

# ---- 2. reset the store ----
try {
    if ($Permanent) {
        Write-Host "Deleting $storeDir permanently..." -ForegroundColor Yellow
        Remove-Item -Path $storeDir -Recurse -Force
    } else {
        $backupName = "contextstore.bak." + (Get-Date -Format "yyyyMMdd-HHmmss")
        Write-Host "Renaming $storeDir -> $backupName (recoverable; delete old .bak.* folders yourself when ready)" -ForegroundColor Yellow
        Rename-Item -Path $storeDir -NewName $backupName
    }
    New-Item -ItemType Directory -Path $storeDir | Out-Null
    Write-Host "Fresh empty contextstore ready at $storeDir." -ForegroundColor Cyan
} catch {
    Write-Host "Still couldn't touch $storeDir -- re-run with -WhatIf to see what Restart Manager finds now." -ForegroundColor Red
    throw
}

Write-Host "`nDone. Any CLI session using the governor will need to reconnect/respawn it." -ForegroundColor Cyan
