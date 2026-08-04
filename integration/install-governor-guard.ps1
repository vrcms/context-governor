# install-governor-guard.ps1 — Creates the Scheduled Task that keeps the governor always on.
# Uses pythonw.exe directly — no console, no taskbar entry, no wrapper.

param(
    [string]$TaskName = 'ContextGovernor',
    [string]$User = ''
)

$ErrorActionPreference = 'Stop'

$scriptDir   = Split-Path $MyInvocation.MyCommand.Path -Parent
$projectRoot = Split-Path $scriptDir -Parent
$pythonw     = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$runner      = Join-Path $scriptDir 'governor-runner.py'

if (-not (Test-Path $pythonw)) {
    Write-Error "pythonw.exe not found at $pythonw"
    exit 1
}
if (-not $User) { $User = $env:USERNAME }

Write-Host "Using: $pythonw" -ForegroundColor Cyan

# remove any old version
Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue |
    Unregister-ScheduledTask -Confirm:$false

$action    = New-ScheduledTaskAction -Execute $pythonw `
                -Argument "`"$runner`"" `
                -WorkingDirectory $projectRoot
$logon     = New-ScheduledTaskTrigger -AtLogOn -User $User
$keepAlive = New-ScheduledTaskTrigger -Once -At (Get-Date) `
                -RepetitionInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -StartWhenAvailable `
                -MultipleInstances IgnoreNew `
                -ExecutionTimeLimit ([TimeSpan]::Zero)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action `
    -Trigger @($logon, $keepAlive) `
    -Principal $principal `
    -Settings $settings `
    -Description "Context Governor — pythonw.exe direct, no console/taskbar" `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName
Write-Host "Task '$TaskName' installed." -ForegroundColor Green
Write-Host "  Execute : $pythonw" -ForegroundColor Gray
Write-Host "  Args    : $runner" -ForegroundColor Gray
Write-Host "  WorkDir : $projectRoot" -ForegroundColor Gray
Write-Host "  No window, no taskbar entry." -ForegroundColor Green
