param(
    [string]$Python = 'python',
    [string]$Version = 'v0.1',
    [switch]$CreateZip
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot '.builddeps'
$RuntimeTools = Join-Path $PSScriptRoot 'Build/runtime/Tools'
if (Test-Path -LiteralPath $RuntimeTools) {
    Remove-Item -LiteralPath $RuntimeTools -Recurse -Force
}
New-Item -ItemType Directory -Path $RuntimeTools -Force | Out-Null
$ExcludeToolPatterns = @(
    'better_fbx_headless.py',
    'check_*.py',
    'diagnose_*.py',
    'finish_export_ready.py',
    'finish_face_cleanup.py',
    'finish_merge_shapes.py',
    'inspect_*.py',
    'render_*.py',
    'try_twist_merge.py'
)
Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'Tools') -Filter '*.py' -File |
    Where-Object {
        $name = $_.Name
        -not ($ExcludeToolPatterns | Where-Object { $name -like $_ })
    } |
    Copy-Item -Destination $RuntimeTools
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Tools\ufbx_bridge.dll') -Destination $RuntimeTools -Force
# Build into a separate directory: PyInstaller clears its destination.
# User Outputs inside the installed release must never be included in that clear.
$PackageRoot = Join-Path $PSScriptRoot 'Build/package'
& $Python -m PyInstaller --noconfirm --clean --windowed --onedir --name scsp_mmd_tools --hidden-import studio_pipeline --add-data 'Build/runtime/Tools;Tools' --add-data 'Resources;Resources' --add-data 'ThirdParty/ufbx/LICENSE;ThirdParty/ufbx' --distpath $PackageRoot --workpath Build/pyinstaller studio_gui.py
if ($LASTEXITCODE -ne 0) { throw '打包失败' }
$Built = Join-Path $PackageRoot 'scsp_mmd_tools'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $Built '使用说明.md') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.en.md') -Destination (Join-Path $Built 'README.en.md') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'LICENSE') -Destination (Join-Path $Built 'LICENSE') -Force

$Installed = Join-Path $PSScriptRoot 'Releases/scsp_mmd_tools'
New-Item -ItemType Directory -Path $Installed -Force | Out-Null
$LegacyInstalled = Join-Path $PSScriptRoot 'Releases/SCSPBlenderPmxStudio'
if (Test-Path -LiteralPath $LegacyInstalled -PathType Container) {
    foreach ($Preserve in @('Outputs', 'studio_settings.json')) {
        $LegacyItem = Join-Path $LegacyInstalled $Preserve
        $TargetItem = Join-Path $Installed $Preserve
        if ((Test-Path -LiteralPath $LegacyItem) -and -not (Test-Path -LiteralPath $TargetItem)) {
            Move-Item -LiteralPath $LegacyItem -Destination $TargetItem
        }
    }
}
# Copy only program files, preserving user outputs; skip identical locked files.
$BuiltPrefix = $Built.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
foreach ($ProgramFile in (Get-ChildItem -LiteralPath $Built -File -Recurse)) {
    $Relative = $ProgramFile.FullName.Substring($BuiltPrefix.Length)
    $Destination = Join-Path $Installed $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    $Same = (Test-Path -LiteralPath $Destination -PathType Leaf) -and ((Get-FileHash -LiteralPath $Destination).Hash -eq (Get-FileHash -LiteralPath $ProgramFile.FullName).Hash)
    if (-not $Same) { Copy-Item -LiteralPath $ProgramFile.FullName -Destination $Destination -Force }
}

if ($CreateZip) {
    $ZipName = "scsp_mmd_tools-$Version-windows-x64.zip"
    $ZipPath = Join-Path $PSScriptRoot ("Releases/" + $ZipName)
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -LiteralPath $ZipPath -Force }
    Compress-Archive -LiteralPath $Built -DestinationPath $ZipPath -Force
    $ZipHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath ($ZipPath + '.sha256') -Value "$ZipHash  $ZipName" -Encoding Ascii
}
