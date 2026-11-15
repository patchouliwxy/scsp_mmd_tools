$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
$gcc = if ($env:SCSP_GCC) {
    $env:SCSP_GCC
} elseif ($gccCmd) {
    $gccCmd.Source
} else {
    'D:\CODE\mingw64\bin\gcc.exe'
}
if (-not (Test-Path $gcc)) {
    throw "MinGW gcc not found in PATH or at $gcc (set SCSP_GCC to override)"
}
$ufbxInc = Join-Path $root 'ThirdParty\ufbx'
$ufbxSrc = Join-Path $root 'ThirdParty\ufbx\ufbx.c'
$bridgeSrc = Join-Path $root 'Tools\native\ufbx_bridge.c'
$outDll = Join-Path $root 'Tools\ufbx_bridge.dll'

& $gcc -O2 -Wall -Wextra -shared -static-libgcc "-I$ufbxInc" $ufbxSrc $bridgeSrc -o $outDll -lm
if ($LASTEXITCODE -ne 0) {
    throw "gcc failed with exit code $LASTEXITCODE"
}
Write-Host "Built $outDll ($( (Get-Item $outDll).Length ) bytes)"
