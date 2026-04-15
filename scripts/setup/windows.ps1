# Windows setup — PowerShell 7+.
# Uses winget (preferred) with choco fallback. CUDA works; SYCL via oneAPI works.
# Requires VS 2022 Build Tools for MSVC. Alternatively install clang and build via MSYS2.

[CmdletBinding()]
param(
  [switch]$EnableCuda,
  [switch]$EnableSycl,
  [switch]$InstallLinters = $true
)

$ErrorActionPreference = "Stop"

function Install-WithFallback {
  param([string]$WingetId, [string]$ChocoId, [string]$DisplayName)
  Write-Host "--- $DisplayName ---"
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id $WingetId --silent --accept-package-agreements --accept-source-agreements
  } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
    choco install $ChocoId -y
  } else {
    throw "Neither winget nor choco is installed. Install one first."
  }
}

Write-Host "=== Windows setup for vmaf fork ==="

# Core toolchain.
Install-WithFallback "Python.Python.3.12"           "python"          "Python 3.12"
Install-WithFallback "Ninja-build.Ninja"            "ninja"           "Ninja"
Install-WithFallback "mesonbuild.meson"             "meson"           "Meson"
Install-WithFallback "NASM.NASM"                    "nasm"            "NASM"
Install-WithFallback "Git.Git"                      "git"             "Git"
Install-WithFallback "Doxygen.Doxygen"              "doxygen.install" "Doxygen"
Install-WithFallback "LLVM.LLVM"                    "llvm"            "LLVM / clang"

# MSVC Build Tools (large download). Skip if user already has VS installed.
if (-not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools")) {
  Install-WithFallback "Microsoft.VisualStudio.2022.BuildTools" "visualstudio2022buildtools" "VS 2022 Build Tools"
  Write-Host "After install, launch 'Visual Studio Installer' and add"
  Write-Host "  'Desktop development with C++' workload (required for MSVC)."
}

if ($InstallLinters) {
  python -m pip install --user --upgrade pre-commit ruff black isort mypy semgrep
  # clang-tidy and clang-format come with LLVM install above.
  # shellcheck + shfmt via scoop or WSL if needed — not strictly required on Windows.
}

if ($EnableCuda) {
  Write-Host "--- NVIDIA CUDA Toolkit ---"
  Install-WithFallback "Nvidia.CUDA" "cuda" "CUDA Toolkit"
  Write-Host "Ensure %CUDA_PATH%\bin is in PATH after install."
}

if ($EnableSycl) {
  Write-Host "--- Intel oneAPI Base Toolkit ---"
  Install-WithFallback "Intel.oneAPI.BaseToolkit" "intel-oneapi-base-toolkit" "Intel oneAPI"
  Write-Host "Before building, run:"
  Write-Host '  & "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"'
}

Write-Host ""
Write-Host "=== done. next steps (from a 'x64 Native Tools Command Prompt for VS 2022' or PowerShell with vcvars loaded) ==="
$cuda = if ($EnableCuda) { "true" } else { "false" }
$sycl = if ($EnableSycl) { "true" } else { "false" }
Write-Host "  meson setup build -Denable_cuda=$cuda -Denable_sycl=$sycl"
Write-Host "  meson compile -C build"
