param(
    [bool]$Clean = $true,
    [switch]$OneFile,
    [switch]$IncludeGenICam
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python in .venv nicht gefunden: $python"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

function New-ZipFromArtifact {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            if (Test-Path $DestinationPath) {
                Remove-Item -LiteralPath $DestinationPath -Force
            }

            if (Test-Path $SourcePath -PathType Container) {
                [System.IO.Compression.ZipFile]::CreateFromDirectory(
                    $SourcePath,
                    $DestinationPath,
                    [System.IO.Compression.CompressionLevel]::Optimal,
                    $false
                )
            } else {
                $archive = [System.IO.Compression.ZipFile]::Open(
                    $DestinationPath,
                    [System.IO.Compression.ZipArchiveMode]::Create
                )
                try {
                    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                        $archive,
                        $SourcePath,
                        [System.IO.Path]::GetFileName($SourcePath),
                        [System.IO.Compression.CompressionLevel]::Optimal
                    ) | Out-Null
                }
                finally {
                    $archive.Dispose()
                }
            }
            return
        }
        catch {
            if ($attempt -eq 5) {
                throw
            }
            Start-Sleep -Seconds 3
        }
    }
}

Push-Location $repoRoot
try {
    if ($Clean) {
        foreach ($path in @("build", "dist", "release")) {
            $fullPath = Join-Path $repoRoot $path
            if (Test-Path $fullPath) {
                Remove-Item -LiteralPath $fullPath -Recurse -Force
            }
        }
    }

    & $python -m pip install -e ".[ui,build]"

    $includeHarvesters = $false
    if ($IncludeGenICam) {
        & $python -m pip install -e ".[genicam]"
        $includeHarvesters = $true
    } else {
        & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('harvesters') else 1)"
        if ($LASTEXITCODE -eq 0) {
            $includeHarvesters = $true
        }
    }

    $pyiArgs = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name", "camcalib2-gui",
        "--paths", "src",
        "--collect-data", "camcalib2.patterns",
        "src/camcalib2/gui_launcher.py"
    )

    if ($OneFile) {
        $pyiArgs += "--onefile"
    }

    if ($includeHarvesters) {
        $pyiArgs += @(
            "--hidden-import", "camcalib2.capture.genicam",
            "--collect-submodules", "harvesters"
        )
    }

    & $python $pyiArgs
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller-Build fehlgeschlagen."
    }

    $releaseDir = Join-Path $repoRoot "release"
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

    if ($OneFile) {
        $artifact = Join-Path $repoRoot "dist\camcalib2-gui.exe"
        $zipPath = Join-Path $releaseDir "camcalib2-gui-windows-onefile.zip"
        New-ZipFromArtifact -SourcePath $artifact -DestinationPath $zipPath
    } else {
        $artifact = Join-Path $repoRoot "dist\camcalib2-gui"
        $zipPath = Join-Path $releaseDir "camcalib2-gui-windows.zip"
        New-ZipFromArtifact -SourcePath $artifact -DestinationPath $zipPath
    }

    Write-Host "Build fertig:"
    Write-Host "  Artifact: $artifact"
    Write-Host "  ZIP:      $zipPath"
    if ($includeHarvesters) {
        Write-Host "  GenICam:  eingebunden (CTI/SDKs werden auf dem Zielrechner weiter benoetigt)"
    } else {
        Write-Host "  GenICam:  nicht eingebunden"
    }
}
finally {
    Pop-Location
}
