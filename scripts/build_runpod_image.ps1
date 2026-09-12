param(
    [string]$Image = "ghcr.io/amperie/neuralsignal-runpod-base:latest",
    [switch]$Push
)

Push-Location "$PSScriptRoot/.."
try {
    $OutputMode = if ($Push) { "--push" } else { "--load" }
    docker buildx build --platform linux/amd64 -f Dockerfile.runpod -t $Image $OutputMode .
    if ($LASTEXITCODE -ne 0) { throw "RunPod image build failed" }
} finally {
    Pop-Location
}

