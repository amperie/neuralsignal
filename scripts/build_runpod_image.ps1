param(
    [string]$Image = "ghcr.io/amperie/neuralsignal-runpod-base:latest"
)

$envPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match "^\s*GH_TOKEN=(.*)$") {
            $env:GH_TOKEN = $Matches[1].Trim().Trim("`\"")
        }
    }
}

if (-not $env:GH_TOKEN) {
    throw "GH_TOKEN is required in .env"
}

$Owner = ($Image -replace "^ghcr\.io/", "").Split("/")[0]
$env:GH_TOKEN | docker login ghcr.io -u $Owner --password-stdin
docker build -f Dockerfile.runpod -t $Image .
docker push $Image
