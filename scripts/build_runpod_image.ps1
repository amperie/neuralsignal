param(
    [string]$Image = "ghcr.io/amperie/neuralsignal-runpod-base:latest",
    [switch]$Push
)

docker build -f Dockerfile.runpod -t $Image .
if ($Push) {
    docker push $Image
}

