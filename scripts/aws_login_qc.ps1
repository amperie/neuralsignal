param(
    [string]$Profile = "qc",
    [string]$Region = "us-east-1",
    [switch]$Configure,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: scripts/aws_login_qc.ps1 [-Profile qc] [-Region us-east-1] [-Configure]"
    Write-Host ""
    Write-Host "Logs into the AWS CLI SSO profile used by NeuralSignal Terraform."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Profile     AWS CLI profile name. Default: qc"
    Write-Host "  -Region      Default AWS region for this process. Default: us-east-1"
    Write-Host "  -Configure   Run aws configure sso before login. Use this if the profile does not exist yet."
    Write-Host "  -Help        Show this help."
    exit 0
}

$ErrorActionPreference = "Stop"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH. Install or expose it first."
    }
}

Require-Command "aws"

$profileRegion = aws configure get region --profile $Profile 2>$null
$ssoStartUrl = aws configure get sso_start_url --profile $Profile 2>$null
$ssoSession = aws configure get sso_session --profile $Profile 2>$null

if ($Configure -or (-not $ssoStartUrl -and -not $ssoSession)) {
    Write-Host "Configuring AWS SSO profile '$Profile'..."
    aws configure sso --profile $Profile
}

if (-not $profileRegion) {
    aws configure set region $Region --profile $Profile
}

$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

Write-Host "Logging into AWS profile '$Profile'..."
aws sso login --profile $Profile

Write-Host "Verifying AWS auth..."
aws sts get-caller-identity --profile $Profile --output table

Write-Host "AWS profile '$Profile' is authenticated."
Write-Host "For this terminal session, run: `$env:AWS_PROFILE = '$Profile'"
