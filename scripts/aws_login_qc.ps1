param(
    [string]$Profile = "qc",
    [string]$Region = "us-east-1",
    [switch]$Configure,
    [string]$AccessKeyId = "",
    [string]$SecretAccessKey = "",
    [string]$SessionToken = "",
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: scripts/aws_login_qc.ps1 [-Profile qc] [-Region us-east-1] [-Configure]"
    Write-Host "       scripts/aws_login_qc.ps1 -AccessKeyId <id> -SecretAccessKey <secret> [-SessionToken <token>]"
    Write-Host ""
    Write-Host "Configures and verifies a standard AWS CLI access-key profile for NeuralSignal Terraform."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Profile          AWS CLI profile name. Default: qc"
    Write-Host "  -Region           Default AWS region. Default: us-east-1"
    Write-Host "  -Configure        Run interactive aws configure for the profile."
    Write-Host "  -AccessKeyId      AWS access key id for non-interactive profile setup."
    Write-Host "  -SecretAccessKey  AWS secret access key for non-interactive profile setup."
    Write-Host "  -SessionToken     Optional AWS session token for temporary credentials."
    Write-Host "  -Help             Show this help."
    exit 0
}

$ErrorActionPreference = "Stop"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH. Install or expose it first."
    }
}

function Get-AwsConfigValue($Name, $ProfileName) {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $value = & aws configure get $Name --profile $ProfileName 2>$null
        if ($LASTEXITCODE -ne 0) {
            return ""
        }
        return ($value | Out-String).Trim()
    }
    finally {
        $script:ErrorActionPreference = $previousErrorActionPreference
    }
}

function Invoke-CheckedAws($Arguments, [string]$FailureMessage = "AWS command failed.") {
    & aws @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Configure-ProfileFromArgs() {
    if (-not $AccessKeyId -or -not $SecretAccessKey) {
        throw "Both -AccessKeyId and -SecretAccessKey are required for non-interactive setup. Otherwise use -Configure."
    }

    Write-Host "Configuring AWS profile '$Profile' from script parameters..."
    Invoke-CheckedAws @("configure", "set", "aws_access_key_id", $AccessKeyId, "--profile", $Profile) "Failed to set AWS access key id."
    Invoke-CheckedAws @("configure", "set", "aws_secret_access_key", $SecretAccessKey, "--profile", $Profile) "Failed to set AWS secret access key."
    if ($SessionToken) {
        Invoke-CheckedAws @("configure", "set", "aws_session_token", $SessionToken, "--profile", $Profile) "Failed to set AWS session token."
    }
}

Require-Command "aws"

$profileAccessKey = Get-AwsConfigValue "aws_access_key_id" $Profile
$profileRegion = Get-AwsConfigValue "region" $Profile

if ($AccessKeyId -or $SecretAccessKey -or $SessionToken) {
    Configure-ProfileFromArgs
} elseif ($Configure -or -not $profileAccessKey) {
    Write-Host "Configuring AWS profile '$Profile'..."
    Invoke-CheckedAws @("configure", "--profile", $Profile) "AWS profile configuration failed."
}

if (-not $profileRegion) {
    Invoke-CheckedAws @("configure", "set", "region", $Region, "--profile", $Profile) "Failed to set AWS region."
}

$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

Write-Host "Verifying AWS auth for profile '$Profile'..."
Invoke-CheckedAws @("sts", "get-caller-identity", "--profile", $Profile, "--output", "table") "AWS auth verification failed. Check the profile credentials."

Write-Host "AWS profile '$Profile' is configured and authenticated."
Write-Host "For this terminal session, run: `$env:AWS_PROFILE = '$Profile'"
