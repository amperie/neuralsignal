param(
    [string]$Profile = "qc",
    [string]$Region = "us-east-1",
    [switch]$Configure,
    [string]$SsoStartUrl = "",
    [string]$SsoRegion = "",
    [string]$SsoAccountId = "",
    [string]$SsoRoleName = "",
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: scripts/aws_login_qc.ps1 [-Profile qc] [-Region us-east-1] [-Configure]"
    Write-Host "       scripts/aws_login_qc.ps1 -SsoStartUrl <url> -SsoRegion <region> -SsoAccountId <id> -SsoRoleName <role>"
    Write-Host ""
    Write-Host "Logs into the AWS CLI SSO profile used by NeuralSignal Terraform."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Profile       AWS CLI profile name. Default: qc"
    Write-Host "  -Region        Default AWS region for this process. Default: us-east-1"
    Write-Host "  -Configure     Run aws configure sso before login. Use this if the profile does not exist yet."
    Write-Host "  -SsoStartUrl   SSO start URL for non-interactive profile setup."
    Write-Host "  -SsoRegion     SSO region for non-interactive profile setup."
    Write-Host "  -SsoAccountId  AWS account id for non-interactive profile setup."
    Write-Host "  -SsoRoleName   AWS role name for non-interactive profile setup."
    Write-Host "  -Help          Show this help."
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

function Invoke-CheckedAws($Arguments) {
    & aws @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS command failed: aws $($Arguments -join ' ')"
    }
}

function Configure-SsoProfile() {
    $hasNonInteractiveConfig = $SsoStartUrl -and $SsoRegion -and $SsoAccountId -and $SsoRoleName
    if ($hasNonInteractiveConfig) {
        Write-Host "Configuring AWS SSO profile '$Profile' from script parameters..."
        Invoke-CheckedAws @("configure", "set", "sso_start_url", $SsoStartUrl, "--profile", $Profile)
        Invoke-CheckedAws @("configure", "set", "sso_region", $SsoRegion, "--profile", $Profile)
        Invoke-CheckedAws @("configure", "set", "sso_account_id", $SsoAccountId, "--profile", $Profile)
        Invoke-CheckedAws @("configure", "set", "sso_role_name", $SsoRoleName, "--profile", $Profile)
        return
    }

    Write-Host "Configuring AWS SSO profile '$Profile'..."
    Invoke-CheckedAws @("configure", "sso", "--profile", $Profile)
}

Require-Command "aws"

$profileRegion = Get-AwsConfigValue "region" $Profile
$ssoStartUrl = Get-AwsConfigValue "sso_start_url" $Profile
$ssoSession = Get-AwsConfigValue "sso_session" $Profile

if ($Configure -or (-not $ssoStartUrl -and -not $ssoSession)) {
    Configure-SsoProfile
}

if (-not $profileRegion) {
    Invoke-CheckedAws @("configure", "set", "region", $Region, "--profile", $Profile)
}

$env:AWS_PROFILE = $Profile
$env:AWS_DEFAULT_REGION = $Region

Write-Host "Logging into AWS profile '$Profile'..."
Invoke-CheckedAws @("sso", "login", "--profile", $Profile)

Write-Host "Verifying AWS auth..."
Invoke-CheckedAws @("sts", "get-caller-identity", "--profile", $Profile, "--output", "table")

Write-Host "AWS profile '$Profile' is authenticated."
Write-Host "For this terminal session, run: `$env:AWS_PROFILE = '$Profile'"
