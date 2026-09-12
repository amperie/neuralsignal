param(
    [string]$Command = "plan",
    [string]$LoginProfile = "qc",
    [string]$Region = "us-west-1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TerraformArgs
)

$ErrorActionPreference = "Stop"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH."
    }
}

function Export-AwsLoginCredentials($ProfileName) {
    aws configure export-credentials --profile $ProfileName --format env | ForEach-Object {
        if ($_ -match '^export\s+([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not export AWS credentials from profile '$ProfileName'. Run: aws login --profile $ProfileName"
    }
}

function AwsAccountId() {
    $account = aws sts get-caller-identity --query Account --output text
    if ($LASTEXITCODE -ne 0 -or -not $account.Trim()) {
        throw "Could not read AWS account id."
    }
    return $account.Trim()
}

function BackendConfigFromOutput() {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $value = terraform output -raw terraform_backend_hcl 2>$null
        if ($LASTEXITCODE -eq 0 -and $value.Trim()) {
            return $value.Trim()
        }
        return ""
    }
    finally {
        $script:ErrorActionPreference = $previous
    }
}

function Ensure-BackendConfig($Path) {
    if (Test-Path $Path) {
        return
    }

    $config = BackendConfigFromOutput
    if (-not $config) {
        $account = AwsAccountId
        $config = @"
bucket  = "neuralsignal-terraform-state-$account"
key     = "neuralsignal/s3-handoff/terraform.tfstate"
region  = "$Region"
encrypt = true
"@.Trim()
    }

    Set-Content -Path $Path -Value $config -Encoding UTF8
    Write-Host "Wrote $Path"
}

Require-Command "aws"
Require-Command "terraform"
Export-AwsLoginCredentials $LoginProfile

$env:AWS_DEFAULT_REGION = $Region
$env:AWS_REGION = $Region
$env:AWS_EC2_METADATA_DISABLED = "true"

$backendFile = "backend.hcl"
Ensure-BackendConfig $backendFile

$fullArgs = @($Command) + @($TerraformArgs)
if ($Command -eq "init" -and -not ($TerraformArgs -match '^-backend-config')) {
    $fullArgs += "-backend-config=$backendFile"
    if ((Test-Path "terraform.tfstate") -and -not ($TerraformArgs -contains "-migrate-state")) {
        $fullArgs += @("-migrate-state", "-force-copy")
    }
}

$commandsWithVars = @("plan", "apply", "destroy", "refresh", "import")
if ($commandsWithVars -contains $Command) {
    $fullArgs += @("-var=aws_profile=", "-var=aws_region=$Region")
}

terraform @fullArgs
exit $LASTEXITCODE

