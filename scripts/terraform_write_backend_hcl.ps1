param(
    [string]$TerraformDir = "infra/terraform/s3-handoff",
    [string]$BackendFile = "backend.hcl",
    [switch]$Migrate,
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: scripts/terraform_write_backend_hcl.ps1 [-TerraformDir infra/terraform/s3-handoff] [-BackendFile backend.hcl] [-Migrate]"
    Write-Host ""
    Write-Host "Writes backend.hcl from the terraform_backend_hcl output after the first local-state apply."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -TerraformDir  Terraform module directory. Default: infra/terraform/s3-handoff"
    Write-Host "  -BackendFile   Backend config filename inside TerraformDir. Default: backend.hcl"
    Write-Host "  -Migrate       Run terraform init -backend-config=<BackendFile> -migrate-state after writing the file."
    Write-Host "  -Help          Show this help."
    exit 0
}

$ErrorActionPreference = "Stop"

function Require-Command($Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH. Install or expose it first."
    }
}

Require-Command "terraform"

$resolvedTerraformDir = Resolve-Path $TerraformDir
$backendPath = Join-Path $resolvedTerraformDir $BackendFile

Write-Host "Reading terraform_backend_hcl from '$resolvedTerraformDir'..."
$backendConfig = terraform -chdir=$resolvedTerraformDir output -raw terraform_backend_hcl
if ($LASTEXITCODE -ne 0 -or -not $backendConfig.Trim()) {
    throw "Could not read terraform_backend_hcl. Apply the stack once with local state first: terraform -chdir=$TerraformDir apply"
}

Set-Content -Path $backendPath -Value $backendConfig -Encoding UTF8
Write-Host "Wrote $backendPath"

if ($Migrate) {
    Write-Host "Migrating Terraform state to S3 backend..."
    terraform -chdir=$resolvedTerraformDir init -backend-config=$BackendFile -migrate-state
    if ($LASTEXITCODE -ne 0) {
        throw "Terraform backend migration failed."
    }
}
