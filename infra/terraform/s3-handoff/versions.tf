terraform {
  required_version = ">= 1.6.0"

backend "local" {
  path = "terraform.tfstate"
}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
