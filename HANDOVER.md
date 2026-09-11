# RecSys Loom machine handover

State captured on 11 September 2026. Read `AGENTS.md` and `README.md` before
changing the repository or AWS account.

## Important status

The recommendation/search project and its DevOps lab are implemented, but the
public application is not deployed.

- Repository: `anshulLuhsna/recsys-loom`
- Working branch: `overnight-ranking-research`
- Base branch: `main`
- Pull request: <https://github.com/anshulLuhsna/recsys-loom/pull/1>
- Pull request state: closed without merge
- Last completed PR CI: all nine checks passed
- `main` currently contains only `.gitignore` and `README.md`
- Vercel is not configured
- ECR repositories contain no images
- The synthetic serving bundle has not been uploaded
- GitHub deployment variables and the `production` environment are not configured
- `main` is not protected, so backend CD cannot run

Do not claim that the application is live. The EC2 host and DNS exist, but no
application containers or HTTPS service have been deployed.

## AWS resources that exist

AWS account `576556099107`, deployment region `ap-south-1`:

- EC2 instance: `i-078ca37b5f05e46d7`
- Elastic IP: `13.205.158.182`
- API DNS: `recsys-api.arbityr.live` resolves to the Elastic IP
- Instance administration: AWS Systems Manager only
- Inbound SSH: disabled
- Lab fault volume: disabled
- State bucket: `recsys-loom-tfstate-576556099107`
- Bundle bucket: `recsys-loom-bundles-576556099107`
- ECR repositories: `recsys-loom-recommendation`, `recsys-loom-search`
- GitHub OIDC role: `arn:aws:iam::576556099107:role/recsys-loom-github-ci`

The EC2 instance, EBS volume, Elastic IP, and CloudWatch resources are billable
now. Destroy `infra/live` if the deployment is abandoned.

Terraform state locations:

- Bootstrap: `s3://recsys-loom-tfstate-576556099107/recsys-loom/bootstrap/terraform.tfstate`
- Live: `s3://recsys-loom-tfstate-576556099107/recsys-loom/demo/development/terraform.tfstate`

Never commit Terraform state, `.env`, AWS credentials, raw H&M CSV files,
images, model caches, or generated serving artifacts.

## Set up the new machine

Install Git, GitHub CLI, AWS CLI v2, Terraform 1.10 or newer, Python 3.12, and
Node.js 22. On macOS:

```bash
brew install gh awscli
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

Authenticate GitHub and clone the working branch:

```bash
gh auth login --hostname github.com --git-protocol ssh --web
git clone --branch overnight-ranking-research \
  git@github.com:anshulLuhsna/recsys-loom.git
cd recsys-loom
```

If the new machine uses the existing personal SSH alias, use
`git@github.com-personal:anshulLuhsna/recsys-loom.git` instead.

Configure AWS IAM Identity Center:

```bash
aws configure sso --profile recsys-loom
```

Use these values:

```text
SSO session name: recsys-loom
SSO start URL: https://d-9066701a12.awsapps.com/start
SSO region: us-east-1
SSO registration scopes: sso:account:access
AWS account: 576556099107
Role: AdministratorAccess
Default client region: ap-south-1
Output: json
```

Then authenticate and verify:

```bash
aws sso login --profile recsys-loom
aws sts get-caller-identity --profile recsys-loom
export AWS_PROFILE=recsys-loom
```

## Reconnect Terraform

Local `*.tfvars` and `backend.hcl` files are intentionally ignored. Recreate
them from the tracked examples.

For bootstrap:

```bash
cd infra/bootstrap
cp backend.hcl.example backend.hcl
cp terraform.tfvars.example terraform.tfvars
```

Set the following values:

```hcl
# backend.hcl
bucket = "recsys-loom-tfstate-576556099107"

# terraform.tfvars
state_bucket_name          = "recsys-loom-tfstate-576556099107"
serving_bundle_bucket_name = "recsys-loom-bundles-576556099107"
github_repository          = "anshulLuhsna/recsys-loom"
github_branch              = "main"

extra_tags = {
  Owner = "anshul"
}
```

Reconnect to remote state and confirm that Terraform proposes no changes:

```bash
terraform init -backend-config=backend.hcl
terraform plan
```

For the live tenant:

```bash
cd ../live
cp backend.hcl.example backend.hcl
cp tenant.tfvars.example tenant.tfvars
cp environment.tfvars.example environment.tfvars
```

Set the live backend bucket to `recsys-loom-tfstate-576556099107`. Replace both
example ECR account IDs with `576556099107`. Set:

```hcl
serving_bundle_bucket_arn = "arn:aws:s3:::recsys-loom-bundles-576556099107"

extra_tags = {
  Owner = "anshul"
}
```

Reconnect and confirm that Terraform proposes no changes:

```bash
terraform init -backend-config=backend.hcl
terraform plan \
  -var-file=tenant.tfvars \
  -var-file=environment.tfvars
```

## Finish the deployment

Use a synthetic serving bundle. Do not upload local H&M competition data:

```bash
cd ../..
BUNDLE_DIR="$(mktemp -d /tmp/recsys-loom-bundle.XXXXXX)"
python scripts/create_interview_lab_bundle.py "$BUNDLE_DIR" \
  --archive /tmp/recsys-loom-synthetic-lab-v1.tar.gz
aws s3 cp /tmp/recsys-loom-synthetic-lab-v1.tar.gz \
  s3://recsys-loom-bundles-576556099107/serving/synthetic-lab-v1.tar.gz \
  --profile recsys-loom
```

Reopen PR #1, wait for green CI, and merge it into `main`. Keep `main`
unprotected during this first merge so the unconfigured backend deployment job
is skipped. Then:

1. Import the repository into Vercel.
2. Set the Vercel project root to `apps/web`.
3. Set `NEXT_PUBLIC_API_URL=https://recsys-api.arbityr.live`.
4. Record the exact HTTPS Vercel production origin.
5. Create a GitHub `production` environment.
6. Configure the repository variables below.
7. Protect `main`.
8. Manually dispatch `.github/workflows/deploy-backend.yml`.

Required GitHub repository variables:

```text
AWS_REGION=ap-south-1
AWS_DEPLOY_ROLE_ARN=arn:aws:iam::576556099107:role/recsys-loom-github-ci
ECR_RECOMMENDATION_REPOSITORY=recsys-loom-recommendation
ECR_SEARCH_REPOSITORY=recsys-loom-search
DEPLOYMENT_CONFIG_BUCKET=recsys-loom-bundles-576556099107
DEPLOYMENT_CONFIG_PREFIX=ops
ALLOWED_ORIGINS=<exact HTTPS Vercel production origin>
BACKEND_BASE_URL=https://recsys-api.arbityr.live
SSM_INSTANCE_ID=i-078ca37b5f05e46d7
SERVING_BUNDLE_URI=s3://recsys-loom-bundles-576556099107/serving/synthetic-lab-v1.tar.gz
SERVING_BUNDLE_VERSION=synthetic-lab-v1
```

Successful backend deployment must make both endpoints ready:

```text
https://recsys-api.arbityr.live/health/recommendation
https://recsys-api.arbityr.live/health/search
```

The production-shaped lab intentionally uses synthetic data and disables
MiniLM, CLIP, and LLM search. The public VM must never receive `.env`, raw H&M
transactions, customers, training caches, or private API keys.
