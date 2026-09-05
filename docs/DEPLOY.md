# Deploying dynavec

Three things: publish to PyPI (pip/uv), deploy the landing page to GitHub Pages,
and give the library AWS credentials with least-privilege IAM.

---

## 1. Publish to PyPI (pip + uv)

`pip` and `uv` both install from PyPI — publish once, both work.

### a. Register the Trusted Publisher on PyPI (one time, no token)

At <https://pypi.org/manage/account/publishing/>, add a **GitHub** publisher with
**exactly** these values (they must match the committed workflow):

| Field | Value |
|-------|-------|
| PyPI Project Name | `dynavec` |
| Owner | `codeforstartups` |
| Repository name | `dynavec` |
| **Workflow name** | **`publish.yml`**  ← not `workflow.yml` |
| **Environment name** | **`pypi`** |

> The two bold fields are the ones people miss. The committed workflow is
> `.github/workflows/publish.yml` and it declares `environment: pypi`, so both
> must match or PyPI will reject the OIDC token.

### b. Create the `pypi` environment on GitHub (one time)

Repo → **Settings → Environments → New environment → `pypi`**. (Optional but
recommended; add required reviewers here if you want a manual approval gate.)

### c. Release

Bump the version in **both** `pyproject.toml` and `src/dynavec/__init__.py`, then:

```bash
git tag v0.2.0
git push origin v0.2.0
```

Then create a **GitHub Release** for that tag (UI, or `gh release create v0.2.0
--generate-notes`). Publishing the release triggers `publish.yml`, which builds
and uploads via OIDC. Done — `pip install dynavec` / `uv add dynavec` now work.

### Manual fallback (from your machine, with a token)

```bash
uv build
UV_PUBLISH_TOKEN=pypi-XXXX uv publish
# or: python -m twine upload dist/*
```

Tip: test on TestPyPI first with `uv publish --publish-url https://test.pypi.org/legacy/`.

---

## 2. Deploy the landing page to GitHub Pages

The workflow `.github/workflows/pages.yml` publishes `opensource/dynavec/`.

1. Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `development` touching `opensource/dynavec/**` (or run the workflow
   manually from the Actions tab). The site goes live at
   `https://codeforstartups.github.io/dynavec/`.

To refresh the benchmark charts before deploying:

```bash
python -m benchmarks.report --out opensource/dynavec/images
```

---

## 3. AWS credentials (.env) + IAM permissions

### The `.env`

Copy `.env.example` → `.env` (already gitignored) and fill it in. dynavec uses
the standard boto3 credential chain, so exported env vars just work:

```bash
cp .env.example .env
# edit .env, then:
export $(grep -v '^#' .env | xargs)      # load into the shell
python examples/openai_1536_retrieval.py
```

Or load it in Python without exporting:

```python
from dotenv import load_dotenv          # pip install python-dotenv
load_dotenv()
from dynavec import Dynavec, DynavecConfig
db = Dynavec(DynavecConfig(..., region="us-east-1"))   # boto3 picks up the env creds
```

Or pass keys explicitly (e.g. multi-account) instead of the env chain:

```python
import os
from dynavec import Dynavec, DynavecConfig, AWSCredentials
creds = AWSCredentials(
    access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region="us-east-1",
    # assume_role_arn="arn:aws:iam::OTHER_ACCOUNT:role/dynavec",  # cross-account
)
db = Dynavec(cfg, credentials=creds)
```

### The IAM policy

Create an IAM user (or role) and attach `docs/iam-policy.json` — replace
`REGION` and `ACCOUNT_ID`. It grants only what dynavec uses:

- **s3vectors**: create/get/list/delete vector buckets & indexes; put/get/list/query/delete vectors
- **dynamodb**: create/describe/delete table; batch + single-item read/write; `UpdateItem` (knowledge graph); `Query` (future access patterns)

Then create the access key for that user and put it in `.env`.

> **Tighten for production:** the s3vectors statement uses `Resource: "*"` because
> S3 Vectors ARN formats are new — scope it to your bucket/index ARNs once
> confirmed in the AWS console. The DynamoDB statement is already scoped to
> `dynavec_*` tables; rename to match your table if different. Drop `DeleteTable`
> / `Delete*` if you never run the cleanup scripts.

### Optional add-ons (only if you use them)

Add these statements to the policy if the corresponding feature is used:

```jsonc
// BedrockEmbedder (embeddings stay in-account)
{ "Effect": "Allow", "Action": ["bedrock:InvokeModel"],
  "Resource": "arn:aws:bedrock:REGION::foundation-model/amazon.titan-embed-text-v2:0" }

// LambdaTransform (transform vectors/metadata in your own Lambda)
{ "Effect": "Allow", "Action": ["lambda:InvokeFunction"],
  "Resource": "arn:aws:lambda:REGION:ACCOUNT_ID:function:YOUR_FUNCTION" }

// DynamoDB TTL cache housekeeping (only if enabling TTL via code)
{ "Effect": "Allow", "Action": ["dynamodb:UpdateTimeToLive","dynamodb:DescribeTimeToLive"],
  "Resource": "arn:aws:dynamodb:REGION:ACCOUNT_ID:table/dynavec_*" }
```

### Never commit secrets

`.env` is in `.gitignore`. For CI, put keys in **GitHub Secrets**, not the repo.
For production apps, prefer an **IAM role** (instance/task role, or OIDC) over
long-lived access keys entirely.
