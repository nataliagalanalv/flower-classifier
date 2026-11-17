# Security

What security checks the project has, what each one is for, what they do **not** cover, and which risks are accepted on purpose. It describes a small personal project, not a formal audit or a regulatory compliance assessment.

The current figures (findings, passed checks, scan dates) are in the [README](../README.md#vulnerability-scanning). They are not repeated here so they cannot go out of date in two places at once.

## Summary of controls

| Control | What it checks | Where it runs | What enforces it |
| --- | --- | --- | --- |
| Image vulnerability scan | Known vulnerabilities (HIGH and CRITICAL) in the OS and Python packages of the built image | `security.yml`, job `trivy-image` | Trivy v0.76.1; the job fails with exit code 1 if there are findings (`--ignore-unfixed`, see [Accepted risks](#accepted-risks)) |
| Configuration scan | The `Dockerfile` and the workflow files against Checkov's built-in rules | `security.yml`, job `checkov` | Checkov 3.4.0, image pinned by digest; the job fails if a rule fails |
| Container hardening | Unprivileged user, no `pip` or `setuptools` in the final image, model kept out of the image, `HEALTHCHECK`, small build context, a targeted upgrade of a base-image package that has a fix | `Dockerfile`, `.dockerignore` | The unprivileged user and the `HEALTHCHECK` are checked by Checkov; the rest is a design decision and shows up in the Trivy results |
| Workflow hardening | Token limited to `contents: read`; actions pinned by commit hash | `tests.yml`, `security.yml` | Permissions are checked by Checkov (`CKV2_GHA_1`); pinning by hash is done by hand |
| Model integrity | The model file is checked against a SHA-256 hash before it is loaded, and loaded with `weights_only=True` | `app/model_store.py`, `app/main.py` | The application itself (unit tests in `tests/test_model_store.py`); the server refuses to start on a mismatch |
| AWS access control (manual proof of concept) | Private bucket, least-privilege IAM roles, one allowed IP address | The AWS account, see [AWS](#aws-manual-proof-of-concept) | AWS IAM and security groups; checked by hand in the console. Not covered by any scanner yet |

The two security jobs run on every pull request and on every push to `main`. The AWS controls are not part of the pipeline: they were built by hand and will be checked by Checkov once they exist as Terraform.

## Design principles

- **Make the build fail instead of only reporting.** A scanner that just prints a table ends up ignored.
- **Show that the gate can fail.** A green check says nothing until the same check has been seen red (see [Negative tests of the gates](#negative-tests-of-the-gates)).
- **Pin what runs in the pipeline.** Actions are pinned by commit hash, Trivy by version and Checkov by image digest. A tag can be moved to point to different code; a commit hash or a digest cannot. This is not theoretical: the versions and tags of Trivy's own actions were affected by a supply-chain incident in March 2026 [link to the advisory: to be added once verified]. Pinning limits what can change under the pipeline without a review, but it does not make a pinned version safe by itself.
- **Least privilege.** The workflow token is read-only. On AWS, the application's role can read one file.
- **Read what was scanned, not just the colour of the check.** The log is part of the review.
- **Fix what has a fix; accept only what has none.** A finding with an available patch is not covered by an accepted risk (see [Accepted risks](#accepted-risks)).

## Negative tests of the gates

| Gate | Negative test | Result |
| --- | --- | --- |
| Trivy | [PR #9](https://github.com/nataliagalanalv/flower-classifier/pull/9) (closed, never merged) pinned the vulnerable Pillow 12.3.0 | `security` failed with exit code 1 and the Pillow findings; `tests` passed |
| Checkov | [PR #13](https://github.com/nataliagalanalv/flower-classifier/pull/13) (closed, never merged) removed the `HEALTHCHECK` | The Dockerfile step failed with `CKV_DOCKER_2`; `tests` and `trivy-image` passed |

The screenshots are in `docs/images/`. Two of these tests did not work the first time, and both taught something:

- **The first Trivy test was not valid.** It used `pillow>=12.3.0`, which still resolves to Pillow 13.0.0, so nothing vulnerable was installed and the check stayed green. A negative test has to really create the condition it claims to test, so it was changed to `pillow==12.3.0`.
- **The first Checkov job only scanned half of its targets.** It passed `--file` and `--directory` in one command, and the log had a `github_actions scan results` block but no `dockerfile scan results` block. The Dockerfile was never checked, so removing the `HEALTHCHECK` stayed green. It was fixed in [PR #14](https://github.com/nataliagalanalv/flower-classifier/pull/14) with one step per target. The cause of the single command behaving that way is not known; the log is the only evidence.

The model-integrity control has its own negative tests in `tests/test_model_store.py`: a wrong hash raises an error and removes the downloaded file, and the server refuses to start in that case.

## AWS (manual proof of concept)

The application was deployed once, by hand, on ECS Fargate. The full description, with evidence and lessons, is in [`aws-manual-deployment.md`](./aws-manual-deployment.md). The security-relevant parts:

| Control | What it does | How it was checked |
| --- | --- | --- |
| Private model bucket | All public access blocked, SSE-S3 encryption, versioning on, in a EU region (`eu-south-2`) | Settings read in the console |
| Least-privilege task role | `s3:GetObject` on one object ARN. No listing, no writes, no other resources | The policy JSON in the console; the app loaded the model with it |
| Two separate roles | Execution role (pulls the image, sends logs) and task role (used by the app code). Trust policy of the task role limited with `aws:SourceAccount` and `aws:SourceArn` against the confused deputy problem | Reviewed in the console; checked against the AWS documentation while designing it |
| No credentials in the image or in the task definition | The app gets temporary credentials from the task role | The model loaded in ECS without any keys configured |
| Model integrity | SHA-256 check and `weights_only=True`, see [Summary of controls](#summary-of-controls) | Unit tests; real end-to-end run |
| Network exposure | A security group allowing TCP 8000 from one IPv4 address (`/32`) and nothing else | A request from the allowed address worked. A request from another network was **not** tested. A task launched with the wrong security group was unreachable (see the lessons in the AWS document) |
| Short log retention | Log group with 1 week retention, because the logs contain client IP addresses | Settings read in the console |
| Immutable image tags | ECR does not allow overwriting a tag | Setting read in the console |

A side effect worth knowing: because the task role cannot list the bucket, S3 will normally answer "access denied" instead of "not found" when the key is wrong. This is S3's behaviour as I understand it; it was not tested here.

Local access to AWS uses an IAM user signed in with `aws login`. Always check that the identity is the IAM user and not the root user before running commands.

## Accepted risks

| Risk | Why it is accepted | Review |
| --- | --- | --- |
| HIGH findings in Debian packages of the base image with no fixed version available | Nothing can be fixed in this repository until Debian publishes a fix. CI hides them with `--ignore-unfixed`; the local command in the README does not use it, so they stay visible. It has **not** been assessed whether each one is reachable from this application | Scan again every time the base image is rebuilt and when the base image tag moves to a newer version |
| Manual AWS test: default VPC, public IP on the task, HTTP only, access from a single home IP address | It was a short test, destroyed the same evening. Nothing like it should stay running | Replaced by Terraform with its own VPC, a load balancer and TLS (see [Next steps](#next-steps)) |
| Logs without a customer-managed KMS key | The logs hold request lines and client IP addresses, kept for 1 week, and nothing else sensitive was noticed. A customer-managed key adds cost and a key policy to maintain | Revisit if the logs start to hold anything more sensitive |
| The execution role uses the AWS managed policy `AmazonECSTaskExecutionRolePolicy` | It is the standard policy for that role and has no access to the model or to user data | Review its permissions when the role is written in Terraform |

An accepted risk is a decision, not a fix: it stays on this list until the findings disappear or are assessed one by one.

**What an accepted risk does not cover.** A finding that has a fixed version available is not covered by the first row. This happened once: Trivy failed in CI on a HIGH finding in `zlib1g` for which Debian had already published a fixed package. The base image tag had not yet picked it up. The fix was a targeted `apt-get install --only-upgrade zlib1g` step in the `Dockerfile`, which changes one package and is easy to review. A blanket `apt-get upgrade` would change an unpredictable set of packages on every build. The extra line becomes redundant once the base image ships the fixed package; it should be reviewed when the base image tag moves.

## Known gaps

Not done yet, in no particular order:

- **No scheduled scan.** Scans run on push and on pull request. The vulnerability database is updated daily, so a new CVE affecting an unchanged image is not detected until the next change. The `zlib1g` case above was found this way.
- **`main` is not protected.** Pull requests are a habit, not a rule: it is possible to push directly to `main` (and it has happened with documentation commits).
- **Dependencies are declared with `>=`.** Two builds on different days may install different versions. Pinning the versions or using a lock file is missing.
- **The base image is referenced by tag** (`python:3.14-slim`), not by digest. The tag keeps receiving rebuilds with security patches, which is useful, but builds are not reproducible.
- **`runs-on: ubuntu-latest` is not pinned**, so the runner image changes over time. Some actions are on old versions (the checkout step shows a Node 20 deprecation warning); upgrading them requires reading each version's release notes first.
- **No secret scanning in CI.** Trivy's secret scanner is disabled (`--scanners vuln`) because it timed out on the 2 GB image. Scanning the repository files instead of the image layers fits better and is still to be evaluated.
- **The application code is not analysed** (neither static nor dynamic analysis).
- **The API has no authentication and no rate limiting**, and the upload size limit has not been verified. `app/main.py` has to be reviewed before claiming anything about this.
- **The container is built and scanned in CI but never started there.** That it starts and becomes `healthy` was checked by hand with `docker run` and `docker ps`.
- **The AWS controls are not scanned or enforced by any tool.** They will be once they are written as Terraform and checked with Checkov.
- **IAM is not restricted to the region in use.** A condition on `aws:RequestedRegion` would limit mistakes and misuse to one region.
- **Local sign-in is an IAM user with `aws login`.** IAM Identity Center would be stricter and is a candidate.
- **No monitoring or alarms yet.** Planned for the CloudWatch stage.
- **The console's defaults created things that were not asked for** (a pre-selected `default` security group and an auto-created log group without retention), as described in the AWS document. The only reliable defence is to write infrastructure as code and review what exists.

## What scanners cannot tell you

- **Trivy** reports *known* vulnerabilities in installed packages. It does not find unknown ones, and a clean scan is a statement about its database on that day.
- **Checkov** applies the rules it ships with. For example, no rule about pinning actions by hash appeared in its output, so no tool verifies that control here.
- **Neither checks regulatory compliance**, GDPR included.

## Privacy (GDPR): partly answered, not yet written up

No scanner checks this and it is not written up yet. Some of the questions can now be answered from the code and from the AWS test; the rest are still open. This is engineering documentation, not legal advice.

**What is known:**

- **Uploaded images are not stored.** The `/predict` handler reads the upload into memory, decodes it and returns the predictions; no code in the handler writes the file anywhere. The rest of the prediction path (`predict_top_k`) has not been audited for side effects.
- **Client IP addresses are logged.** Uvicorn's access log line includes the client IP address, and in AWS the logs go to CloudWatch Logs in `eu-south-2` (Spain), with 1 week retention in the manual test. An IP address is personal data under the GDPR.

**Still open:**

- Can uploads carry personal data (for example, GPS coordinates in the EXIF metadata)? The handler does not use that metadata, but `predict_top_k` and the logs have not been checked for it.
- Is anything collected beyond what the prediction needs (data minimisation)? Should the access log keep the full client IP?
- Where will the logs and stored data live in the final deployment, and for how long?
- What the application tells users about all of this (a privacy notice).

The write-up is planned for the last weeks of the project.

## Next steps

Planned, in order of priority:

1. Protect `main` (require pull requests and green checks).
2. Add a scheduled run of the security workflow.
3. Pin the dependency versions and decide how to pin the base image.
4. Add a smoke test to CI that starts the container and waits for it to be `healthy`.
5. Write the AWS setup as Terraform, with its own VPC instead of the default one, the security group declared explicitly and a load balancer with TLS, and run Checkov on it (the same tool, a new target).
6. GitHub Actions authenticating to AWS through OIDC instead of long-lived keys, pushing images tagged with the commit hash.
7. CloudWatch monitoring and alarms; restrict IAM to the region in use.
8. Write up the privacy section above.

All AWS items are plans; each one has to be checked against the current AWS documentation when it is implemented.