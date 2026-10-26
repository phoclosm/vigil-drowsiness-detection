# Feature-only inference and AWS deployment

The service receives **only 60 raw derived feature rows with eight channels**,
their exact channel names, and a bundle SHA-256 identifier. It accepts no images,
video, labels or subject/session IDs. Raw frames remain on the edge machine.
Normalized values must not be sent: the server applies its loaded bundle's
normalizer exactly once. The API reuses the stable GRU bundle loader. Candidate
LSTM/TCN training does not imply that those architectures can be exported yet.

Install a server without OpenCV, MediaPipe or GUI dependencies:

```powershell
python -m pip install -e ".[cloud,ml,dev]"
$env:VIGIL_BUNDLE_PATH = 'C:/outside-repo/approved-bundle.pt'
$env:PORT = '8080'
python -m vigil.cloud
```

`GET /health` returns readiness and bundle digest; missing/invalid bundles return
503. `GET /model-info` exposes only digest, feature order, window/stride and
encoding. `POST /predict` validates 60×8 raw finite values, binary masks, channel
order and digest. It returns canonical state, class ID, confidence, probability
mapping, server preprocessing/inference latency in ms and digest. Invalid JSON
or schema is 422, a changed bundle is 409, oversized bodies are 413 and inference
failure is 503. Errors never echo request bodies, paths or exceptions. Logs
contain generated request IDs and response status only. The server is stateless
between requests; continuity/warm-up/stride are enforced on the edge.

The client accepts an HTTPS base URL with no credentials/query, verifies TLS,
rejects redirects, defaults to a 3-second per-attempt timeout and one retry
(configurable 0–2 retries with 0.1/0.2-second backoff). It retries transport errors
and HTTP 429/502/503/504. Contract errors do not retry. Cloud failure stops the
demo clearly; no state or fallback is invented. Local inference is the default
and works offline. Remote payloads contain no frame pixels.

```powershell
python -m vigil --webcam 0 --inference-mode cloud --cloud-endpoint https://YOUR-SERVICE.awsapprunner.com --cloud-timeout 3 --cloud-retries 1 --landmark-model artifacts/vision/face_landmarker.task --session-id local-session --closure-threshold-ear APPROVED_EAR --perclos-window-frames APPROVED_WINDOW
```

Use bundle-approved EAR/blink/PERCLOS settings. Obtain those settings and model
quality evidence from the bundle owner. For local mode add
`--inference-mode local --model PATH` instead of cloud flags.

## Docker

The allowlisted build context includes only source, package metadata and README.
Models/secrets never enter image layers. The image runs as UID 10001, uses CPU
PyTorch, port 8080 by default and starts one Uvicorn worker. Docker's healthcheck
uses `/health`; App Runner must configure its own HTTP healthcheck on that path.

```powershell
docker build -t vigil-cloud:local .
docker run --rm -p 8080:8080 --mount type=bind,source=C:/outside-repo/approved-bundle.pt,target=/models/bundle.pt,readonly -e VIGIL_BUNDLE_PATH=/models/bundle.pt vigil-cloud:local
```

In a second terminal:

```powershell
python scripts/cloud_smoke.py --url http://127.0.0.1:8080
```

For infrastructure testing only, create an ignored **untrained** bundle with
`python scripts/cloud_smoke.py --create-fixture artifacts/untrained-infrastructure.pt`.
Never use that fixture as trained-demo evidence. The smoke sends a synthetic
feature payload and checks response structure, not accuracy.

## AWS ECR and App Runner

**Account eligibility gate:** AWS documents that App Runner is
[closed to new customers](https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html).
Existing customers may continue creating services. Confirm this account is an
existing App Runner customer before using the commands below. AWS recommends
ECS Express Mode for new customers; changing the hosting target requires a
maintainer decision. The local image/API are independent of that decision.

These commands create billable resources. Configure AWS CLI/IAM outside Git.
Replace all account, bucket, key and region values. The image role can pull ECR;
the instance role can read only the approved private S3 bundle. Use an immutable
S3 object key and an owner-supplied SHA-256. Download uses finite connect/read
timeouts, bounded retries, a 128 MiB cap and mandatory digest verification.

```powershell
$region = 'ap-south-1'
$account = aws sts get-caller-identity --query Account --output text
$registry = "$account.dkr.ecr.$region.amazonaws.com"
$image = "$registry/vigil-cloud:v0.1-candidate"
$bundleBucket = 'YOUR-PRIVATE-BUCKET'
$bundleKey = 'approved/IMMUTABLE-BUNDLE.pt'
$bundleDigest = 'OWNER-SUPPLIED-64-LOWERCASE-HEX-SHA256'
aws ecr create-repository --repository-name vigil-cloud --region $region --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $registry
docker build --platform linux/amd64 -t vigil-cloud:local .
docker tag vigil-cloud:local $image
docker push $image
aws iam create-role --role-name VigilEcrAccess --assume-role-policy-document file://deploy/ecr-trust.json
aws iam attach-role-policy --role-name VigilEcrAccess --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
aws iam create-role --role-name VigilRuntime --assume-role-policy-document file://deploy/instance-trust.json
$s3Policy = @{Version='2012-10-17'; Statement=@(@{Effect='Allow'; Action=@('s3:GetObject'); Resource="arn:aws:s3:::$bundleBucket/$bundleKey"})} | ConvertTo-Json -Depth 8 -Compress
aws iam put-role-policy --role-name VigilRuntime --policy-name ReadApprovedBundle --policy-document $s3Policy
$source = @{ImageRepository=@{ImageIdentifier=$image; ImageRepositoryType='ECR'; ImageConfiguration=@{Port='8080'; RuntimeEnvironmentVariables=@{VIGIL_BUNDLE_S3_URI="s3://$bundleBucket/$bundleKey"; VIGIL_BUNDLE_SHA256=$bundleDigest; PORT='8080'}}}; AuthenticationConfiguration=@{AccessRoleArn="arn:aws:iam::${account}:role/VigilEcrAccess"}; AutoDeploymentsEnabled=$false} | ConvertTo-Json -Depth 10 -Compress
$instance = @{Cpu='1 vCPU'; Memory='2 GB'; InstanceRoleArn="arn:aws:iam::${account}:role/VigilRuntime"} | ConvertTo-Json -Compress
$health = @{Protocol='HTTP'; Path='/health'; Interval=10; Timeout=5; HealthyThreshold=1; UnhealthyThreshold=5} | ConvertTo-Json -Compress
$serviceArn = aws apprunner create-service --region $region --service-name vigil-inference --source-configuration $source --instance-configuration $instance --health-check-configuration $health --query Service.ServiceArn --output text
aws apprunner describe-service --region $region --service-arn $serviceArn --query Service.Status
$serviceHost = aws apprunner describe-service --region $region --service-arn $serviceArn --query Service.ServiceUrl --output text
Invoke-RestMethod "https://$serviceHost/health"
python scripts/cloud_smoke.py --url "https://$serviceHost"
```

Upload the approved bundle to that private S3 object outside Git before service
creation. SSE-KMS objects also require narrowly scoped decrypt permission. IAM
propagation can take time. Wait for `RUNNING` and verify both endpoints.
App Runner does not mount local files; use S3 environment configuration there.
`VIGIL_BUNDLE_PATH` takes precedence and is intended for local volume mounts.
`PORT` must match the service image configuration. Never pass AWS keys in the
Docker build or commit CLI files.

The prototype endpoint has no application authentication/rate limiting. Use a
restricted App Runner ingress/VPC configuration or add reviewed access controls
before exposing it beyond an approved demo. No production availability or driver
safety assurance is claimed. AWS deployment remains pending when credentials or
an approved bundle are unavailable; local Docker success does not prove AWS health.
