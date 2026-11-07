# Feature-only inference and deployment

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
$env:VIGIL_BUNDLE_SHA256 = '<approved-64-character-sha256>'
$env:VIGIL_API_TOKEN = '<random-secret-of-at-least-32-characters>'
$env:PORT = '8080'
python -m vigil.cloud
```

`GET /health` returns readiness and bundle digest; missing/invalid bundles or
missing authentication configuration return
503. `GET /model-info` exposes only digest, feature order, window/stride and
encoding to authenticated callers. `POST /predict` validates 60×8 raw finite values, binary masks, channel
order and digest. It returns canonical state, class ID, confidence, probability
mapping, server preprocessing/inference latency in ms and digest. Invalid JSON
or schema is 422, a changed bundle is 409, oversized bodies are 413 and inference
failure is 503. Errors never echo request bodies, paths or exceptions. Logs
contain generated request IDs and response status only. The server is stateless
between predictions; continuity/warm-up/stride are enforced on the edge.
`/model-info` and `/predict` require `Authorization: Bearer <VIGIL_API_TOKEN>`;
invalid/missing credentials get 401. A single worker limits authenticated
prediction requests to 60 per minute by default (configurable 1–600 via
`VIGIL_RATE_LIMIT_PER_MINUTE`); excess requests get 429. This is a per-process
limit, so use one worker or enforce a shared ingress limit when scaling out.
Use a generated, high-entropy token and rotate it in the hosting secret manager.

The client accepts an HTTPS base URL with no credentials/query, verifies TLS,
rejects redirects, defaults to a 3-second per-attempt timeout and one retry
(configurable 0–2 retries with 0.1/0.2-second backoff). It retries transport errors
and HTTP 429/502/503/504. Contract errors do not retry. Cloud failure stops the
demo clearly; no state or fallback is invented. Local inference is the default
and works offline. Set `VIGIL_API_TOKEN` in the desktop environment before
starting cloud mode. The client sends it only in the HTTPS Authorization header;
remote payloads contain no frame pixels.

```powershell
python -m vigil --webcam 0 --inference-mode cloud --cloud-endpoint https://YOUR-SERVICE.onrender.com --cloud-timeout 3 --cloud-retries 1 --landmark-model artifacts/vision/face_landmarker.task --session-id local-session --closure-threshold-ear APPROVED_EAR --perclos-window-frames APPROVED_WINDOW
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
docker run --rm -p 8080:8080 --mount type=bind,source=C:/outside-repo/approved-bundle.pt,target=/models/bundle.pt,readonly -e VIGIL_BUNDLE_PATH=/models/bundle.pt -e VIGIL_BUNDLE_SHA256 -e VIGIL_API_TOKEN vigil-cloud:local
```

In a second terminal:

```powershell
python scripts/cloud_smoke.py --url http://127.0.0.1:8080
```

For infrastructure testing only, create an ignored **untrained** bundle with
`python scripts/cloud_smoke.py --create-fixture artifacts/untrained-infrastructure.pt`.
Never use that fixture as trained-demo evidence. The smoke sends a synthetic
feature payload and checks response structure, not accuracy.

## Render paid Docker service (default deployment target)

The proposed [Render Blueprint configuration](../render.yaml) specifies one
paid 1 CPU/2 GB Docker web service, `/health` readiness, manual deploys, a
single process, and secret placeholders. Render provides HTTPS for the web
service. Creating the service is billable and requires operator authorization.
Use a restricted demo token; a public URL alone is not access control.

1. Obtain an M2-approved GRU bundle, its exact SHA-256, preprocessing settings,
   class mapping, and evaluation report. Upload the bundle to an immutable key
   in a private S3 bucket outside Git and Docker. Give the Render service a
   narrowly scoped `s3:GetObject` credential for that key; add KMS decrypt
   permission only if the object uses a customer-managed KMS key.
2. Connect this repository to Render and review `render.yaml` before creating
   the paid service. Supply all `sync: false` values in the Render Dashboard.
   Generate `VIGIL_API_TOKEN` with at least 32 random printable ASCII characters.
   Never paste tokens or AWS keys into a PR or terminal log. Set the S3 URI,
   digest, and AWS region; leave `VIGIL_BUNDLE_PATH` unset.
3. Deploy the reviewed branch manually. Confirm `/health` returns 200 with the
   expected digest. Run `python scripts/cloud_smoke.py --url https://<service>`
   with the token set locally. The smoke checks unauthenticated denial,
   authenticated model metadata and prediction schema. Its synthetic payload
   provides no model-quality evidence.
4. Restart the service from the Render Dashboard; repeat health and smoke
   checks. For rollback, redeploy the previous known-good image/commit and
   restore its matching immutable model object, digest and token settings.
   Verify health and prediction again. Record commit, bundle digest, timestamps,
   service URL, and results without logging credentials.

Keep automatic deployments off while the model and runtime are reviewed.
Render secret files are limited to 1 MB, so the model uses private object
storage rather than a secret file. The service must be treated as pending until
an authorized operator performs and verifies these steps.

## M2 model handoff contract

M2 supplies an approved GRU `.pt` bundle outside Git, its 64-character lowercase
SHA-256, class mapping, fitted normalizer, training/evaluation provenance, and
the exact EAR closure threshold, minimum blink duration, and PERCLOS window
size used to derive training features. The runtime sends 60 raw rows in
`LOCKED_FEATURE_CHANNELS` order with binary validity masks and stride 15 under
`raw_with_masks_v1`; the server normalizes once and returns `ACTIVE`, `DROWSY`,
or `SLEEPING` with confidence and all three probabilities. The bundle loader
rejects a different channel order; the HTTP client rejects a changed digest.
M2 should confirm that its recorded feature semantics and live extraction
settings match this contract before a trained-model demonstration.

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

The AWS example omits the required API token and therefore deliberately cannot
become healthy as written. Before deploying it, supply `VIGIL_API_TOKEN` through
an App Runner Secrets Manager reference and grant only the runtime role permission
to read that secret; do not put the token in CLI JSON or shell history.
Upload the approved bundle to the private S3 object outside Git before service
creation. SSE-KMS objects also require narrowly scoped decrypt permission. IAM
propagation can take time. Wait for `RUNNING` and verify both endpoints.
App Runner does not mount local files; use S3 environment configuration there.
`VIGIL_BUNDLE_PATH` takes precedence and is intended for local volume mounts.
`PORT` must match the service image configuration. Never pass AWS keys in the
Docker build or commit CLI files.

The API uses a shared bearer token and process-local inference limit. An
internet-facing deployment needs narrowly distributed credentials, rotation,
and ingress limits if scaled beyond one worker. No production availability or
driver-safety assurance is claimed. AWS deployment remains pending when
credentials or an approved bundle are unavailable; local Docker success does
not prove AWS health.
