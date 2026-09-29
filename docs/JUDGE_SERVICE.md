# Judge service

Training uses a multimodal critic (`gcep/judge_client.py`) and a reward judge (`agent_rm.py`).  
Some evaluation scorers also need a semantic judge. These roles can share an OpenAI-compatible  
endpoint; the supplement reports using Qwen3.5-397B-A17B-FP8 under the alias  
`qwen3.5-397b-judge`. Credentials, weights and a deployed service are not included.

## Deployment

Use a separate Linux serving environment that registers `Qwen3_5MoeForConditionalGeneration`.  
The training environment's vLLM 0.9.2 is insufficient. The supplement refers to more than one  
serving build and does not establish a complete, consistent serving lockfile. The launcher  
checks architecture registration with the same Python interpreter it uses to serve; this check  
alone does not certify compatibility of all serving flags or successful model startup.

```bash
export MODEL="<local-Qwen3.5-397B-A17B-FP8-directory>"
export API_KEY="<private-key>"
export JUDGE_PYTHON="<serving-environment>/bin/python"
bash thyme-infer/scripts/launch_judge_qwen35_397b_vllm.sh
```

The default settings are tensor parallelism 8, GPU memory utilization 0.92, maximum context  
24576 tokens, maximum sequences 32, at most 8 images and no videos, and multimodal processor  
cache size 0. These follow the supplied settings; full startup was not tested during this  
merge. `HOST`, `PORT`, `LOG_DIR` and the settings named in the launcher can be overridden.  
`RUN_IN_BACKGROUND=1` requires Linux `setsid`; a PID file indicates process launch, not readiness.  
No automatic process-killing watchdog is included.

## Client configuration

From the package root, configure the common endpoint before activation:

```bash
export JUDGE_BASE_URL="http://<judge-host>:8000/v1"
export JUDGE_API_KEY="<private-key>"
export JUDGE_MODEL="qwen3.5-397b-judge"
# The reward interface additionally consumes host and port separately:
export REWARD_API_ADDRESS="<judge-host>"
export QWEN_API_PORT="8000"
source thyme-infer/activate_thyme.sh
```

Activation maps common judge settings to the critic and evaluation aliases. Explicit subsystem  
settings take precedence; check that they point to the intended endpoint.

| Variables                                                                   | Consumer                                       |
| --------------------------------------------------------------------------- | ---------------------------------------------- |
| `GCEP_JUDGE_BASE_URL`, `GCEP_JUDGE_API_KEY`, `GCEP_JUDGE_MODEL`             | Training critic                                |
| `REWARD_API_ADDRESS`, `QWEN_API_PORT`, `REWARD_API_KEY`, `JUDGE_MODEL_NAME` | Training reward judge                          |
| `REMOTE_VLM_BASE_URL`, `REMOTE_VLM_API_KEY`, `REMOTE_VLM_MODEL`             | Evaluation semantic fallback                   |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`                            | Shared activation aliases and evaluation suite |

## Request and failure contract

The release critic explicitly sends `chat_template_kwargs={"enable_thinking": false}` and  
`response_format={"type": "json_schema", "json_schema": JUDGE_JSON_SCHEMA}`. With the Python  
client these fields go through `extra_body`, which is merged into the HTTP body. In raw HTTP  
they must be top-level fields; an HTTP body containing a nested `extra_body` is not equivalent.  
The multimodal smoke and text-only probe use the same no-thinking setting. The reward judge  
uses `REWARD_JUDGE_ENABLE_THINKING=0` by default; evaluation's custom text judge disables thinking.

The critic schema requires `applicable`, `sufficient_states`, `reading`, `grounding` and  
`rollout_diagnoses`. Diagnoses use CORRECT, ACQUIRE, READ, GROUND or OTHER. One critic request  
covers a rollout group. The existing critic output budget remains 2048 tokens; the evaluation  
text judge starts at 512 and can increase its budget on parse retries. These are different  
roles, so this merge does not silently replace the training budget with 512.

The critic caps images at `GCEP_JUDGE_MAX_IMAGES` (default 8), prioritizes correct rollouts,  
and retains a text marker when dropping overflow images. Its image resize limit is  
`GCEP_JUDGE_MAX_IMAGE_DIM` (default 1024 pixels on the longest side). Setting it to 0 preserves  
resolution and may exceed the service context budget. The input allowance is the context size  
minus requested output tokens; with defaults it is 22528 tokens before model-specific overhead.

Transport failures, invalid JSON and invalid critic outputs lead to clean-teacher OPD fallback  
for the affected group. A training job that continues running does not establish that critic  
reflection is working. Inspect fallback telemetry and successful structured responses. The  
supplement does not supply a complete breakdown of fallback causes for all paper runs.

## Endpoint checks

Run these only against the endpoint you intend to use:

```bash
python tools/probe_judge.py --out /tmp/judge_probe.json
python thyme-infer/scripts/judge_multiimage_json_smoke.py \
  --base-url "$JUDGE_BASE_URL" --api-key "$JUDGE_API_KEY" --model "$JUDGE_MODEL"
```

The text-only probe checks alias availability and validates the complete critic JSON schema  
shape, loaded from the shipped client. It sends no images and does not validate semantic  
correctness. The two-image smoke checks its own smaller schema, image count, and an HTTP 401/403  
rejection for an incorrect key. Neither replaces a full training/evaluation smoke test.

The supplement's reported live probe changed both the thinking switch and JSON-schema field  
placement between requests. That result supports using the corrected combined request, but is  
not a controlled test isolating the effect of the thinking switch. No live endpoint was called  
while integrating this release.
