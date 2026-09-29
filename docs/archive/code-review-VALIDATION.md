# Release validation

## Local checks

- All 684 Python files parse with Python 3.10 grammar under Python 3.12.14.
- All 9 shell files pass `bash -n`; all JSON and JSONL records parse, and the
  reference TSV is checked with a TSV parser.
- Third-party license file hashes and local notice links are checked.
- Source provenance resolves the package root and fingerprints released files
  without requiring or inspecting a Git repository.
- The evaluation launcher is byte-identical to the version that passed 13
  isolated mock tests: paths containing spaces and quotes, inherited W&B
  settings, worker failure propagation, and missing/invalid/incomplete/valid
  result artifacts. Those tests performed no inference.
- Eight further isolated test groups cover the critic's actual HTTP request
  shape, strict schema validation, missing model aliases, the two-image smoke,
  authentication versus connectivity failures, portable judge deployment,
  local evaluation-data verification, and data-converter input/output guards.
  All transports, model dependencies and serving processes in these tests
  were mocked.
- Private-path, credential, internal-history and archive-content scans found
  no remaining identified author-specific values. Public third-party
  attribution, reproducibility seeds, system paths and required sandbox paths
  are retained. The archive has normalized metadata and a per-file manifest.

## Execution changes

Release paths and dataset-root defaults were corrected. Supplied entrypoints
force W&B off. Missing images raise errors; unavailable matched-teacher modes
and optional ATS are rejected explicitly. The evaluation launcher fails when
its required result artifact is absent or invalid.

The actual critic request explicitly disables model thinking and sends the
configured JSON schema. Its prompt instructions, schema, image policy and
output budget were retained. One prompt's historical heading was shortened
during anonymization, so the release is not byte-identical to every
experiment-time prompt. The training loss and token-impact calculation were
not intentionally changed.

Data conversion now requires explicit input/output locations and rejects
missing splits or nonempty outputs. Dataset verification checks parsed rows,
image-path syntax and local image availability. Reference data manifests are
limited to the nine custom datasets in the main suite; retained values are
unchanged, apart from removing operation timestamps.

## Limits

The merged release has not been installed in a fresh Linux/CUDA environment.
No GPU training, model inference, live judge request, dataset download or
benchmark reproduction was run locally. A full scorer import could not be
completed in the local audit runtime because Torch is absent; isolated mock
imports do not establish that every framework import succeeds.

Checkpoints, HallusionBench recovery
inputs, exact serving builds and runtime dependency compatibility still need
resources and checks described in README and docs. No local check establishes
agreement with the paper's numerical results. See `docs/ENVIRONMENT.md` for
the distinction between server-reported packages and the release target.

`MANIFEST.sha256` lists every package file other than itself. Runtime outputs
can contain the user's paths, service information and sample content and need
separate review before redistribution.
