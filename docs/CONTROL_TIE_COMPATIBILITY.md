# Control choice compatibility before first execution

The control runner has been aligned with the already-frozen Jev adapter before any control request was executed. `results/review_controls/execution/` was absent both before and after this change. No inference request, credential lookup, ledger initialization or original prediction edit was performed.

## Exact change

The old collector-side check required the first index attaining the maximum probability. The frozen Jev adapter in `src/jevbench/providers.py` instead accepts its returned class when `p[label] >= max(p) - 1e-6`. `run_review_controls.validate_prediction` now uses that same choice check. An exact tied maximum or a selected probability within that existing absolute tolerance is accepted with its original class and vector; a selected probability farther below the maximum is rejected.

This fixes audit compatibility; it does not infer that rounding caused any response, relabel an accepted result, or change classification metrics. Shape, class-range, finite-value, probability-range and sum-to-one checks are unchanged. No normalization tolerance was relaxed. Requests, model, prompt, route, pricing, reservation/settlement rules, retry/failure policy and historical helpers are unchanged. The frozen adapter itself is unchanged.

## Runner identity

- Before SHA-256: `dea3045a68a2fa968df4be149aa94af28217ec097a6be81a6b44b2dd5a4d0e72`.
- After SHA-256: `d048bb4f17a5fa023ed54ee81d7152da1dc4b55814f9d343ef8e27f802afd455`.
- Replacing only the new choice-check block with the original block reproduces the exact before hash. There are no other production-code changes in this runner.
- The existing completion wrapper dynamically pins this new control producer at first execution. Its own source is unchanged; no historical frozen producer is repinned by this document.

## Validation

The focused tests accept a non-first exact tied maximum and an adapter-valid near-maximum choice, verify that validation does not mutate either response, and re-audit those saved choices through the normal ledger/prediction path. A gap larger than 1e-6 is rejected. Existing failed-response, nonfinite value, identity, accounting, orphan, resume and preparation tests also pass.

```sh
.venv/bin/python -m pytest -q \
  tests/test_review_controls_runner.py \
  tests/test_review_controls_summary.py \
  tests/test_review_controls_preparation.py \
  tests/test_restore_review_controls_payload.py \
  tests/test_run_control_failed_retries.py \
  tests/test_control_recovery_summary.py
```

**91 tests passed.** No paid requests were made. The real pending-control report test prevents network access and verifies that the frozen preparation files and absent execution directory remain unchanged.

## Unchanged protected evidence

The frozen inference-core aggregate `environment()["source_sha256"]` remains `d5547ccde4224e182653315d81c0a631c789bbe294ad7bcf95ef0cddd25ce608`. Every file below had the same SHA-256 before and after this change. Paths are relative to the repository root.

| Protected file | Unchanged SHA-256 |
|---|---|
| `configs/jev_openrouter.json` | `cce5b87496d878fc09b1b921bd549277c3f8fa23ecb94d44bf1f7a335569bb74` |
| `docs/REVIEW_CONTROLS_PROTOCOL.md` | `293ff0903d9a69f53e7098c32d12ed961b9bbce400f18ec5031b5b945bc893a0` |
| `results/review_controls/prepared_full/manifest.json` | `786c68ac439b400310ef4a2fcc77369bb8508e981d6d7372a6395a9d814f8ef4` |
| `results/review_controls/prepared_full/protocol.json` | `c1c94d336ad8700c1bb39b306069ca426ac22e392f588d5f967aeb03d2783992` |
| `results/review_controls/prepared_full/requests.jsonl` | `ae24a2a13a952bddac30f591a5d9b34559a96119c355e73e26e3c5fb5a327d60` |
| `scripts/prepare_review_controls.py` | `da709c914194c22839f93f37b3b2d051b7eb2e4ba4948cfa5113218e8bc85980` |
| `scripts/run_budgeted_hosted.py` | `3a6447218cba5025fef3aa58eb56d9045bd89173a35c4150544a87e7a55b6f92` |
| `scripts/run_expanded_numeric_review.py` | `68a16db205cceb81b46916630c776827ccdb09e7df6aed60dd1db7413d4d7a88` |
| `scripts/run_jev_completion.py` | `5771ab6081179bb9d8bf00411bc3a6f2ff546b3f80ca09597b1e2d0ccb7a7173` |
| `scripts/run_openrouter_jev.py` | `12a4e66649a2f9533ba8bb16fd7bd386a0c9dfc808f400b4da2a6df26e5ce89f` |
| `scripts/run_text_jev_review.py` | `bdc3d1cac8e52791359023b57abac8833d898af6ff7b0240512e337baa3ba43e` |
| `src/jevbench/__init__.py` | `ba19fd98083ce60c00281ca4dbfc8a1596481da6cfd1d18a138378637f75bfea` |
| `src/jevbench/classical.py` | `8c40d49c2f100e09368e16480cc26fcb515f0a76b0e1c4723b2c5799bf84d0bc` |
| `src/jevbench/cli.py` | `7d9c274055eae20e89b3dac4c22293a4bfb17247f546d627cce984a7febe7c62` |
| `src/jevbench/data.py` | `df1f41643d745a6a075598ea4c071b17770313a7a691189c949297e148035b21` |
| `src/jevbench/lora.py` | `e61d35c68fc98cbbd4f59b5421f80ed7133188104439812cf9dfe0792c4a1330` |
| `src/jevbench/metrics.py` | `551146253f47dedc9354e6e1b1ed81fdab57c07b0c77b8471a31eb4e2b6528c9` |
| `src/jevbench/prompts.py` | `1b687e697edf2065c4d59b171f03cf6d8b5f327d17c992a906dbae03b9a8a875` |
| `src/jevbench/providers.py` | `fbeb2156be56246639bd87da309931e0f62928617ab23a4a6b3da60353b1c149` |
| `src/jevbench/report.py` | `381c07b459dd06bb1b35be6af14aaea90e43b69f172421fe6d4434ca2f752f7e` |
| `src/jevbench/runner.py` | `ac655ec86ee261228efe8c682c880c43ab063853c8f6f40003007a14641b1ff6` |
| `src/jevbench/types.py` | `9334c2741d48f1cbda7fd9757ff7651d3b80fe87e61d7a7038452bbc5e770f96` |

The prepared request payload remains local and ignored by Git; only its hash is documented. These checks do not rewrite the prepared protocol, manifest, payload or any inference artifact.
