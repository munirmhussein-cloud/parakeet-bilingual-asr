# Silver v3 Production Workflow

Silver v3 uses four independent hosted NVIDIA Parakeet inference views:

1. whole lecture;
2. canonical 20-second parents;
3. 10-second windows with a 5-second stride;
4. contiguous 2.5-second windows.

Production modules:

- `pipeline.silver_v3.prepare_views`
- `pipeline.silver_v3.run_hosted_view`
- `pipeline.silver_v3.normalize_views`
- `scripts/reconcile_silver_v3_lattice.py`

The workflow is deterministic and resumable. Hosted inference outputs are
skipped when valid files already exist unless `--force` is supplied.

## Lecture 001 benchmark

A100 runtime:

- multiview audio preparation: 46.417 seconds;
- whole view: 39.169 seconds;
- canonical view: 14.518 seconds;
- context view: 38.666 seconds;
- local view: 42.909 seconds;
- normalization: 5.065 seconds;
- first reconciliation attempt: 6.495 seconds.

All 1,786 hosted Parakeet calls completed with zero failures and zero retries.

The first reconciliation produced 138 canonical records and accounted for all
evidence observations. Its report failed one validation gate:

- `immediate_duplicate_6gram_count = 1`

A separate surface-token diagnostic found zero exact adjacent duplicated
six-grams. This discrepancy must be resolved in the validator before Silver v3
is marked fully passed.

The final partial canonical segment, 2740.0 to 2744.23075 seconds, was empty.

## Final reconciliation validation

Lecture 001 exposed one overlapping-context duplication:

`for only one of his servants`

Both copies came from overlapping context windows with the same 
estimated center. The reconciler merges the duplicate evidence 
into the retained tokens, removes only the second lexical span, 
and preserves separator punctuation and the zero-drop invariant.

Final Lecture 001 validation:

- 138 canonical segments;
- 14,222 reconciled tokens;
- one overlap collapse;
- zero remaining immediate duplicate six-grams;
- zero chronology errors;
- zero unaccounted observations;
- zero-drop invariant passed;
- one explicit empty trailing segment from 2740.0 to 2744.23075 seconds.

Finalize a completed multiview lecture:

```bash
python scripts/run_silver_v3.py \
  --lecture-id lecture_001 \
  --lecture-root /path/to/lectures/lecture_001
```

Export a portable package:

```bash
python scripts/export_silver_v3_package.py \
  --lecture-id lecture_001 \
  --lecture-root /path/to/lectures/lecture_001 \
  --output-root /path/to/exports/silver_v3
```
