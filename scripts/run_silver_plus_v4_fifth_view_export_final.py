from __future__ import annotations

"""Final Silver+ v4 runner with the handoff's cross-segment duplicate metric.

The v4 handoff defines global repeated 6-grams as six-gram types occurring in
more than one distinct segment. Repetition within a single segment is valid
spoken content and is not counted against the global duplicate budget.
"""

from collections import defaultdict
from typing import Any

import run_silver_plus_v4_fifth_view_export as v4


def repeated_6grams(
    rows: list[dict[str, Any]],
    eval_module: Any,
) -> int:
    """Count six-gram types repeated across distinct segments.

    Each six-gram contributes at most once per segment. The returned value is
    the number of unique six-gram types whose segment set has size greater
    than one. This preserves legitimate within-segment rhetorical repetition
    and matches the corrected baseline documented in the Silver+ v4 handoff.
    """

    gram_segments: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for position, row in enumerate(rows):
        segment_id = str(row.get("embedded_seg_id") or f"segment_{position:06d}")
        tokens = eval_module.toks(row["silver_plus_v4_text"])
        segment_grams = {
            tuple(tokens[index:index + 6])
            for index in range(max(0, len(tokens) - 5))
        }

        for gram in segment_grams:
            gram_segments[gram].add(segment_id)

    return sum(
        1
        for segment_ids in gram_segments.values()
        if len(segment_ids) > 1
    )


# Patch only the metric defect. All integration, honorific alignment, naming,
# validation schema, thresholds, lineage stamps, packaging, and stop behavior
# remain in the authoritative v4 implementation.
v4.repeated_6grams = repeated_6grams


if __name__ == "__main__":
    raise SystemExit(v4.main())
