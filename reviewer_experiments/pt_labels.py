"""Validate supported PT class encodings against filename-derived metadata."""
from __future__ import annotations

import math


def stored_label_value(value):
    try:
        scalar = value.item() if hasattr(value, "item") else value
        number = float(scalar)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"PT stored label must be a finite integer scalar: {value!r}") from exc


def audit_pt_labels(frame):
    """Require one consistent encoding over all input bundles; never infer binary labels by >0."""
    if frame.empty:
        raise ValueError("No PT observations to audit")
    canonical = [0 if row.treatment == "mock" else {2: 1, 4: 2, 6: 3}[int(row.dpi)]
                 for row in frame.itertuples(index=False)]
    schemes = {
        "mock0_infected123": canonical,
        "infected012_mock3": [(value - 1) % 4 for value in canonical],
    }
    observed = frame.groupby(["treatment", "dpi", "stored_label"]).size().rename("observations").reset_index()
    matches = [name for name, expected in schemes.items()
               if frame.stored_label.tolist() == expected]
    if len(matches) != 1:
        raise ValueError(
            "PT stored label encoding disagrees with filenames across the supplied bundles. "
            "Supported: mock=0, infected 2/4/6 dpi=1/2/3; or mock=3, infected 2/4/6 dpi=0/1/2. "
            "Mixed or unknown encodings are rejected. Observed treatment/dpi/label counts:\n"
            + observed.to_string(index=False)
        )
    encoding = matches[0]
    frame["pt_label_encoding"] = encoding
    return dict(encoding=encoding, observed_counts=observed.to_dict("records"),
                label_source="Filename treatment defines binary labels: mock=0, infected=1. Stored PT labels are retained and validated, not used as binary labels.")
