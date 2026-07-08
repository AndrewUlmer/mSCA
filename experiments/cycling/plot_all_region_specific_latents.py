"""
Plot all latents per (variant, lam_region) cell, colored by trial condition.

Per-cell version (like plot_all_latents.py): walks
<results-root>/<dataset>/<variant>/lam_region=*/, loads each msca.pt, runs
transform on the full cycling data, and saves one figure per cell with a subplot
per latent component. In each subplot every trial of every region is overlaid,
encoded as:

    * hue       <- brain region:       M1 = RED, SMA = GREEN
    * shade     <- cycling direction:  forward = darker, backward = lighter
                   (i.e. different shades of the region's color per direction)
    * linestyle <- start_pos:          solid for one value, dotted for the other

The per-trial `direction` and `position` (start_pos) labels are re-read from the
region's .mat mask (the cycling loader discards them). The numeric value ->
encoding mappings are dataset-specific, so they're exposed as editable constants
below (cycling.py used direction in {1, -1} and position in {0.0, 0.5}).

CYCLING DATASETS ONLY (needs the .mat condition mask).

Usage (from this directory):
    python plot_all_region_specific_latents.py --dataset cousteau_cycling_40D
    python plot_all_region_specific_latents.py --dataset drake_cycling_40D \\
        --variants group_sparse_L2 --lam-regions 0,1,10 --overwrite
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save-to-file only; no GUI backend (debugger/headless safe).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[2]
for _p in (str(_PROJECT_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_all_latents import (  # noqa: E402
    build_X_cached,
    load_model,
    discover_variants,
    discover_lams,
)
from common import load_data, _CYCLING_DATA_DIR  # noqa: E402

_DEFAULT_RESULTS_ROOT = _HERE / "results"

# --- Condition -> visual encoding (edit to match your .mat's values) ---------
# Direction value treated as "forwards" (RED); the other value -> backwards
# (GREEN). cycling.py used direction in {1, -1}.
FORWARD_VALUE = 1
# start_pos (position) value drawn SOLID; the other value -> dotted. The user's
# convention is start_pos=-1 -> solid, +1 -> dotted. cycling.py used {0.0, 0.5},
# in which case the lower value is drawn solid (see _build_encoders fallback).
SOLID_STARTPOS = -1

# Base hue per region (saturated for high inter-region contrast). Regions not
# listed fall back to the _FALLBACK_HUES cycle (by region rank).
_REGION_HUE = {
    "M1": np.array([0.90, 0.0, 0.0]),  # red
    "SMA": np.array([0.0, 0.65, 0.0]),  # green
}
_FALLBACK_HUES = [
    np.array([0.0, 0.0, 0.90]),  # blue
    np.array([0.60, 0.40, 0.0]),  # amber
    np.array([0.55, 0.0, 0.70]),  # purple
]

# Direction shade: forward darkened toward black, backward lightened toward
# white. Bigger values -> more separation between forward/backward.
_FORWARD_DARKEN = 0.45  # forward: brightness *= (1 - this)
_BACKWARD_LIGHTEN = 0.55  # backward: blend toward white by this

# Line styling (thicker + less transparent for visibility).
_LINE_ALPHA = 0.75
_LINE_WIDTH = 1.1


def _region_hue(region: str, rank: int) -> np.ndarray:
    """Base hue for a region (named map first, then a fallback cycle by rank)."""
    if region in _REGION_HUE:
        return _REGION_HUE[region]
    return _FALLBACK_HUES[rank % len(_FALLBACK_HUES)]


def _shade_by_direction(rgb: np.ndarray, is_forward: bool) -> tuple:
    """Forward -> darker (toward black); backward -> lighter (toward white)."""
    rgb = np.asarray(rgb, dtype=float)
    if is_forward:
        return tuple(rgb * (1.0 - _FORWARD_DARKEN))
    return tuple(rgb + (1.0 - rgb) * _BACKWARD_LIGHTEN)


def load_conditions(dataset_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-read per-trial (direction, position) from the cycling .mat mask. Uses the
    first region's file (conditions are shared across regions) and returns arrays
    aligned to the trial order the loader produces.
    """
    if dataset_spec.get("loader") != "cycling":
        raise ValueError(
            "plot_all_region_specific_latents only supports cycling datasets; "
            f"got loader={dataset_spec.get('loader')!r}."
        )
    files = dataset_spec["files"]
    _region0, fname = next(iter(files.items()))

    data_dir = _CYCLING_DATA_DIR
    if "data_dir" in dataset_spec:
        p = Path(dataset_spec["data_dir"])
        data_dir = p if p.is_absolute() else (_HERE / p).resolve()

    # load_data -> (data, mask, start_idxs, end_idxs, num_cycles, direction, position)
    _d, _m, _si, _ei, _nc, direction, position = load_data(f"{data_dir}/", fname)
    return np.asarray(direction).squeeze(), np.asarray(position).squeeze()


def _build_encoders(direction: np.ndarray, position: np.ndarray):
    """
    Build per-trial encoders from the values actually present:
      hue_key(d)   -> "forward"/"backward"
      linestyle(p) -> "-"/":"
    Falls back to value ordering if the configured constants aren't present
    (forward = larger direction value; solid = smaller start_pos value).
    """
    dir_vals = sorted({float(x) for x in np.ravel(direction)})
    fwd = float(FORWARD_VALUE) if float(FORWARD_VALUE) in dir_vals else max(dir_vals)

    pos_vals = sorted({float(x) for x in np.ravel(position)})
    solid = (
        float(SOLID_STARTPOS) if float(SOLID_STARTPOS) in pos_vals else min(pos_vals)
    )

    def hue_key(d):
        return "forward" if float(d) == fwd else "backward"

    def linestyle(p):
        return "-" if float(p) == solid else ":"

    return hue_key, linestyle, fwd, solid


def plot_cell(latents: dict, direction, position, out_path: Path, title: str = ""):
    regions = list(latents.keys())
    K = latents[regions[0]][0].shape[1]
    n_trials = len(latents[regions[0]])
    if len(direction) != n_trials or len(position) != n_trials:
        raise ValueError(
            f"condition length mismatch: {len(direction)} direction / "
            f"{len(position)} position vs {n_trials} trials"
        )

    hue_key, linestyle, fwd, solid = _build_encoders(direction, position)

    # Region -> base hue (M1 red, SMA green); direction sets the shade.
    region_hue = {r: _region_hue(r, i) for i, r in enumerate(regions)}

    n_cols = 10
    n_rows = max(1, int(np.ceil(K / n_cols)))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2 * n_cols, 1.2 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for k in range(K):
        ax = axes_flat[k]
        for r in regions:
            for t, trial in enumerate(latents[r]):
                is_fwd = hue_key(direction[t]) == "forward"
                color = _shade_by_direction(region_hue[r], is_fwd)
                ax.plot(
                    trial[:, k],
                    color=color,
                    ls=linestyle(position[t]),
                    alpha=_LINE_ALPHA - 0.1,
                    lw=_LINE_WIDTH,
                )
        ax.set_title(f"comp {k}", fontsize=7)
        ax.tick_params(labelsize=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for k in range(K, len(axes_flat)):  # hide unused panels
        axes_flat[k].set_visible(False)

    # Legend: region hue, direction shade, start_pos linestyle.
    handles = [Line2D([0], [0], color=region_hue[r], lw=3, label=r) for r in regions]
    _ref = region_hue[regions[0]]  # reference hue for the direction-shade swatches
    handles += [
        Line2D(
            [0],
            [0],
            color=_shade_by_direction(_ref, True),
            lw=2,
            label="forward (darker)",
        ),
        Line2D(
            [0],
            [0],
            color=_shade_by_direction(_ref, False),
            lw=2,
            label="backward (lighter)",
        ),
    ]
    handles += [
        Line2D([0], [0], color="k", ls="-", lw=2, label=f"start_pos {solid:g} (solid)"),
        Line2D([0], [0], color="k", ls=":", lw=2, label="other start_pos (dotted)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="cousteau_cycling_40D",
        help="Cycling results-folder name (default: cousteau_cycling_40D).",
    )
    p.add_argument(
        "--variants",
        default=None,
        help="Comma-separated variants (default: all found under dataset).",
    )
    p.add_argument(
        "--lam-regions",
        default=None,
        help="Comma-separated lam values (default: auto-discover).",
    )
    p.add_argument(
        "--results-root",
        default=None,
        help="Root holding <dataset>/<variant>/... (default: local results/).",
    )
    p.add_argument(
        "--out-name",
        default="region_specific_latents.png",
        help="Filename saved into each cell dir.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-plot even if the output already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_root = (
        Path(args.results_root) if args.results_root else _DEFAULT_RESULTS_ROOT
    )
    dataset_dir = results_root / args.dataset
    variants = (
        [v for v in args.variants.split(",")]
        if args.variants
        else discover_variants(dataset_dir)
    )
    lam_filter = (
        {float(x) for x in args.lam_regions.split(",")} if args.lam_regions else None
    )

    cells: list[Path] = []
    for variant in variants:
        vdir = dataset_dir / variant
        for lam in discover_lams(vdir):
            if lam_filter is not None and lam not in lam_filter:
                continue
            cells.append(vdir / f"lam_region={lam:g}")
    if not cells:
        raise SystemExit(f"No (variant, lam_region) cells found under {dataset_dir}")

    x_cache: dict = {}
    cond_cache: dict = {}
    for run_dir in tqdm(cells, desc="cells"):
        out_path = run_dir / args.out_name
        cfg_path = run_dir / "config.json"
        if not (cfg_path.exists() and (run_dir / "msca.pt").exists()):
            tqdm.write(f"[skip] missing config.json/msca.pt: {run_dir}")
            continue
        if out_path.exists() and not args.overwrite:
            tqdm.write(f"[skip] exists (use --overwrite): {out_path}")
            continue

        cfg = json.loads(cfg_path.read_text())
        spec = cfg["dataset_spec"]
        spec_key = json.dumps(spec, sort_keys=True)
        try:
            X = build_X_cached(spec, x_cache)
            if spec_key not in cond_cache:
                cond_cache[spec_key] = load_conditions(spec)
            direction, position = cond_cache[spec_key]

            msca = load_model(run_dir, cfg, X)
            latents = msca.transform(X)
        except Exception as e:  # don't let one bad cell kill the whole sweep
            tqdm.write(f"[error] {run_dir}: {e}")
            continue

        title = (
            f"{args.dataset} / {cfg.get('variant')} / lam_region={cfg['lam_region']:g}"
        )
        plot_cell(latents, direction, position, out_path, title=title)
        tqdm.write(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
