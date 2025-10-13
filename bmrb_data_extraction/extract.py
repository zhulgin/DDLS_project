#!/usr/bin/env python3
"""
Extract 1H chemical shifts from BMRB NMR-STAR files into simple TXT peaklists.

- For each .str file: writes <entry_id>.txt with one ppm value per line (sorted, high→low).
- Also writes an index CSV: nmr_txt_index.csv (entry_id, txt_path, n_peaks)

Usage:
  python export_1H_shifts.py \
      --in_dir "bmrb_nmrstar/*/*.str" \
      --out_dir "out_txt"

Requires:
  pip install pynmrstar pandas
"""

import os, sys, glob, argparse
from pathlib import Path
import pandas as pd
import pynmrstar

def get_first(v, default=None):
    """pynmrstar returns lists for get_tag(); this normalizes to a single scalar."""
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    return v if v is not None else default

def extract_1h_ppm(entry):
    """
    Return a list of 1H chemical shift ppm values from an Entry.
    We look into saveframes under category 'assigned_chemical_shifts' and loops of 'Atom_chem_shift'.
    Filters for 1H using either Atom_type == 'H' or Atom_isotope_number == 1 (if present).
    Ensures units are ppm when available.
    """
    ppm_values = []

    # some files put units at frame level; capture if present (fallback to 'ppm')
    frame_level_units = "ppm"

    for sf in entry.get_saveframes_by_category("assigned_chemical_shifts"):
        # frame-level units (optional)
        try:
            u = get_first(sf.get_tag("_Atom_chem_shift.Chem_shift_units"), None)
            if isinstance(u, str) and u.strip():
                frame_level_units = u.strip()
        except Exception:
            pass

        # one or more atom_chem_shift loops per frame
        loops = []
        try:
            loops = sf.get_loops_by_category("Atom_chem_shift")
        except Exception:
            # older files may have a single loop accessor
            try:
                loop = sf.get_loop_by_category("Atom_chem_shift")
                loops = [loop] if loop else []
            except Exception:
                loops = []

        for loop in loops:
            tags = set(loop.tags or [])
            # figure out which columns exist
            has_atom_type = "Atom_type" in tags or "_Atom_chem_shift.Atom_type" in tags
            has_isotope  = "Atom_isotope_number" in tags or "_Atom_chem_shift.Atom_isotope_number" in tags
            has_val      = "Val" in tags or "_Atom_chem_shift.Val" in tags
            has_units    = "Chem_shift_units" in tags or "_Atom_chem_shift.Chem_shift_units" in tags

            if not has_val:
                # can't do much without the shift value
                continue

            # build the list of columns to fetch in one call (works with either short or full tag names)
            wanted = []
            for col in ("Atom_type", "Atom_isotope_number", "Val", "Chem_shift_units"):
                if col in tags:
                    wanted.append(col)
                else:
                    full = f"_Atom_chem_shift.{col}"
                    if full in tags:
                        wanted.append(full)

            # pull rows
            try:
                rows = loop.get_tag(wanted)  # list of lists, order matches 'wanted'
            except Exception:
                continue

            # map for index lookup
            idx = {c: i for i, c in enumerate(wanted)}

            for r in rows:
                # value (ppm)
                val_col = "Val" if "Val" in idx else "_Atom_chem_shift.Val"
                try:
                    ppm = float(r[idx[val_col]])
                except Exception:
                    continue

                # filter to protons
                is_proton = False
                # Atom_type == 'H'
                at_col = "Atom_type" if "Atom_type" in idx else "_Atom_chem_shift.Atom_type" if "_Atom_chem_shift.Atom_type" in idx else None
                if at_col is not None:
                    atype = str(r[idx[at_col]]).strip() if r[idx[at_col]] is not None else ""
                    if atype.upper() == "H":
                        is_proton = True
                # or Atom_isotope_number == 1
                iso_col = "Atom_isotope_number" if "Atom_isotope_number" in idx else "_Atom_chem_shift.Atom_isotope_number" if "_Atom_chem_shift.Atom_isotope_number" in idx else None
                if not is_proton and iso_col is not None:
                    try:
                        if int(str(r[idx[iso_col]]).strip()) == 1:
                            is_proton = True
                    except Exception:
                        pass
                if not is_proton:
                    continue

                # ensure units are ppm (when present)
                u = frame_level_units
                if has_units:
                    units_col = "Chem_shift_units" if "Chem_shift_units" in idx else "_Atom_chem_shift.Chem_shift_units"
                    try:
                        u = str(r[idx[units_col]]).strip() or frame_level_units
                    except Exception:
                        u = frame_level_units
                if str(u).lower() != "ppm":
                    # skip non-ppm or unknown units
                    continue

                ppm_values.append(ppm)

    return ppm_values

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="bmrb_nmrstar/*/*.str",
                    help="Glob for input .str files (default: bmrb_nmrstar/*/*.str)")
    ap.add_argument("--out_dir", type=str, default="out_txt",
                    help="Output directory for per-entry TXT files")
    ap.add_argument("--descending", action="store_true",
                    help="Sort peaks high→low (NMR-style). Default is ascending.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    index_rows = []
    files = sorted(glob.glob(args.in_dir))
    if not files:
        print(f"No .str files matched {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    for fp in files:
        try:
            entry = pynmrstar.Entry.from_file(fp)
            entry_id = entry.entry_id or Path(fp).stem
            ppm_values = extract_1h_ppm(entry)
            if not ppm_values:
                # still write an empty file to record that we processed it
                out_txt = Path(args.out_dir) / f"{entry_id}.txt"
                out_txt.write_text("")  # empty
                index_rows.append({"entry_id": entry_id, "txt_path": str(out_txt), "n_peaks": 0, "source": fp})
                continue

            ppm_values = sorted(ppm_values, reverse=args.descending)

            # Write simple HMDB-like list (one peak per line). If you later want intensities,
            # add a second column; many BMRB entries don’t have intensities for assigned shifts.
            out_txt = Path(args.out_dir) / f"{entry_id}.txt"
            with open(out_txt, "w", encoding="utf-8") as f:
                # a tiny header as comments (optional)
                f.write(f"# entry_id: {entry_id}\n")
                f.write(f"# nucleus: 1H\n")
                f.write(f"# units: ppm\n")
                for v in ppm_values:
                    f.write(f"{v:.6f}\n")

            index_rows.append({"entry_id": entry_id, "txt_path": str(out_txt), "n_peaks": len(ppm_values), "source": fp})

        except Exception as e:
            # Record failures but keep going
            entry_id = Path(fp).stem
            out_txt = Path(args.out_dir) / f"{entry_id}.txt"
            index_rows.append({"entry_id": entry_id, "txt_path": str(out_txt), "n_peaks": 0, "source": fp, "error": str(e)})
            print(f"[WARN] Failed: {fp} -> {e}", file=sys.stderr)

    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(Path(args.out_dir) / "nmr_txt_index.csv", index=False)
    print(f"Done. Wrote {len(index_rows)} records. TXT files in: {args.out_dir}")

if __name__ == "__main__":
    main()
