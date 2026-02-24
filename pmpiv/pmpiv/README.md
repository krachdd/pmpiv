# pmpiv package

Source code for the `pmpiv` library. Import as:

```python
import pmpiv
```

## Modules

| Module | Contents |
|---|---|
| `annotations.py` | `Annotation_Handler` — converts COCO JSON masks to pixel DataFrames; computes annotated area/volume. `Annotation_Reader` — reads previously exported annotation CSVs. |
| `df_io.py` | `read_csv` / `write2csv` — semicolon-delimited trajectory DataFrame I/O. |
| `filtering.py` | `Filtering` — percentile-based feature filtering. `Annotation_Filtering` — removes or extracts particles by spatial region. `filter_static` — removes non-moving trajectories. `filter_stubs` — removes short trajectories. `complete_removal` — applies all REMOVAL annotations from metadata. |
| `fstats.py` | `Frame_Statistics` — histogram of feature properties for a single frame. `Sequence_Statistics` — aggregated statistics over an entire image sequence (parallel or sequential); results are cached in `seq_stats.json`. |
| `helper.py` | `_check_metadata`, `_folder_sanity`, `_file_sanity`, `read_tif` — shared utilities. |
| `image_sequence.py` | `Image_Sequence` — PIMS wrapper with frame subsampling, annotated PNG/TIFF output (sequential and parallel). |
| `metadata.py` | `Metadata` — parses and validates the plain-text experiment config file. |
| `motion_stats.py` | `Motion_Statistics` — per-frame and sequence-wide displacement and velocity statistics. |
| `ploting.py` | `trajectory` — trajectory overlay plot. `ymapped_velocity` — mean velocity as a function of cross-channel position. |

## Tests

Unit and integration tests live in `tests/`. Run with:

```bash
PYTHONPATH=.:../pims:../trackpy MPLBACKEND=Agg pytest tests/ -v
```
