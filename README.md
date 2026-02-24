# pmpiv

A `python3`-based **P**orous **M**edia **P**article **I**mage **V**elocimetry toolbox for microfluidic domains, developed at [MIB](https://www.mib.uni-stuttgart.de/) and the [Porous Media Lab](https://www.mib.uni-stuttgart.de/pml/).

`pmpiv` provides tools to track PIV particles and evaluate their properties inside porous microfluidic geometries. It wraps and extends [`trackpy`](https://github.com/soft-matter/trackpy) and [`PIMS`](https://soft-matter.github.io/pims/v0.6.1/) (Python IMage Sequence), and adds region-based filtering via COCO-format annotations.

---

## Requirements

Tested on Linux. Python 3.10+ is required. Dependencies are managed with [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main); the full environment specification is in `envs/pmpiv.yml`.

**Create the environment:**
```bash
conda env create -f envs/pmpiv.yml
conda activate pmpiv
```

**Add the package to `PYTHONPATH`** (required each session, or add to your shell profile):
```bash
source addpythonpath.sh
```

---

## Project Structure

```
pmpiv/
├── addpythonpath.sh          # Sets PYTHONPATH for pmpiv, pims, trackpy
├── envs/
│   └── pmpiv.yml             # Conda environment specification
├── pmpiv/
│   ├── pmpiv/                # Library source code
│   │   ├── annotations.py    # COCO annotation I/O and mask conversion
│   │   ├── df_io.py          # DataFrame CSV read/write helpers
│   │   ├── filtering.py      # Particle filtering (spatial, static, stub)
│   │   ├── fstats.py         # Per-frame and sequence statistics
│   │   ├── helper.py         # Shared utilities (metadata, TIF I/O)
│   │   ├── image_sequence.py # Image sequence loading and annotated output
│   │   ├── metadata.py       # Config file parser
│   │   ├── motion_stats.py   # Displacement and velocity computation
│   │   └── ploting.py        # Trajectory and velocity profile plots
│   ├── tests/                # Pytest test suite (107 tests)
│   ├── pims/                 # Bundled PIMS (external subtree)
│   └── trackpy/              # Bundled trackpy (external subtree)
├── studies/
│   └── template/             # Annotated workflow template
└── tests/
    └── test1/                # Integration test with sample data
```

---

## Usage

Define your experiment parameters in a `.txt` config file and pass its path as a command-line argument to your study script:

```bash
python3 studies/template/template.py studies/template/template.txt
```

See `studies/template/` for a fully annotated example.

---

## Input Parameters

All parameters are specified as `KEY value` (one per line) in a plain-text config file. Comments start with `#`.

| Parameter | Type | Description |
|---|---|---|
| `IN_PATH` | str | Directory containing the input image files. |
| `WORKING_DIR` | str | Directory for output data, plots, and CSVs. |
| `IN_FORMAT` | str | Image file extension: `tif`, `tiff`, `TIF`, or `TIFF`. |
| `PIXELSIZE` | float | Physical size of one pixel in metres (e.g. `1.3e-06`). |
| `HEIGHT` | float | Channel/capillary height in metres (e.g. `100.0e-06`). |
| `START_FRAME` | int | Index of the first frame to process. |
| `END_FRAME` | int | Index of the last frame to process. Set to `0` to use all frames. |
| `RATE` | int | Frame subsampling rate (e.g. `2` uses every second frame). |
| `FEATURE_SIZE` | int | Approximate feature diameter in pixels — **must be an odd integer**. |
| `FEATURE_MIN_SIZE` | int | Minimum integrated brightness (mass) threshold; increase to reject noise. |
| `FEATURES_ARE_DARK` | bool | Set to `True` if particles are darker than the background. |
| `FPS` | float | Acquisition frame rate in Hz. |
| `MAX_PARTICLE_SPEED` | int | Maximum particle displacement between consecutive frames in pixels. Smaller values speed up linking. |
| `MEMORY` | int | Number of frames a particle may disappear and still be re-linked. |
| `DURATION` | int | Minimum trajectory length in frames; shorter trajectories are removed as stubs. |
| `REMOVE_STATIC` | bool | Set to `True` to enable static particle filtering. |
| `CHECK_STATIC` | int | Minimum total displacement (pixels) for a trajectory to be considered moving. |
| `STATIC_DEV_PARAMETER` | float | Position standard-deviation threshold (legacy; `CHECK_STATIC` is preferred). |
| `JSON_PATH` | str | Directory containing COCO annotation `.json` files. |
| `REMOVAL` | str | Comma-separated list of `.json` annotation files. Particles **inside** these regions are removed. Leave blank for none. |
| `EXTRACTION` | str | Comma-separated list of `.json` annotation files. Only particles **inside** these regions are kept. Leave blank for none. |

**Example config:**
```
IN_PATH       /data/experiment1/images
WORKING_DIR   /data/experiment1/results
IN_FORMAT     tif
PIXELSIZE     1.3e-06
HEIGHT        100.0e-06
START_FRAME   0
END_FRAME     0
RATE          1
FEATURE_SIZE  11
FEATURE_MIN_SIZE 500
FEATURES_ARE_DARK True
FPS           70.0
MAX_PARTICLE_SPEED 4
MEMORY        6
DURATION      5
REMOVE_STATIC True
CHECK_STATIC  50
STATIC_DEV_PARAMETER 1.0
JSON_PATH     /data/experiment1/annotations
REMOVAL
EXTRACTION    capillary.json
```

---

## Typical Workflow

1. **Load image sequence** — reads all frames from `IN_PATH` using PIMS.
2. **Compute sequence statistics** — locates features in each frame and computes per-frame statistics (mass, size, eccentricity). Results are cached in `seq_stats.json`.
3. **Batch feature detection** — locates Gaussian-like blobs in all frames with `trackpy.batch`.
4. **(Optional) Percentile filtering** — removes features with extreme mass, size, or eccentricity values.
5. **Link trajectories** — connects per-frame detections into particle tracks using the Crocker–Grier algorithm (`trackpy.link`).
6. **Remove stub trajectories** — discards trajectories shorter than `DURATION` frames.
7. **Region-based removal** — removes particles located inside annotated solid structures (COCO JSON via `REMOVAL`).
8. **Region-based extraction** — retains only particles inside regions of interest (COCO JSON via `EXTRACTION`).
9. **Remove static particles** — discards trajectories with total displacement below `CHECK_STATIC`.
10. **Compute velocity statistics** — calculates mean displacement and velocity per frame and over the full sequence.
11. **Export results** — saves trajectory DataFrames as semicolon-separated CSVs and produces trajectory/velocity plots.

---

## Annotations

Regions of interest (e.g. capillary lumen, solid walls) are annotated as COCO-format polygons in `.json` files.

**Recommended annotation tool:** [digitalsreeni-image-annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)

```bash
conda activate image_anno
digitalsreeni-image-annotator
```

After annotation, export as COCO JSON and place the files in the directory specified by `JSON_PATH`. List them under `REMOVAL` or `EXTRACTION` in your config file as appropriate.

---

## Running the Tests

The test suite requires `pytest` and the project dependencies to be installed.

```bash
cd pmpiv/pmpiv
PYTHONPATH=pmpiv:pims:trackpy MPLBACKEND=Agg pytest pmpiv/tests/ -v
```

Or, from inside `pmpiv/pmpiv/pmpiv/`:

```bash
PYTHONPATH=.:../pims:../trackpy MPLBACKEND=Agg pytest tests/
```

---

## Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy (Project number 390740016 – EXC 2075) and the Collaborative Research Center 1313 (Project number 327154368 – SFB 1313). Support by the Stuttgart Center for Simulation Science (SimTech) is gratefully acknowledged.

---

## Developer

[David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/) — [david.krach@mib.uni-stuttgart.de](mailto:david.krach@mib.uni-stuttgart.de)

## Contact

- Software support: [software@mib.uni-stuttgart.de](mailto:software@mib.uni-stuttgart.de)
- Data support: [data@mib.uni-stuttgart.de](mailto:data@mib.uni-stuttgart.de)
