# Data preparation

The pipeline works with volumetric scroll segments: a stack of grayscale TIFF
slices, an optional ink annotation, and optional masks. Large volumes are
converted to chunked Zarr stores so training can read individual tiles without
loading a complete segment into memory.

## Obtain and organize the data

Data and trained checkpoints are not included in this repository. Supply the
Vesuvius Challenge segments and annotations needed by your experiment.

[`utils/download_data_from_huggingface.py`](utils/download_data_from_huggingface.py)
contains a downloader for selected ScrollPrize data. Review its `SEGMENTS` and
`LOCAL_DIR` settings before running it: it is an experiment-specific script,
and some entries download annotations only (`volume: None`). It is not an
automatic download of every segment referenced by the training configs.

Place TIFF input under the directory selected by `DATA_DIR` in your root `.env`:

```text
<DATA_DIR>/
└── data_tiff/
    └── 0172/                         # Scroll ID
        ├── labeled/
        │   └── <segment>/
        │       ├── layers/
        │       │   ├── 000.tif
        │       │   ├── 001.tif
        │       │   └── ...
        │       ├── <segment>_inklabels.png
        │       ├── <segment>_mask.png
        │       └── <segment>_validation_mask.png
        └── unlabeled/
            └── <another_segment>/
                └── layers/
                    └── ...
```

Slice filenames must sort in depth order. Ink labels, masks, and validation
masks are optional at conversion time. Training and validation segments must
have ink labels. PNG annotations are converted to binary arrays using a pixel
threshold of `> 127`. A mask marks valid pixels; a validation mask can separate
training and validation regions within the same segment.

Configuration IDs combine the scroll and segment folder names, for example
`0172_20241025145341`. The on-disk segment folder in that example is
`20241025145341`, inside the `0172` scroll folder.

## Convert TIFF stacks to Zarr

Run the conversion from the repository root in the activated environment.
Replace the paths with your actual data locations:

```bash
python data/utils/tiff_to_zarr.py /absolute/path/to/data/data_tiff /absolute/path/to/data/data_zarr
```

The converter preserves the scroll/category/segment hierarchy. Each segment
contains `<segment>.zarr`, `metadata.json`, and a `.conversion_complete` marker.
The store contains `volume` with shape `(depth, height, width)` and, when
available, `label`, `mask`, and `validation_mask` with shape `(height, width)`.
Image values and dtype are preserved; spatial dimensions are cropped to the
common minimum size of the input images and annotations.

Useful options include `--segments <segment> ...`, `--chunks Z Y X`,
`--crops SEGMENT:Y_START,Y_END,X_START,X_END ...`, and `--workers N`.
Use `python data/utils/tiff_to_zarr.py --help` for the full interface.
Completed segments are skipped on subsequent conversions; incomplete outputs
are rebuilt.

For conversion **and** tar archives for cluster staging, use the original
preparation wrapper from its own directory:

```bash
cd data/utils
bash prepare_data.sh /absolute/path/to/data
```

This reads `data_tiff/` and creates `data_zarr/` and `data_tar/` beneath the
supplied directory. Tile extraction happens in the dataset during training;
the wrapper's separate tile-export step is disabled.

## Connect the data to an experiment

Use `path=local` to read `${DATA_DIR}/data_zarr` directly. The default
`path=alex` profile uses tar archives and cluster scratch storage; see the
[configuration guide](../conf/README.md) and
[Slurm instructions](../slurm_job_scripts/README.md).

Set `train.labeled`, `train.unlabeled`, and `train.val` to the IDs actually
available in your data. Prediction uses `train.segments` in
[`conf/predict.yaml`](../conf/predict.yaml). Dataset depth, center slice, tile
size, and normalization must match the intended model and checkpoint.

Keep training and validation segments separate. If the same segment is used
in both, set `train.use_validation_mask=true` and supply a `validation_mask`;
the training startup checks require it.
