import json
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
import zarr
from huggingface_hub import HfApi, download_bucket_files
from huggingface_hub.errors import HfHubHTTPError

BUCKET = "scrollprize/datasets"

SEGMENTS = [
    # {
    #     "scroll": "9b",
    #     "segment": "auto_grown_20250919055754487_inp_hr_2um",
    #     "volume": "ink/0009b/auto_grown_20250919055754487_inp_hr_2um/auto_grown_20250919055754487_inp_hr_2um.zarr",
    #     "label": "ink/0009b/auto_grown_20250919055754487_inp_hr_2um/auto_grown_20250919055754487_inp_hr_2um_inklabels.tif",
    #     "mask": "ink/0009b/auto_grown_20250919055754487_inp_hr_2um/auto_grown_20250919055754487_inp_hr_2um_supervision_mask.tif",
    #     "meta": "ink/0009b/auto_grown_20250919055754487_inp_hr_2um/meta.json",
    # },
    # {
    #     "scroll": "500p2",
    #     "segment": "-1",
    #     "volume": "ink/0500p2/-1/-1.zarr",
    #     "label": "ink/0500p2/-1/-1_inklabels.tif",
    #     "mask": "ink/0500p2/-1/-1_supervision_mask.tif",
    #     "meta": "ink/0500p2/-1/meta.json",
    # },
    {
        "scroll": "p343p",
        "segment": "20250511003658",
        "volume": "ink/unused/P343p/20250511003658/20250511003658.zarr",
        "label": "ink/unused/P343p/20250511003658/20250511003658_inklabels.tif",
        "mask": "ink/unused/P343p/20250511003658/20250511003658_supervision_mask.tif",
        "meta": "ink/unused/P343p/20250511003658/meta.json",
    },


    # {
    #     "scroll": "1667",
    #     "segment": "w013_20240304141531_2um",
    #     "volume": "ink/1667/w013_20240304141531_2um/w013_20240304141531_2um.zarr",
    #     "label": "ink/1667/w013_20240304141531_2um/w013_20240304141531_2um_inklabels.zarr",
    #     "mask": "ink/1667/w013_20240304141531_2um/w013_20240304141531_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w013_20240304141531_2um/meta.json",
    # },
    # {
    #     "scroll": "1667",
    #     "segment": "w018_20240304144031_2um",
    #     "volume": "ink/1667/w018_20240304144031_2um/w018_20240304144031_2um.zarr",
    #     "label": "ink/1667/w018_20240304144031_2um/w018_20240304144031_2um_inklabels.zarr",
    #     "mask": "ink/1667/w018_20240304144031_2um/w018_20240304144031_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w018_20240304144031_2um/meta.json",
    # },
    # {
    #     "scroll": "1667",
    #     "segment": "w023_20240304161941_2um",
    #     "volume": "ink/1667/w023_20240304161941_2um/w023_20240304161941_2um.zarr",
    #     "label": "ink/1667/w023_20240304161941_2um/w023_20240304161941_2um_inklabels.zarr",
    #     "mask": "ink/1667/w023_20240304161941_2um/w023_20240304161941_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w023_20240304161941_2um/meta.json",
    # },
    # {
    #     "scroll": "1667",
    #     "segment": "w028_20251208130119156_2um",
    #     "volume": "ink/1667/w028_20251208130119156_2um/w028_20251208130119156_2um.zarr",
    #     # "volume": None,
    #     "label": "ink/1667/w028_20251208130119156_2um/w028_20251208130119156_2um_inklabels.zarr",
    #     "mask": "ink/1667/w028_20251208130119156_2um/w028_20251208130119156_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w028_20251208130119156_2um/meta.json",
    # },
    # {
    #     "scroll": "1667",
    #     "segment": "w029_20251212185248662_2um",
    #     "volume": "ink/1667/w029_20251212185248662_2um/w029_20251212185248662_2um.zarr",
    #     "label": "ink/1667/w029_20251212185248662_2um/w029_20251212185248662_2um_inklabels.zarr",
    #     "mask": "ink/1667/w029_20251212185248662_2um/w029_20251212185248662_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w029_20251212185248662_2um/meta.json",
    # },
    # {
    #     "scroll": "1667",
    #     "segment": "w031_2025122323_2um",
    #     "volume": "ink/1667/w031_2025122323_2um/w031_2025122323_2um.zarr",
    #     "label": "ink/1667/w031_2025122323_2um/w031_2025122323_2um_inklabels.zarr",
    #     "mask": "ink/1667/w031_2025122323_2um/w031_2025122323_2um_supervision_mask.zarr",
    #     "meta": "ink/1667/w031_2025122323_2um/meta.json",
    # },
]

LOCAL_DIR = Path(r"D:\Dokumente\Studium\Masterarbeit\hf_data")

MAX_WORKERS = 12
BATCH_SIZE = 100


api = HfApi()


def list_folder(prefix, max_retries=10):
    for attempt in range(max_retries):
        try:
            files = []

            for item in api.list_bucket_tree(BUCKET, prefix=prefix, recursive=True):
                if item.__class__.__name__ == "BucketFile":
                    files.append(item)
            return files
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait_time = min(60 * (attempt + 1), 300)
                print(f"Rate Limit erreicht. Warte {wait_time} Sekunden...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Auflisten fehlgeschlagen: {prefix}")


# def download_batch(files, destination_root):
#     download_list = []

#     for file in files:
#         # Remove bucket path and put file into destination_root
#         relative_path = file.path

#         # This preserves the structure below the Zarr prefix
#         local_path = destination_root / Path(relative_path)
#         local_path.parent.mkdir(parents=True, exist_ok=True)

#         if local_path.exists():
#             continue

#         download_list.append((file, str(local_path)))

#     if not download_list:
#         return f"SKIP ({len(files)} Dateien bereits vorhanden)"

#     for attempt in range(10):
#         try:
#             download_bucket_files(BUCKET, files=download_list)
#             return f"OK ({len(download_list)} Dateien)"
#         except HfHubHTTPError as e:
#             if e.response is not None and e.response.status_code == 429:
#                 wait_time = min(60 * (attempt + 1), 300)
#                 print(f"Rate Limit beim Download. Warte {wait_time} Sekunden...")
#                 time.sleep(wait_time)
#             else:
#                 raise

#     return f"FAILED ({len(download_list)} Dateien)"


def download_single_file(file, output_path, max_retries=10):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"SKIP: {output_path.name} bereits vorhanden.")
        return

    download_list = [(file, str(output_path))]

    for attempt in range(max_retries):
        try:
            download_bucket_files(BUCKET, files=download_list)
            return
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait_time = min(60 * (attempt + 1), 300)
                print(f"Rate Limit beim Download. " f"Warte {wait_time} Sekunden...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Download fehlgeschlagen: {file.path}")


def download_zarr_store(zarr_volume, output_dir):
    """
    Download a complete Zarr store.

    The Zarr store consists of many individual files.
    All files are downloaded in parallel batches.
    """

    print("")
    print(f"Liste Zarr-Dateien auf: " f"{zarr_volume}")

    zarr_files = list_folder(zarr_volume)

    print(f"  {len(zarr_files):,} " f"Zarr-Dateien gefunden.")

    if not zarr_files:
        raise FileNotFoundError(f"Keine Dateien für Zarr gefunden: " f"{zarr_volume}")

    batches = [zarr_files[i : i + BATCH_SIZE] for i in range(0, len(zarr_files), BATCH_SIZE)]

    print(f"  {len(batches):,} Batches")

    def download_zarr_batch(batch):
        download_list = []

        for file in batch:
            relative_path = Path(file.path).relative_to(Path(zarr_volume))
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists():
                continue

            download_list.append((file, str(destination)))

        if not download_list:
            return f"SKIP ({len(batch)} Dateien)"

        for attempt in range(10):
            try:
                download_bucket_files(BUCKET, files=download_list)
                return f"OK ({len(download_list)} Dateien)"
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait_time = min(60 * (attempt + 1), 300)
                    print("Rate Limit beim Download. " f"Warte {wait_time} Sekunden...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Download-Batch fehlgeschlagen: " f"{zarr_volume}")

    download_success = True

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_zarr_batch, batch) for batch in batches]

        completed = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1
                print(f"[{completed}/{len(batches)}] " f"{result}")
            except Exception as e:
                download_success = False
                print(f"ERROR: {e}")

    if not download_success:
        raise RuntimeError(f"Zarr-Download fehlgeschlagen: " f"{zarr_volume}")

    print(f"Zarr gespeichert: {output_dir}")

    return zarr_files


def print_zarr_shapes(zarr_path):
    z = zarr.open(zarr_path, mode="r")

    print(f"\nZarr structure: {zarr_path}")
    print(z.tree())
    # print("")

    def visit(group, path=""):
        for name in group.keys():
            item = group[name]

            current_path = f"{path}/{name}" if path else name

            # if hasattr(item, "shape"):
            #     print(f"  {current_path}: shape={item.shape}, dtype={item.dtype}")
            # elif hasattr(item, "keys"):
            #     visit(item, current_path)

            if hasattr(item, "shape"):
                d, h, w = item.shape

                print(current_path)
                print(f"Volume shape:     {item.shape}")
                print(f"Dtype:            {item.dtype}")
                print(f"Chunks:           {item.chunks}")
                print(f"Number of slices: {d}")
                print("")
            elif hasattr(item, "keys"):
                visit(item, current_path)
    
    visit(z)


def find_file(path):
    files = list_folder(path)

    for file in files:
        if file.path == path:
            return file

    raise FileNotFoundError(f"Datei nicht im Bucket gefunden: {path}")


# def convert_tiff_to_png(input_file, output_file):
#     if output_file.exists():

#         print(f"SKIP: {output_file.name} bereits vorhanden.")

#         return

#     print(f"Konvertiere {input_file.name} -> " f"{output_file.name}")

#     image = tifffile.imread(input_file)

#     image = np.asarray(image)

#     # Remove unnecessary dimensions
#     image = np.squeeze(image)

#     if image.ndim != 2:
#         raise ValueError(f"Erwartetes 2D-Bild, aber " f"erhalten: {image.shape}")

#     # PIL can directly save uint8 and uint16 images
#     if image.dtype not in (np.uint8, np.uint16):
#         print(
#             f"WARNING: Unerwarteter Datentyp " f"{image.dtype}. Konvertiere nach uint8."
#         )

#         if image.max() > 0:
#             image = (image.astype(np.float32) / image.max() * 255).astype(np.uint8)

#         else:
#             image = image.astype(np.uint8)

#     Image.fromarray(image).save(output_file, format="PNG")


def convert_to_png(input_path, output_file):
    """
    Convert either a TIFF file or a Zarr array
    into a PNG file.

    Supported input formats:
        .tif
        .tiff
        .zarr
    """

    if output_file.exists():
        print(f"SKIP: {output_file.name} " "bereits vorhanden.")
        return

    print(f"Konvertiere {input_path.name} -> " f"{output_file.name}")

    if input_path.suffix.lower() in (".tif", ".tiff"):
        image = tifffile.imread(input_path)
    elif input_path.suffix.lower() == ".zarr":
        z = zarr.open(input_path, mode="r")

        if "0" not in z:
            available = list(z.keys())

            raise ValueError(
                f"No '0' array found in Zarr:\n"
                f"{input_path}\n"
                f"Available arrays/groups: {available}"
            )

        image = np.asarray(z["0"][32])
    else:
        raise ValueError(f"Nicht unterstütztes Format: " f"{input_path}")

    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(f"Erwartetes 2D-Bild, aber erhalten: " f"{image.shape}")

    # if image.dtype not in (np.uint8, np.uint16):
    #     print(f"WARNING: Unerwarteter Datentyp {image.dtype}. Konvertiere nach uint8.")

    #     if image.max() > 0:
    #         image = (image.astype(np.float32) / image.max() * 255).astype(np.uint8)
    #     else:
    #         image = image.astype(np.uint8)

    if image.max() > 0:
        image = (image.astype(np.float32) / image.max() * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    Image.fromarray(image).save(output_file, format="PNG")


def create_metadata(segment_info, zarr_files, output_segment_dir):
    total_size = 0

    for file in zarr_files:
        size = getattr(file, "size", None)

        if size is not None:
            total_size += size

    segment = segment_info["segment"]
    local_zarr = f"{segment}.zarr"
    local_label = f"{segment}_inklabels.png" if segment_info.get("label") is not None else None
    local_mask = f"{segment}_mask.png" if segment_info.get("mask") is not None else None
    local_meta = "meta.json" if segment_info.get("meta") is not None else None

    metadata = {
        "dataset": {
            "bucket": BUCKET,
            "scroll": segment_info["scroll"],
            "segment": segment,
            "zarr_source": segment_info["volume"],
            "label_source": segment_info.get("label"),
            "mask_source": segment_info.get("mask"),
            "meta_source": segment_info.get("meta"),
            "huggingface_url": f"https://huggingface.co/buckets/{BUCKET}/tree/main/{segment_info['volume']}",
        },
        "local": {
            "category": "labeled" if segment_info.get("label") is not None else "unlabeled",
            # "directory": str(output_segment_dir),
            "zarr": local_zarr,
            "label": local_label,
            "mask": local_mask,
            "meta": local_meta,
        },
        "conversion": {
            "label": {"source_format": "TIFF", "output_format": "PNG"} if segment_info.get("label") is not None else None,
            "mask": {"source_format": "TIFF", "output_format": "PNG"} if segment_info.get("mask") is not None else None,
            "meta": {"source_format": "JSON", "output_format": "JSON", "modified": False} if segment_info.get("meta") is not None else None,
        },
        "download": {
            "status": "complete",
            "downloaded_at": (datetime.now(timezone.utc).isoformat()),
            "zarr": {
                "num_files": len(zarr_files),
                "size_bytes": total_size,
                "size_gb": round(total_size / (1024**3), 3),
            },
            "settings": {"max_workers": MAX_WORKERS, "batch_size": BATCH_SIZE},
        },
    }

    metadata_path = output_segment_dir / "meta_download_info.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Download-Metadaten gespeichert: " f"{metadata_path}")


# def process_segment(segment_info):
#     scroll = segment_info["scroll"]
#     segment = segment_info["segment"]

#     is_labeled = segment_info["label"] is not None

#     label_type = "labeled" if is_labeled else "unlabeled"

#     scroll_dir = LOCAL_DIR / scroll
#     label_dir = scroll_dir / label_type
#     segment_dir = label_dir / segment
#     zarr_dir = segment_dir / f"{segment}.zarr"
#     segment_dir.mkdir(parents=True, exist_ok=True)
#     zarr_dir.mkdir(parents=True, exist_ok=True)

#     print("")
#     print("=" * 70)
#     print(f"Scroll:   {scroll}")
#     print(f"Segment:  {segment}")
#     print(f"Status:   {label_type}")
#     print(f"Output:   {segment_dir}")
#     print("=" * 70)
#     print("")

#     print(f"Liste Zarr-Dateien auf: " f"{segment_info['prefix']}")
#     zarr_files = list_folder(segment_info["prefix"])
#     print(f"  {len(zarr_files):,} Zarr-Dateien gefunden.")

#     # We need paths relative to the Zarr prefix.
#     zarr_prefix = segment_info["prefix"]

#     batches = []

#     current_batch = []

#     for file in zarr_files:
#         # Path inside the Zarr directory
#         relative_path = Path(file.path).relative_to(Path(zarr_prefix).parent)

#         # Destination:
#         # segment/<segment>.zarr/<remaining path>
#         destination = zarr_dir / Path(file.path).relative_to(Path(zarr_prefix))

#         current_batch.append((file, destination))

#         if len(current_batch) >= BATCH_SIZE:

#             batches.append(current_batch)

#             current_batch = []

#     if current_batch:
#         batches.append(current_batch)

#     print("")
#     print(f"Starte Zarr-Download mit " f"{MAX_WORKERS} Workern...")
#     print(f"{len(batches):,} Batches")
#     print("")

#     def download_zarr_batch(batch):
#         download_list = []

#         for file, destination in batch:
#             destination.parent.mkdir(parents=True, exist_ok=True)

#             if destination.exists():
#                 continue

#             download_list.append((file, str(destination)))

#         if not download_list:
#             return f"SKIP ({len(batch)} Dateien)"

#         for attempt in range(10):
#             try:
#                 download_bucket_files(BUCKET, files=download_list)
#                 return f"OK ({len(download_list)} Dateien)"
#             except HfHubHTTPError as e:
#                 if e.response is not None and e.response.status_code == 429:
#                     wait_time = min(60 * (attempt + 1), 300)
#                     print(
#                         f"Rate Limit beim Download. " f"Warte {wait_time} Sekunden..."
#                     )
#                     time.sleep(wait_time)
#                 else:
#                     raise

#         return f"FAILED ({len(download_list)} Dateien)"

#     download_success = True

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = [executor.submit(download_zarr_batch, batch) for batch in batches]

#         completed = 0

#         for future in as_completed(futures):
#             try:
#                 result = future.result()
#                 completed += 1
#                 print(f"[{completed}/{len(batches)}] " f"{result}")
#             except Exception as e:
#                 download_success = False
#                 print(f"ERROR: {e}")

#     if not download_success:
#         raise RuntimeError(f"Zarr-Download für " f"{segment} war nicht erfolgreich.")

#     if segment_info["label"] is not None:
#         print("")
#         print("Download Label...")

#         label_file = find_file(segment_info["label"])
#         temporary_label = segment_dir / f"{segment}_inklabels.tif"
#         output_label = segment_dir / f"{segment}_inklabels.png"
#         download_single_file(label_file, temporary_label)
#         convert_tiff_to_png(temporary_label, output_label)

#         # Delete temporary TIFF
#         if temporary_label.exists():
#             temporary_label.unlink()

#     if segment_info["mask"] is not None:
#         print("")
#         print("Download Mask...")

#         mask_file = find_file(segment_info["mask"])
#         temporary_mask = segment_dir / f"{segment}_supervision_mask.tif"
#         output_mask = segment_dir / f"{segment}_mask.png"
#         download_single_file(mask_file, temporary_mask)
#         convert_tiff_to_png(temporary_mask, output_mask)

#         # Delete temporary TIFF
#         if temporary_mask.exists():
#             temporary_mask.unlink()

#     if segment_info.get("meta") is not None:
#         print("")
#         print("Download meta.json...")

#         meta_file = find_file(segment_info["meta"])
#         output_meta = segment_dir / "meta.json"
#         download_single_file(meta_file, output_meta)
#         print(f"meta.json gespeichert: " f"{output_meta}")

#     print("")
#     print("Erstelle Metadaten...")

#     create_metadata(segment_info, zarr_files, segment_dir)

#     print("")
#     print(f"Segment abgeschlossen: {segment}")


def process_segment(segment_info):
    scroll = segment_info["scroll"]

    segment = segment_info["segment"]

    is_labeled = segment_info["label"] is not None

    label_type = "labeled" if is_labeled else "unlabeled"

    scroll_dir = LOCAL_DIR / scroll
    label_dir = scroll_dir / label_type
    segment_dir = label_dir / segment
    segment_dir.mkdir(parents=True, exist_ok=True)

    zarr_files = []

    # --------------------------------------------------------
    # Main Zarr
    # --------------------------------------------------------

    if segment_info.get("volume") is not None:
        zarr_dir = segment_dir / f"{segment}.zarr"

        print("")
        print("=" * 70)
        print(f"Scroll:   {scroll}")
        print(f"Segment:  {segment}")
        print(f"Status:   {label_type}")
        print(f"Output:   {segment_dir}")
        print("=" * 70)
        print("")

        zarr_files.extend(download_zarr_store(segment_info["volume"], zarr_dir))
        print_zarr_shapes(zarr_dir)

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if segment_info.get("label") is not None:
        print("")
        print("Download Label...")

        label_source = segment_info["label"]
        output_label = segment_dir / f"{segment}_inklabels.png"

        if label_source.lower().endswith(".zarr"):
            temporary_label = segment_dir / f"{segment}_inklabels.zarr"

            zarr_files.extend(download_zarr_store(label_source, temporary_label))
            print_zarr_shapes(temporary_label)
            convert_to_png(temporary_label,output_label)

            if temporary_label.exists():
                shutil.rmtree(temporary_label)
        else:
            label_file = find_file(label_source)
            temporary_label = segment_dir / f"{segment}_inklabels.tif"

            download_single_file(label_file, temporary_label)
            convert_to_png(temporary_label, output_label)

            if temporary_label.exists():
                temporary_label.unlink()

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    if segment_info.get("mask") is not None:
        print("")
        print("Download Mask...")

        mask_source = segment_info["mask"]
        output_mask = segment_dir / f"{segment}_mask.png"

        if mask_source.lower().endswith(".zarr"):
            temporary_mask = segment_dir / f"{segment}_supervision_mask.zarr"

            zarr_files.extend(download_zarr_store(mask_source, temporary_mask))
            print_zarr_shapes(temporary_mask)
            convert_to_png(temporary_mask, output_mask)

            if temporary_mask.exists():
                shutil.rmtree(temporary_mask)
        else:
            mask_file = find_file(mask_source)
            temporary_mask = segment_dir / f"{segment}_supervision_mask.tif"

            download_single_file(mask_file, temporary_mask)
            convert_to_png(temporary_mask, output_mask)

            if temporary_mask.exists():
                temporary_mask.unlink()

    # --------------------------------------------------------
    # Meta
    # --------------------------------------------------------

    if segment_info.get("meta") is not None:
        print("")
        print("Download meta.json...")

        meta_file = find_file(segment_info["meta"])

        output_meta = segment_dir / "meta.json"

        download_single_file(meta_file, output_meta)

        print(f"meta.json gespeichert: " f"{output_meta}")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("")
    print("Erstelle Metadaten...")

    create_metadata(segment_info, zarr_files, segment_dir)

    print("")
    print(f"Segment abgeschlossen: " f"{segment}")


def main():
    print("")
    print("=" * 70)
    print("Hugging Face Segment Downloader")
    print("=" * 70)
    print("")

    print(f"Anzahl Segmente: {len(SEGMENTS)}")

    print(f"Lokaler Speicherort: {LOCAL_DIR}")

    print("")

    successful = 0
    failed = 0

    for i, segment_info in enumerate(SEGMENTS, start=1):
        print("")
        print(f"[{i}/{len(SEGMENTS)}] Verarbeite {segment_info['scroll']} / {segment_info['segment']}")

        try:
            process_segment(segment_info)
            successful += 1
        except Exception as e:
            failed += 1
            print("")
            print(f"ERROR bei " f"{segment_info['segment']}: " f"{e}")

    print("")
    print("=" * 70)
    print("Alle Segmente verarbeitet.")
    print(f"Erfolgreich verarbeitet: {successful}/{len(SEGMENTS)}")
    if failed:
        print(f"Fehlgeschlagen:          {failed}/{len(SEGMENTS)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
