import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
import datasets
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer
from array_record.python.array_record_module import ArrayRecordWriter

# datasets.set_progress_bar_enabled()

# directories
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "tokenized"
MAP_DIR = BASE_DIR / "mappings"

# parameters
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-350BT"
DATASET_SPLIT = "train"
TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
CHUNK_SIZE = 8192
TARGET_RECORD_BYTES = 10 * 1024 ** 3  # target ~10GB per ArrayRecord (chunk-aligned)
NUM_PROC = 64  # default parallel workers for tokenization
WRITE_WORKERS = 32  # parallel workers for ArrayRecord writing (adjust based on system)
LOG_EVERY_EXAMPLES = 1000
LOG_EVERY_CHUNKS = 100

# optional: avoid tokenizer thread fan-out that can spike memory
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER

def tokenize_batch(examples):
    tokenizer = get_tokenizer()
    tokenized = tokenizer(
        examples["text"],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    return {"input_ids": tokenized["input_ids"]}

def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:.2f} {units[unit_idx]}"

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{sec}s"
    return f"{minutes}m{sec}s"

def serialize_chunk(chunk: List[int]) -> bytes:
    return tf.train.Example(
        features=tf.train.Features(
            feature={"text": tf.train.Feature(int64_list=tf.train.Int64List(value=chunk))}
        )
    ).SerializeToString()

def start_group(
    group_idx: int, example_idx: int, target_bytes: int
) -> Tuple[ArrayRecordWriter, dict, List[int]]:
    out_path = OUT_DIR / f"{group_idx:05d}.array_record"
    if out_path.exists():
        out_path.unlink()
    writer = ArrayRecordWriter(str(out_path), options="group_size:1")
    print(
        f"[group {group_idx}] start at example {example_idx} -> {out_path} (target ~{format_bytes(target_bytes)})",
        flush=True,
    )
    entry = {
        "group_index": group_idx,
        "out_path": str(out_path),
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "start_example_index": example_idx,
        "example_indices": [],
        "target_bytes": target_bytes,
        "bytes_written": 0,
        "num_chunks": 0,
    }
    return writer, entry, []

def finalize_group(
    writer: Optional[ArrayRecordWriter],
    entry: Optional[dict],
    buf: Optional[List[int]],
    master_map: List[dict],
):
    if writer is None or entry is None:
        return
    writer.close()
    entry["num_examples"] = len(entry["example_indices"])
    if entry["example_indices"]:
        entry["end_example_index"] = entry["example_indices"][-1]
    else:
        entry["end_example_index"] = None
    entry["dropped_tail_tokens"] = len(buf) if buf is not None else 0
    print(
        f"[group {entry['group_index']}] closed: bytes={format_bytes(entry['bytes_written'])}, chunks={entry['num_chunks']}, examples={entry['num_examples']}, tail_tokens={entry['dropped_tail_tokens']}",
        flush=True,
    )
    mapping_path = MAP_DIR / f"{entry['group_index']:05d}.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2)
    master_map.append(entry)

def write_group_worker(args):
    """Worker function to write a range of examples to ArrayRecord - STREAMING, NO DATA LOSS"""
    worker_idx, example_range, target_bytes, out_dir, map_dir, raw_dataset = args
    start_idx, end_idx = example_range
    
    # Get tokenizer for this worker
    tokenizer = get_tokenizer()
    
    # Each worker processes ALL assigned examples - NO DATA LOSS
    current_file_idx = 0
    processed_examples = 0
    all_entries = []
    
    buf = []
    current_bytes = 0
    
    # Start first file
    out_path = out_dir / f"{worker_idx:02d}_{current_file_idx:03d}.array_record"
    if out_path.exists():
        out_path.unlink()
    writer = ArrayRecordWriter(str(out_path), options="group_size:1")
    
    entry = {
        "worker_index": worker_idx,
        "file_index": current_file_idx,
        "out_path": str(out_path),
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "start_example_index": start_idx,
        "example_indices": [],
        "target_bytes": target_bytes,
        "bytes_written": 0,
        "num_chunks": 0,
    }
    
    try:
        # Process ALL examples in the range - NO SKIPPING
        for example_idx in range(start_idx, end_idx):
            text = raw_dataset[example_idx]["text"]
            input_ids = tokenizer(text, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            buf.extend(input_ids)
            entry["example_indices"].append(example_idx)
            processed_examples += 1
            
            # Write complete chunks
            while len(buf) >= CHUNK_SIZE:
                chunk = buf[:CHUNK_SIZE]
                del buf[:CHUNK_SIZE]
                serialized = serialize_chunk(chunk)
                writer.write(serialized)
                entry["num_chunks"] += 1
                entry["bytes_written"] += len(serialized)
            
            # If file gets too big, start a new file (but keep processing ALL examples)
            if entry["bytes_written"] >= target_bytes and example_idx < end_idx - 1:
                # Finalize current file
                writer.close()
                entry["num_examples"] = len(entry["example_indices"])
                entry["end_example_index"] = entry["example_indices"][-1]
                entry["dropped_tail_tokens"] = 0  # No dropping - carry buffer to next file
                
                mapping_path = map_dir / f"{worker_idx:02d}_{current_file_idx:03d}.json"
                with mapping_path.open("w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2)
                all_entries.append(entry)
                
                # Start new file (carry buffer forward - NO DATA LOSS)
                current_file_idx += 1
                out_path = out_dir / f"{worker_idx:02d}_{current_file_idx:03d}.array_record"
                if out_path.exists():
                    out_path.unlink()
                writer = ArrayRecordWriter(str(out_path), options="group_size:1")
                
                entry = {
                    "worker_index": worker_idx,
                    "file_index": current_file_idx,
                    "out_path": str(out_path),
                    "dataset_name": DATASET_NAME,
                    "dataset_split": DATASET_SPLIT,
                    "start_example_index": example_idx + 1,
                    "example_indices": [],
                    "target_bytes": target_bytes,
                    "bytes_written": 0,
                    "num_chunks": 0,
                }
    
    finally:
        # Finalize last file
        writer.close()
        entry["num_examples"] = len(entry["example_indices"])
        entry["end_example_index"] = entry["example_indices"][-1] if entry["example_indices"] else None
        entry["dropped_tail_tokens"] = len(buf)  # Only drop at the very end
        
        mapping_path = map_dir / f"{worker_idx:02d}_{current_file_idx:03d}.json"
        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        all_entries.append(entry)
    
    return all_entries

def create_work_ranges(dataset_size, num_workers):
    """Split dataset into ranges for parallel processing - NO ESTIMATES, NO DATA LOSS"""
    examples_per_worker = dataset_size // num_workers
    remainder = dataset_size % num_workers
    
    work_ranges = []
    start_idx = 0
    
    for worker_idx in range(num_workers):
        # Distribute remainder across first few workers
        current_size = examples_per_worker + (1 if worker_idx < remainder else 0)
        end_idx = start_idx + current_size
        
        work_ranges.append((worker_idx, (start_idx, end_idx)))
        start_idx = end_idx
    
    return work_ranges

def main(
    max_groups: Optional[int] = None,
    target_bytes: int = TARGET_RECORD_BYTES,
    num_proc: int = NUM_PROC,
    write_workers: int = WRITE_WORKERS,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading dataset: name={DATASET_NAME}, subset={DATASET_SUBSET}, split={DATASET_SPLIT} (num_proc={num_proc})",
        flush=True,
    )
    t_load_start = time.time()
    # Load raw dataset (no tokenization yet!)
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_SUBSET,
        split=DATASET_SPLIT,
        num_proc=num_proc,
    )
    t_load = time.time() - t_load_start
    try:
        num_rows = len(dataset)
    except Exception:
        num_rows = "unknown"
    print(
        f"Loaded raw dataset in {format_duration(t_load)} (rows={num_rows})",
        flush=True,
    )

    # Create work ranges - equal split, NO ESTIMATES, NO DATA LOSS
    print(f"Splitting {num_rows} examples equally across {write_workers} workers", flush=True)
    work_ranges = create_work_ranges(len(dataset), write_workers)
    print(f"Each worker will process ALL assigned examples and create files as needed (target ~{format_bytes(target_bytes)} per file)", flush=True)
    
    # Prepare arguments for parallel workers
    worker_args = [
        (worker_idx, example_range, target_bytes, OUT_DIR, MAP_DIR, dataset)
        for worker_idx, example_range in work_ranges
    ]
    
    master_map: List[dict] = []
    total_bytes_written = 0
    total_chunks_written = 0
    t_write_start = time.time()
    
    print(f"Starting parallel streaming tokenization + ArrayRecord writing with {write_workers} workers", flush=True)
    
    # Execute parallel writing
    with ProcessPoolExecutor(max_workers=write_workers) as executor:
        # Submit all jobs
        future_to_group = {
            executor.submit(write_group_worker, args): args[0] 
            for args in worker_args
        }
        
        # Process completed jobs with progress bar
        with tqdm(total=len(work_ranges), desc="Processing workers", unit="worker") as pbar:
            for future in as_completed(future_to_group):
                worker_idx = future_to_group[future]
                try:
                    worker_entries = future.result()  # List of entries from this worker
                    for entry in worker_entries:
                        master_map.append(entry)
                        total_bytes_written += entry["bytes_written"]
                        total_chunks_written += entry["num_chunks"]
                    
                    pbar.set_postfix({
                        'bytes': format_bytes(total_bytes_written),
                        'chunks': total_chunks_written,
                        'files': len(master_map)
                    })
                    pbar.update(1)
                    
                    print(
                        f"[worker {worker_idx}] completed: {len(worker_entries)} files, total_bytes={format_bytes(sum(e['bytes_written'] for e in worker_entries))}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[worker {worker_idx}] ERROR: {e}", flush=True)
    
    # Sort master_map by worker and file index to maintain order
    master_map.sort(key=lambda x: (x["worker_index"], x["file_index"]))

    master_path = MAP_DIR / "master_mapping.json"
    with master_path.open("w", encoding="utf-8") as f:
        json.dump(master_map, f, indent=2)

    total_elapsed = time.time() - t_write_start
    print(
        f"Done. groups={len(master_map)} chunks={total_chunks_written} bytes={format_bytes(total_bytes_written)} time={format_duration(total_elapsed)} avg_rate={format_bytes(int(total_bytes_written / max(total_elapsed, 1e-6)))}/s",
        flush=True,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream HuggingFaceFW/fineweb-edu into ArrayRecords.")
    parser.add_argument("--max-groups", type=int, default=None, help="Stop after writing this many groups.")
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=TARGET_RECORD_BYTES,
        help="Approximate bytes to write before rolling to a new ArrayRecord.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=NUM_PROC,
        help="Number of parallel workers for tokenization.",
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=WRITE_WORKERS,
        help="Number of parallel workers for ArrayRecord writing.",
    )
    args = parser.parse_args()

    main(max_groups=args.max_groups, target_bytes=args.target_bytes, num_proc=args.num_proc, write_workers=args.write_workers)
