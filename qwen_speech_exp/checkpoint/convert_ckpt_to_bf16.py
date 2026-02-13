"""Convert a MaxText Orbax checkpoint from float32 to bfloat16.

Loads the full checkpoint tree, casts all float32 arrays to bfloat16,
and saves to a new checkpoint path. This halves checkpoint size on disk
and eliminates the fp32->bf16 cast during TPU loading.

Usage:
  python qwen_speech_exp/checkpoint/convert_ckpt_to_bf16.py \
    --source gs://arabic-asr-dataset/checkpoints/qwen3-omni-30b-a3b-thinking \
    --dest gs://arabic-asr-dataset/checkpoints/qwen3-omni-30b-a3b-thinking-bf16

Requires a machine with enough host RAM to hold the full fp32 checkpoint
(~120GB for 30B params). TPU worker hosts (340GB RAM) work well for this.
"""

import argparse
import gc
import time

from etils import epath
from flax.training import train_state
import jax
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
from orbax.checkpoint._src.serialization import type_handlers


def cast_leaf_to_bf16(arr):
  """Cast a single array from float32 to bfloat16 if applicable."""
  if isinstance(arr, np.ndarray) and arr.dtype == np.float32:
    return arr.astype(ml_dtypes.bfloat16)
  if isinstance(arr, jax.Array) and arr.dtype == np.float32:
    return np.array(arr).astype(ml_dtypes.bfloat16)
  return arr


def main():
  parser = argparse.ArgumentParser(description="Convert MaxText checkpoint from fp32 to bf16")
  parser.add_argument("--source", required=True, help="Source checkpoint base dir (e.g. gs://bucket/checkpoints/model)")
  parser.add_argument("--dest", required=True, help="Destination checkpoint base dir for bf16 version")
  parser.add_argument("--step", type=int, default=0, help="Checkpoint step to convert (default: 0)")
  parser.add_argument("--use_ocdbt", type=bool, default=True, help="Use OCDBT format (default: True)")
  parser.add_argument("--use_zarr3", type=bool, default=True, help="Use Zarr3 format (default: True)")
  args = parser.parse_args()

  source_items_path = f"{args.source}/{args.step}/items"
  print(f"Source checkpoint: {source_items_path}")
  print(f"Destination: {args.dest}")

  # --- Step 1: Load the checkpoint ---
  print("\n[1/4] Loading checkpoint from source...")
  start = time.time()

  ckptr = ocp.PyTreeCheckpointer(
      use_ocdbt=args.use_ocdbt,
      use_zarr3=args.use_zarr3,
  )
  state = ckptr.restore(epath.Path(source_items_path))
  elapsed = time.time() - start
  print(f"  Loaded in {elapsed / 60:.1f} min")

  # --- Step 2: Inspect dtypes before conversion ---
  flat_leaves, treedef = jax.tree_util.tree_flatten(state)
  dtype_counts = {}
  total_bytes = 0
  for leaf in flat_leaves:
    if hasattr(leaf, "dtype"):
      key = str(leaf.dtype)
      dtype_counts[key] = dtype_counts.get(key, 0) + 1
      total_bytes += leaf.nbytes
  print(f"\n[2/4] Checkpoint stats before conversion:")
  print(f"  Total arrays: {len(flat_leaves)}")
  print(f"  Total size: {total_bytes / (1024**3):.2f} GiB")
  for dtype, count in sorted(dtype_counts.items()):
    print(f"  {dtype}: {count} arrays")

  # --- Step 3: Cast float32 -> bfloat16 leaf by leaf ---
  print("\n[3/4] Casting float32 -> bfloat16...")
  start = time.time()
  converted = 0
  bf16_leaves = []
  for i, leaf in enumerate(flat_leaves):
    new_leaf = cast_leaf_to_bf16(leaf)
    if new_leaf is not leaf:
      converted += 1
    bf16_leaves.append(new_leaf)
    # Free the original fp32 leaf to reduce peak memory
    flat_leaves[i] = None
    if (i + 1) % 500 == 0:
      gc.collect()
      print(f"  Processed {i + 1}/{len(bf16_leaves)} arrays...")

  del flat_leaves
  gc.collect()

  state_bf16 = treedef.unflatten(bf16_leaves)
  del bf16_leaves
  gc.collect()

  elapsed = time.time() - start
  print(f"  Converted {converted} arrays in {elapsed:.1f}s")

  # Verify new dtypes
  flat_new, _ = jax.tree_util.tree_flatten(state_bf16)
  new_bytes = sum(leaf.nbytes for leaf in flat_new if hasattr(leaf, "nbytes"))
  print(f"  New total size: {new_bytes / (1024**3):.2f} GiB")
  del flat_new

  # --- Step 4: Save bf16 checkpoint ---
  print(f"\n[4/4] Saving bf16 checkpoint to {args.dest}...")
  start = time.time()

  # Wrap in TrainState to match the expected checkpoint structure
  state_to_save = train_state.TrainState(
      step=0,
      apply_fn=None,
      params=state_bf16["params"] if "params" in state_bf16 else state_bf16,
      tx=None,
      opt_state={},
  )
  del state_bf16, state
  gc.collect()

  # Create checkpoint manager
  dest_path = epath.Path(args.dest)
  dest_path.mkdir(exist_ok=True, parents=True)
  checkpoint_manager = ocp.CheckpointManager(
      dest_path,
      item_names=("items",),
      item_handlers={
          "items": ocp.PyTreeCheckpointHandler(
              save_concurrent_gb=96,
              use_ocdbt=args.use_ocdbt,
              use_zarr3=args.use_zarr3,
          )
      },
      options=ocp.CheckpointManagerOptions(
          create=True,
          enable_async_checkpointing=False,
          save_decision_policy=ocp.checkpoint_managers.save_decision_policy.FixedIntervalPolicy(interval=1),
      ),
  )

  checkpoint_args = ocp.args.PyTreeSave(
      item=state_to_save,
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(chunk_byte_size=2147483648), state_to_save),
      ocdbt_target_data_file_size=2147483648,
  )
  checkpoint_manager.save(0, args=ocp.args.Composite(items=checkpoint_args))
  checkpoint_manager.wait_until_finished()

  elapsed = time.time() - start
  print(f"  Saved in {elapsed / 60:.1f} min")
  print(f"\nDone! bf16 checkpoint at: {args.dest}/0/items")


if __name__ == "__main__":
  main()
