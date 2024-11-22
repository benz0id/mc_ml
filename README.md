# Applying Diffusion Models for Minecraft World Generation

Currently implements efficient minecraft world -> tensor conversion and tensor manipulation, to allow for the constrution of training sets from minecraft worlds.

## Installation 

Requires python 3.12

```bash
git clone https://github.com/benz0id/mc_ml
pip install -r mc_ml/requirements.txt
```

## Convert Minecraft Worlds Into Tensor Objects

Convert all chunks in a world into a tensor objects.

```python
from data_prep.world_parsing.block_id_mapping import BlockIDMapper
from data_prep.world_parsing.convert_to_tensors import TensorConverter

BLOCK_MAPPING = Path("data_prep/fixtures/all_blocks.tsv")
NUM_CORES = 8
MC_WORLD = <your world directory here>
PT_WORLD = <your output tensor directory here>

# Path to block dictionary.
tokenizer = BlockIDMapper(BLOCK_MAPPING)
tensor_converter = TensorConverter(tokenizer, num_cores=NUM_CORES)

# Convert world to set of tensors.
tensor_converter.convert_world_to_torch(MC_WORLD, PT_WORLD, overwrite=False, version='vectorized')
```

### Important Notes
- Uses a supplied block -> id mapping, adding to it as blocks are encountered.
- This works on all known versions of minecraft at time of writing (22-10-2024).
- Blocks with different states (e.g. different heights of water) are given different ids.

## Manipulate Tensor Worlds

The TorchWorld class allows for easy editing of tensor world files.

```python
from from data_prep.world_parsing.torch_world import TorchWorld

world = TorchWorld(PT_WORLD)
OUT_MOD_WORLD = <path to modified output world>

# Set a chunk to all air.
chunk = world.get_chunk(x=1, z=1)
chunk.zero_() # assumes 0 -> air in your dictionary.
world.set_chunk(chunk, x=1, z=1)

# Convert the world back into a minecraft world.
tensor_converter.torch_world_to_world(world, OUT_MOD_WORLD)
```

## Diffusion Model Training and Evaluation
coming soon!
