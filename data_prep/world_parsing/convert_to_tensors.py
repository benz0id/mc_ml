import itertools
import os
import queue
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Union, Callable
import anvil
from anvil.errors import ChunkNotFound

import torch
from anvil.chunk import Chunk, _section_height_range
from anvil.block import Block
from anvil.region import Region
from anvil.versions import VERSION_17w47a, VERSION_20w17a
from bitarray import bitarray
from bitarray.util import ba2int
from nbt import nbt
from tqdm import tqdm

from data_prep.world_parsing.block_id_mapping import BlockIDMapper
from data_prep.world_parsing.torch_world import TorchWorld

IMP_VERSION = 3955


def _nibble_array(byte_array) -> torch.tensor:
    uint8_array = torch.frombuffer(byte_array, dtype=torch.uint8)
    # Extract the upper nibbles (shift right by 4 bits)
    upper_nibbles = uint8_array >> 4

    # Extract the lower nibbles (bitwise AND with 0b1111 to keep the lower 4 bits)
    lower_nibbles = uint8_array & 0b1111

    # Stack the nibbles into a new array: first the upper, then the lower
    return torch.column_stack((upper_nibbles, lower_nibbles)).flatten()

class TensorConverter:
    """
    Converts minecraft worlds to tensor objects.
    """

    def __init__(
            self,
            block_mapper: BlockIDMapper,
            verbose: bool = True,
            num_cores: int = 1):
        self._block_mapper = block_mapper
        self.verbose = verbose
        self.num_cores =num_cores

    # ===== Minecraft -> Pytorch ======

    def region_to_tensors(self, region: Union[Region, Path],
                          section_dims: int = 32,
                          verbose: bool = True,
                          version: str = 'org') \
            -> List[List[Union[None, torch.tensor]]]:
        """
        Converts all chunks in the given region into tensors.

        :param region: A region in a minecraft world.

        :return: An array of tensors indexable by [x, z] coords of the chunk.
                Chunks that are not yet generated are marked by None.
        """
        if isinstance(region, Path):
            region = anvil.region.Region.from_file(str(region))

        chunks = []
        it = itertools.product(range(section_dims), range(section_dims))
        if verbose:
            it = tqdm(it)
            it.total = section_dims ** 2

        for x, z in it:
            if x >= len(chunks):
                chunks.append([])

            try:
                chunk = region.get_chunk(x, z)
            except ChunkNotFound:
                chunk = None

            if chunk is not None:
                chunk = self.chunk_to_tensor(chunk, version)
            chunks[x].append(chunk)
        return chunks

    def chunk_to_tensor(self, chunk: Chunk, version: str = 'org') -> torch.tensor:
        """
        Converts the given chunk into a pytorch tensor.

        :param chunk: A chunk in minecraft world.

        :return: a tensor with shape [16, 64 + 320, 16]
        """
        if version == 'org':
            stream = chunk.stream_chunk()
        elif version == 'vectorized':
            return self._get_chunk_vectorized(chunk)
        else:
            raise ValueError('Did not get valid algorithm version.')

        i = -1
        chunk_tensor = torch.zeros(98304, dtype=torch.int16)

        if chunk.version < VERSION_17w47a:
            for block in stream:
                block = block.convert()
                block_id = self._block_mapper.get_block_id(block)
                i += 1
                chunk_tensor[i] = block_id
        else:
            for block in stream:
                block_id = self._block_mapper.get_block_id(block)
                i += 1
                chunk_tensor[i] = block_id

        air = Block('minecraft', 'air')
        air_id = self._block_mapper.get_block_id(air)
        while i < 98304:
            chunk_tensor[i] = air_id
            i += 1
        chunk_tensor = chunk_tensor.reshape((384, 16, 16)).permute(2, 0, 1)
        return chunk_tensor

    def convert_world_to_torch(self,
                               world_path: Path,
                               out_path: Path,
                               overwrite: bool = True,
                               version: str = 'vectorized') -> TorchWorld:
        """
        Converts all chunks in a given world into sparse pytorch tensors.

        :param world_path: Path to the world file.
        :param out_path: Path in which to store converted chunks.
        :param overwrite: Overwrite existing files.
        """

        out_path.mkdir(exist_ok=True)

        regions_dir = world_path / 'region'
        regions = os.listdir(regions_dir)

        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = []
            for region in regions:
                futures.append(
                    executor.submit(self._store_region,
                                    regions_dir / region,
                                    out_path,
                                    overwrite,
                                    version))
            if self.verbose:
                futures = tqdm(futures)

            for future in futures:
                future.result()
        return TorchWorld(out_path)

    def _store_region(self, region_path: Path, out: Path,
                      overwrite: bool, version: str):
        """
        Helper for the above.
        """
        region = anvil.region.Region.from_file(str(region_path))

        x, z = region_path.name[2:-4].split('.')
        x, z = int(x), int(z)
        tensors = self.region_to_tensors(region, verbose=False, version=version)

        for i in range(len(tensors)):
            for j in range(len(tensors)):
                if tensors[i][j] is None:
                    continue

                name = f'{x * 32 + i}.{z * 32 + j}.pt'
                chunk_path = out / name
                if chunk_path.exists() and not overwrite:
                    continue

                torch.save(tensors[i][j], out / name)

        del tensors
        del region


    # ====== Pytorch -> Minecraft ======

    def tensors_to_region(self,
                          tensors: List[List[torch.tensor]],
                          version: int,
                          x: int = 0,
                          z: int = 0,
                          verbose: bool = True) \
        -> anvil.empty_region.EmptyRegion:
        assert tensors

        region = anvil.empty_region.EmptyRegion(x, z)
        it = itertools.product(range(len(tensors)), range(len(tensors[0])))
        if verbose:
            it = tqdm(it)
            it.total = len(tensors) * len(tensors[0])

        for chunk_x, chunk_z in it:
            tensor = tensors[chunk_x][chunk_z]
            if tensor is not None:
                chunk = self.tensor_to_chunk(tensor,
                                             version,
                                             x = chunk_x + 32 * x,
                                             z = chunk_z + 32 * z)
                region.add_chunk(chunk)
        return region

    def tensor_to_chunk(self,
                        tensor: torch.tensor,
                        version: int,
                        x: int = 0,
                        z: int = 0) \
            -> anvil.empty_chunk.EmptyChunk:

        chunk = anvil.empty_chunk.EmptyChunk(x, z, version)

        it = itertools.product(range(16), range(-64, 320), range(16))
        for x, y, z in it:
            block_id = tensor[x, y + 64, z].item()
            block = self._block_mapper.get_block(block_id)
            chunk.set_block(block, x, y, z)
        return chunk

    def torch_world_to_world(self,
                             torch_world: TorchWorld,
                             out_world: Path):
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            futures = []

            for region in torch_world.regions:
                #self._write_out_region(torch_world, out_world, *region)
                futures.append(executor.submit(self._write_out_region,
                                               torch_world, out_world, *region))

            if self.verbose:
                futures = tqdm(futures)

            for future in futures:
                future.result()

    def _write_out_region(self,
                          torch_world: TorchWorld,
                          out_path: Path,
                          x: int,
                          z: int):
        tensors = torch_world.get_region(x, z)
        region = self.tensors_to_region(tensors, IMP_VERSION)
        region.save(str(out_path / 'region' / f'r.{x}.{z}.mca'))
        
    # === Low Level Helpers ===

    def _get_chunk_vectorized(self, chunk: Chunk) -> torch.tensor:
        sections = []
        for section in _section_height_range(chunk.version):
            sections.append(
                self._get_blocks_vectorized(chunk, section))
        return torch.cat(sections, dim=1)


    def _get_blocks_vectorized(self, chunk: Chunk,
                               section: Union[int, nbt.TAG_Compound] = None
                              ) -> torch.tensor:

        if section is None or isinstance(section, int):
            section = chunk.get_section(section or 0)

        if chunk.version < VERSION_17w47a:
            if section is None or 'Blocks' not in section:
                block_ids = (torch.zeros(4096, dtype=torch.uint8)
                             .reshape((16, 16, 16))
                             .permute(2, 0, 1))
                return block_ids

            block_ids = torch.frombuffer(bytearray(section['Blocks']), dtype=torch.int8)
            if 'Add' in section:
                block_ids += _nibble_array(bytearray(section['Add']))
            data = _nibble_array(bytes(section['Data']))

            ids = self._block_mapper.map_legacy(block_ids.to(torch.int64),
                                  data.to(torch.int64))
            return ids.reshape((16, 16, 16)).permute(2, 0, 1)

        if section is None or 'BlockStates' not in section:
            return torch.zeros((16, 16, 16), dtype=torch.int16)

        states = section['BlockStates'].value
        palette = section['Palette']


        ids = torch.zeros(len(palette), dtype=torch.int16)
        for i, p in enumerate(palette):
            ids[i] = self._block_mapper.get_block_id(Block.from_palette(p))


        bits_per_val = max((len(palette) - 1).bit_length(), 4)
        stretches = chunk.version < VERSION_20w17a

        if not stretches:
            cutoff = 64 - (64 // bits_per_val) * bits_per_val
        else:
            cutoff = 64

        i = 0
        bits = bitarray(len(section['BlockStates'].value) * 64)
        # Format into stream of bits with the packed size rep.
        for num in section['BlockStates'].value:

            if num < 0:
                num += 2 ** 64
            b = bitarray()
            b.frombytes(num.to_bytes(8, byteorder='big', signed=False))
            bits[i: i + cutoff] = b[cutoff - 1::-1]
            i += cutoff
        bits = bits[:bits_per_val * 4096]

        # Unpack bits into tensor.
        unpacked_array = torch.empty(4096, dtype=torch.int16)
        for ind, i in enumerate(range(0, 4096 * bits_per_val, bits_per_val)):
            val = bits[i: i + bits_per_val][::-1]
            unpacked_array[ind] = ba2int(val)

        # Map tensor to block ids using the keys.
        unpacked_array = torch.gather(ids, 0, unpacked_array.to(torch.int64))
        # unpacked_array[drop] = 0
        return unpacked_array.reshape((16, 16, 16)).permute(2, 0, 1)
    
    
