import collections
import itertools
import os
import subprocess
from pathlib import Path
from typing import Set, List, Tuple, Callable, Iterable

import torch


def compress_world(world_path: Path, out_dir: Path) -> None:
    d = os.getcwd()
    os.chdir(world_path.parent)
    subprocess.run(['tar', '-czf', f'{out_dir / world_path.name}.tar.gz', world_path.name], check=True)
    os.chdir(d)

ChunkCoords = collections.namedtuple('ChunkCoords', ['x', 'z'])

class ChunkTrio:
    """
       X
     X Y

     A set of three chunks - given X, predict Y
    """

    def __init__(self,
                 get_chunk: Callable[[int, int], torch.tensor],
                 top: ChunkCoords,
                 left: ChunkCoords,
                 target: ChunkCoords,
                 ):
        self.top = top
        self.left = left
        self.target = target
        self.get_chunk = get_chunk

    def get_flip_dims(self) -> List[int]:
        """
        Get flips required to convert:

                 X
                XY

                XY
                 X

                 YX
                 X

                 X
                 YX

        into

                X
               XY   =   X, X, Y



        """
        on_top = self.top.z > self.target.z
        on_left = self.left.x < self.target.x
        flip_dims = []

        # Flip on Z axis
        if not on_top:
            flip_dims.append(2)

        # Flip on X axis
        if not on_left:
            flip_dims.append(0)

        return flip_dims

    def get_chunks_tensors(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        """
        coord_list = [self.top, self.left, self.target]
        chunks = [self.get_chunk(*coords) for coords in coord_list]
        flip_dims = self.get_flip_dims()

        if flip_dims:
            for i in range(len(chunks)):
                chunks[i] = chunks[i].flip(dims=flip_dims)

        return tuple(chunks)


class TorchWorld:
    """
    Allows the user to easily iterate over a pytorch-converted world stored in memory.
    """
    chunks: Set[ChunkCoords]
    regions: Set[Tuple[int, int]]

    def __init__(self, world_path: Path) -> None:
        self.world_path = world_path
        chunk_names = os.listdir(world_path)

        self.chunks = set()
        self.regions = set()
        for chunk in chunk_names:
            x, z = chunk[:-3].split('.')
            x, z = int(x), int(z)
            self._register_chunk(x, z)

    def _register_chunk(self, x: int, z: int):
        self.chunks.add((x, z))

        x, z = x // 32, z // 32

        if (x, z) not in self.regions:
            self.regions.add((x, z))

    def get_chunk(self, x: int, z: int) -> torch.tensor:
        if (x, z) in self.chunks:
            return torch.load(self.world_path / f'{x}.{z}.pt')
        else:
            return None

    def delete_chunk(self, x: int, z: int) -> torch.tensor:
        if (x, z) not in self.chunks:
            raise ValueError('Unknown chunk')
        self.chunks.remove((x, z))
        os.system(f'rm {str(self.world_path / f'{x}.{z}.pt')}')

    def set_chunk(self, tensor: torch.tensor, x: int, z: int) -> None:
        self._register_chunk(x, z)
        torch.save(tensor, self.world_path / f'{x}.{z}.pt')

    def get_region(self, x: int, z: int) -> List[List[torch.tensor]]:
        region = []
        x_off = x * 32
        z_off = z * 32

        for x in range(32):
            region.append([])
            for z in range(32):
                region[x].append(self.get_chunk(x + x_off, z + z_off))

        return region

    def _get_neighbor_tuples(self, chunk: ChunkCoords) \
        -> List[Tuple[ChunkCoords, ChunkCoords, ChunkCoords]]:
        """
        Scans <chunks> local area for neighbors.
        """
        valid_triples = []

        it = itertools.product([1, -1], [1, -1])
        for x_off, z_off in it:
            top = ChunkCoords(chunk.z + z_off, chunk.x)
            left = ChunkCoords(chunk.z, chunk.x + x_off)
            if top in self.chunks and left in self.chunks:
                valid_triples.append((top, left, chunk))

        return valid_triples


    def iterate_chunk_trios(self) \
            -> Iterable[Tuple[torch.tensor, torch.tensor, torch.tensor]]:

        for chunk in self.chunks:
            for chunks in self._get_neighbor_tuples(ChunkCoords(*chunk)):
                trio = ChunkTrio(self.get_chunk, *chunks)
                yield trio.get_chunks_tensors()






