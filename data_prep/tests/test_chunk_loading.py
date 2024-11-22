import itertools
import os
import unittest
from pathlib import Path
import torch
import anvil

from data_prep.world_parsing.block_id_mapping import BlockIDMapper
from data_prep.world_parsing.convert_to_tensors import TensorConverter

TESTS_DIR = Path(__file__).parent
NUM_CORES = 4

WORLDS = TESTS_DIR / 'test_fixtures'

TINY_NEW = WORLDS / 'tiny_latest'
TINY_OLD = WORLDS / 'tiny_old'
MAP_PATH = TESTS_DIR.parent / 'fixtures' / 'all_blocks.tsv'

mapper = BlockIDMapper(MAP_PATH)
tensor_converter = TensorConverter(mapper, num_cores=NUM_CORES)


class TestChunkLoading(unittest.TestCase):

    # RIP GPT version 2024-2024
    chunk_loder_methods = ['vectorized', 'org']

    def test_all_chunkloaders_latest(self):
        world = TINY_NEW
        regions = world / 'region'

        for region_name in os.listdir(regions):
            region = anvil.Region.from_file(str(regions / region_name))

            sols = {}
            for v in self.chunk_loder_methods:
                sols[v] = tensor_converter.region_to_tensors(region, version=v)

            for x, z in itertools.product(range(32), range(32)):
                for m1, m2 in itertools.combinations(self.chunk_loder_methods, 2):

                    if sols[m1][x][z] is None and sols[m2][x][z] is None:
                        continue

                    msg = (f'{m1} != {m2} at chunk {x}, {z} in {region_name}'
                           f'\n{sols[m1][x][z]}'
                           f'\n{sols[m2][x][z]}')
                    self.assertTrue(torch.all(sols[m1][x][z] == sols[m2][x][z]), msg)

    def test_all_chunkloaders_old(self):
        world = TINY_OLD
        regions = world / 'region'

        for region_name in os.listdir(regions):
            region = anvil.Region.from_file(str(regions / region_name))

            sols = {}
            for v in self.chunk_loder_methods:
                sols[v] = tensor_converter.region_to_tensors(region, version=v)

            for x, z in itertools.product(range(32), range(32)):
                for m1, m2 in itertools.combinations(self.chunk_loder_methods,
                                                     2):

                    if sols[m1][x][z] is None and sols[m2][x][z] is None:
                        continue

                    msg = (f'{m1} != {m2} at chunk {x}, {z} in {region_name}'
                           f'\n{sols[m1][x][z]}'
                           f'\n{sols[m2][x][z]}')

                    try:
                        equal = torch.all(sols[m1][x][z] == sols[m2][x][z])
                    except RuntimeError as err:
                        err.add_note(msg)
                        raise err

                    if not equal:
                        pass
                    self.assertTrue(equal, msg)


if __name__ == '__main__':
    unittest.main()
