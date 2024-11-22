import anvil.block
from anvil.block import Block
from anvil.legacy import LEGACY_ID_MAP

import json
from pathlib import Path
from typing import Dict, List
import torch


def block_to_rep(block: Block) -> str:
    if block.properties:
        props = str(block.properties)
    else:
        props = ''
    return f'{block.id}\t{props}'.replace('\'', '').strip()

def rep_to_block(rep: str) -> Block:
    s =  rep.split('\t')
    if len(s) > 1:
        block, props = s
        d = {}
        for prop in props.strip()[1:-1].split(','):
            s = prop.split(':')
            if len(s) == 1:
                pass
            k, v  = prop.split(':')
            k = k.strip()
            v = v.strip()
            d[k] = v
    else:
        block = s[0]
        d = {}
    return Block('minecraft', block, d)


def parse_variant(variant_string: str) -> Dict[str, str]:
    """
    Converts a variant string like "facing=east,in_wall=false,open=false"
    into a dictionary for easier processing.

    Parameters
    ----------
    variant_string: str
        The variant string from the JSON file.

    Returns
    -------
    Dict[str, str]
        The properties as a dictionary.
    """
    properties = {}
    if variant_string.strip() == "":
        return properties  # Handle empty variant strings

    for item in variant_string.split(','):
        if '=' not in item:
            continue  # Skip malformed properties without '='

        key, value = item.split('=', 1)
        properties[key.strip()] = value.strip()

    return properties


def parse_json_files(directory: str) -> List[Block]:
    """
    Parses all JSON files in the specified directory and converts their
    variants to instances of the Block class.

    Parameters
    ----------
    directory: str
        The directory containing the JSON files.

    Returns
    -------
    List[Block]
        A list of Block instances created from the JSON files.
    """
    blocks = []
    path = Path(directory)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Directory {directory} does not exist or is not a directory")

    for json_file in path.glob("*.json"):
        with open(json_file, 'r') as file:
            data = json.load(file)

            if 'variants' not in data:
                continue  # Skip files without variants

            for variant, variant_data in data['variants'].items():
                # Parse the properties from the variant string
                properties = parse_variant(variant)

                # Handle case where variant_data is a list (e.g., blue_concrete_powder.json)
                if isinstance(variant_data, list):
                    for model_data in variant_data:
                        model_name = model_data.get('model')
                        if not model_name or ':' not in model_name:
                            print('Discarding', model_name)
                            continue  # Skip invalid or malformed entries
                        namespace, block_id = model_name.split(':', 1)
                        block_id = block_id.split('/')[1]
                        block = Block(namespace=namespace, block_id=block_id,
                                      properties=properties)
                        blocks.append(block)
                else:
                    # Handle case where variant_data is a single dictionary (most cases)
                    model_name = variant_data.get('model')
                    if not model_name or ':' not in model_name:
                        print('Discarding', model_name)
                        continue  # Skip invalid or malformed entries
                    namespace, block_id = model_name.split(':', 1)
                    block_id = block_id.split('/')[1]
                    block = Block(namespace=namespace, block_id=block_id,
                                  properties=properties)
                    blocks.append(block)

    return blocks


class UnknownBlockError(Exception):
    """Raised when a world contains a block that is unknown to the current
    version of the game"""
    pass

class BlockIDMapper:
    """
    Maps blocks to a unique id and visa versa. Maintains and updates the
    mapping.
    """

    def __init__(self, mapping_path: Path):
        self._rep_to_id = {}
        self._mapping_path = mapping_path
        self._i = 0

        if mapping_path.exists():
            with open(mapping_path, 'r') as inf:
                self._i = 0
                for line in inf.readlines():
                    self._rep_to_id[line.strip()] = self._i
                    self._i += 1

        self._id_to_block = {v: rep_to_block(k) for k, v in self._rep_to_id.items()}

        self.make_legacy_maps()

    def make_legacy_maps(self):
        """
        Create tensors that allow for vectorized mapping of block_id - data
        pairs to their block ids.

        1. Extract tensor of block IDs
        2. Map IDs to IDs with space added between for data variants.
        3. Add data value to give unique id for each block - data pairing.
        4. Map unique IDs to known ids in registry.


                1      2            3      4
        1 - 0      1      1     0      1      234
        1 - 1      1      1     1      2      2333
        2 - 0      2      3     0      3      33
        3 - 0  ->  3  ->  4  +  0  ->  4  ->  66
        3 - 1      3      4     1      5      152
        3 - 2      3      4     2      6      1351
        4 - 0      4      7     0      7      652
        4 - 1      4      7     1      8      009
                       ^                   ^
        self.id_to_spaced_id      self.added_ids_to_final_id
        """
        # Extra data slots to account for unused block_id - data pairs.
        extra_slots = 200

        bids = []
        for lid in LEGACY_ID_MAP:
            bid, v = lid.split(':')
            bids.append((int(bid), int(v)))

        self.id_to_spaced_id = torch.empty(len(set(bids)), dtype=torch.int16)
        self.added_ids_to_final_id = torch.empty(len(LEGACY_ID_MAP) +
                                                 extra_slots, dtype=torch.int16)
        last = None
        i = 0
        j = 0
        for bid, data in sorted(bids):
            bid = bid
            data = data
            # Unused data slots.
            if bid != last:
                self.id_to_spaced_id[bid] = i
                j = 0
                last = bid

            while j < data:
                i += 1
                j += 1

            try:
                block = anvil.block.Block.from_numeric_id(bid, j)
                self.added_ids_to_final_id[i + j] = self.get_block_id(block)
            except KeyError:
                self.added_ids_to_final_id[i + j] = -1
            i += 1
            j += 1
        self.added_ids_to_final_id = self.added_ids_to_final_id[:i]

    def map_legacy(self, block_ids: torch.tensor, block_data: torch.tensor):
        """
        see self.make_legacy_maps
        """
        spaced = torch.gather(self.id_to_spaced_id, 0, block_ids)
        unique = spaced + block_data
        return torch.gather(self.added_ids_to_final_id, 0, unique)

    def get_block_id(self, block: Block):
        rep = block_to_rep(block)

        if rep not in self._rep_to_id:

            # Reload to see if any other threads have seen this block.
            with open(self._mapping_path, 'r') as inf:
                self._i = 0
                for line in inf.readlines():
                    self._rep_to_id[line.strip()] = self._i
                    self._i += 1
            # Recurse and try again.
            if rep in self._rep_to_id:
                return self._rep_to_id[rep]

            # Reload did not help. Create new block mapping.
            print(f'Creating new block! {rep}')
            with open(self._mapping_path, 'a') as out:
                out.write(rep + '\n')
            self._rep_to_id[rep] = self._i
            self._id_to_block[self._i] = block
            self._i += 1

        return self._rep_to_id[rep]

    def get_block(self, block_id: int) -> Block:
        return self._id_to_block[block_id]

    def get_max_id(self) -> int:
        return self._i - 1







