import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import json
import os
import pickle
import platform
from typing import Any, List

import lmdb
from tqdm import tqdm
from torch.utils.data import DataLoader

from oscar.datasets.lmdb_readers import VQA_pt_Reader


# fmt: off
parser = argparse.ArgumentParser("Serialize a COCO Captions split to LMDB.")
parser.add_argument(
    "-d", "--data-root", default="./data/vqa",
    help="Path to the root directory of COCO dataset.",
)
parser.add_argument(
    "-s", "--split", choices=["train", "val", "test-dev-vqa2", "test-vqa2"], default="train",
    help="Which split to process, either `train` or `val`.",
)
parser.add_argument(
    "-b", "--batch-size", type=int, default=16,
    help="Batch size to process and serialize data. Set as per CPU memory.",
)
parser.add_argument(
    "-j", "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for data loading.",
)
parser.add_argument(
    "-o", "--output_path", default="data/vqa/lmdb/",
    help="Path to store the file containing serialized dataset.",
)


def collate_fn(instances: List[Any]):
    r"""Collate function for data loader to return list of instances as-is."""
    return instances


if __name__ == "__main__":

    _A = parser.parse_args()
    _A.output_file = _A.output_path + f"{_A.split}_img_frcnn_feats.lmdb"
    _A.output_index_file = _A.output_path + f"{_A.split}_img_frcnn_feats_idx_to_img_id.json"
    os.makedirs(os.path.dirname(_A.output_path), exist_ok=True)

    dloader = DataLoader(
        VQA_pt_Reader(_A.data_root, _A.split),

        batch_size=_A.batch_size,
        num_workers=_A.cpu_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    # Open an LMDB database.
    # Set a sufficiently large map size for LMDB (based on platform).
    map_size = 1099511627776 * 2 if platform.system() == "Linux" else 30000000000
    db = lmdb.open(
        _A.output_file, map_size=map_size, subdir=False, meminit=False, map_async=True
    )

    # Serialize each instance (as a dictionary). Use `pickle.dumps`. Key will
    # be an integer (cast as string) starting from `0`.
    INSTANCE_COUNTER: int = 0
    image_ids = []
    lmdb_idx_to_img_id = {}
    for idx, batch in enumerate(tqdm(dloader)):
        txn = db.begin(write=True)
        for instance in batch:
            instance = (instance["image_id"], instance["image"])
            txn.put(
                f"{INSTANCE_COUNTER}".encode("ascii"),
                pickle.dumps(instance, protocol=-1)
            )
            lmdb_idx_to_img_id[instance[0]] = INSTANCE_COUNTER
            INSTANCE_COUNTER += 1
            # image_ids += [instance[0]]
        txn.commit()

    # lmdb_idx_to_img_id = {i:img_id for i, img_id in enumerate(image_ids)}
    with open(_A.output_index_file, 'w') as outfile:
        json.dump(lmdb_idx_to_img_id, outfile)

    db.sync()
    db.close()
