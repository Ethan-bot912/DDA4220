"""Opt-in fast read paths for the frozen ``XBDPairedDataset``.

``stage_b_damage/dataset.py`` is frozen because its SHA256 is recorded in the
published run metadata. Nothing here changes it: the sampler below only changes
the *order* in which samples are requested, so every ``__getitem__`` call still
goes through the same GDAL read path with the same window and the same
bilinear resampling. That keeps the decoded tensors bit-identical.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


class TileGroupedBatchSampler(Sampler[list[int]]):
    """Yield batches ordered by ``(event, tile_id)``.

    Each tile contributes ~60 buildings in the xBD manifests, and the baseline
    sampler visits them in random order, so the same 1024x1024 raster is
    re-decoded dozens of times per epoch. Grouping keeps a tile hot in GDAL's
    block cache for all of its buildings: tile order is shuffled per epoch and
    the buildings inside a tile are shuffled too, so the sampling stays random
    at tile level rather than being a fixed sequential pass.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self._groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self._groups[(row["event"], row["tile_id"])].append(index)
        self._keys = list(self._groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        keys = list(self._keys)
        if self.shuffle:
            order = torch.randperm(len(keys), generator=generator).tolist()
            keys = [keys[i] for i in order]
        pending: list[int] = []
        for key in keys:
            indices = list(self._groups[key])
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]
            pending.extend(indices)
            while len(pending) >= self.batch_size:
                yield pending[: self.batch_size]
                pending = pending[self.batch_size :]
        if pending:
            yield pending

    def __len__(self) -> int:
        total = sum(len(v) for v in self._groups.values())
        return (total + self.batch_size - 1) // self.batch_size


def distinct_tiles_for_batch(dataset, indices: list[int]) -> int:
    """Number of unique paired rasters a batch touches (locality proxy)."""
    tiles = {
        (dataset.rows[i]["pre_image"], dataset.rows[i]["post_image"]) for i in indices
    }
    return len(tiles)
