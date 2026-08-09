from torch.utils.data import DataLoader

from .dataset import collate_fn


def create_dataloader(
    dataset,
    sampler=None,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=None,
    persistent_workers=None,
):
    """
    Creates a PyTorch DataLoader from a given dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to be wrapped by the DataLoader.

    sampler : torch.utils.data.Sampler, optional
        Sampling strategy used to draw samples from the dataset.
        If provided, shuffle must be False.

    batch_size : int
        Number of samples per batch.

    shuffle : bool, default=False
        Whether to shuffle the dataset at the beginning of each epoch.
        Cannot be used together with a sampler.

    num_workers : int, default=4
        Number of subprocesses used for data loading.

    pin_memory : bool, default=True
        If True, the data loader will copy tensors into CUDA pinned
        memory before returning them.

    Returns
    -------
    torch.utils.data.DataLoader
        DataLoader wrapping the provided dataset.
    """

    if sampler is not None and shuffle is True:
        raise ValueError("Cannot use both a sampler and shuffle=True.")

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            if prefetch_factor < 1:
                raise ValueError("prefetch_factor must be at least 1 when num_workers > 0.")
            loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    return loader
