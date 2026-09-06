"""Model training. Nothing here is imported by the serving path.

Training pulls in PyTorch Lightning and torchgeo, which the API has no business
depending on: a checkout that only serves reads a checkpoint bundle through
:mod:`satquery.training.cd.checkpoint` and never touches a datamodule or a
trainer. Install what this package needs with ``uv sync --extra cd``.
"""
