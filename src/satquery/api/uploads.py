"""Spooling multipart uploads to disk so rasterio can open them.

GDAL wants a file path (or a VSI handle); FastAPI hands us a stream. This module
is the bridge, and it enforces the byte budget while streaming rather than after,
so an oversized upload never fully materialises on disk.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import UploadFile
from pydantic import ValidationError

from satquery.ingest.errors import ImageTooLargeError, InvalidOptionsError, NoImagesError
from satquery.ingest.pipeline import SourceImage
from satquery.schemas.api import AnalyzeOptions

CHUNK_BYTES = 1 << 20


def parse_options(raw: str | None) -> AnalyzeOptions:
    """Parse the ``options`` multipart part. Absent is fine; malformed is a 400.

    This used to swallow both bad JSON and a failed validation and hand back the
    defaults, so ``{"seed": "abc"}`` or a mistyped key produced a silently
    different run with a 200. A client that sent options meant them.

    Raises:
        InvalidOptionsError: The part is not JSON, or not a valid AnalyzeOptions.
    """
    if not raw or not raw.strip():
        return AnalyzeOptions()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise InvalidOptionsError(f"options is not valid JSON: {error.msg}") from error
    try:
        return AnalyzeOptions.model_validate(payload)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'options'}: {item['msg']}"
            for item in error.errors()
        )
        raise InvalidOptionsError(problems) from error


async def persist_uploads_async(
    uploads: list[UploadFile], max_bytes: int
) -> tuple[Path, list[SourceImage]]:
    """:func:`persist_uploads` off the event loop.

    The copy is synchronous file IO over up to two 512 MB parts; done inline in
    an ``async def`` handler it stalls every other request's SSE heartbeat for
    its duration. The byte-budget check stays in the synchronous function,
    which is the one that is tested.
    """
    return await asyncio.to_thread(persist_uploads, uploads, max_bytes)


def persist_uploads(
    uploads: list[UploadFile], max_bytes: int
) -> tuple[Path, list[SourceImage]]:
    """Write each upload to a fresh temporary directory the **caller** owns.

    Split out of :func:`spooled_uploads` for ``POST /v1/jobs``, whose files have
    to outlive the request that carried them: the response returns as soon as the
    job is queued, but the background analysis reads the pixels long after. The
    caller is responsible for removing the returned directory.

    Args:
        uploads: The multipart file parts, in request order.
        max_bytes: Per-file byte budget.

    Returns:
        The directory holding the files, and one :class:`SourceImage` per
        upload in the same order.

    Raises:
        NoImagesError: The request carried no usable file parts.
        ImageTooLargeError: A file exceeded *max_bytes*.
    """
    if not uploads:
        raise NoImagesError()

    directory = Path(tempfile.mkdtemp(prefix="satquery-upload-"))
    try:
        images: list[SourceImage] = []
        for index, upload in enumerate(uploads):
            target = directory / f"img_{index}"
            written = 0
            upload.file.seek(0)
            with target.open("wb") as handle:
                while chunk := upload.file.read(CHUNK_BYTES):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ImageTooLargeError(
                            detail=(
                                f"Image {index + 1} exceeds the "
                                f"{max_bytes // (1 << 20)} MB upload limit."
                            ),
                            ref=f"img_{index}",
                        )
                    handle.write(chunk)
            if written == 0:
                raise NoImagesError()
            images.append(SourceImage(path=target, filename=upload.filename or target.name))
    except BaseException:
        # A rejected upload must not leave bytes on disk.
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory, images


@contextmanager
def spooled_uploads(uploads: list[UploadFile], max_bytes: int) -> Iterator[list[SourceImage]]:
    """Write each upload to a temporary directory, cleaned up on exit.

    The synchronous path (``/v1/analyze``, ``/v1/validate``): the files are
    needed only for the life of the request.

    Args:
        uploads: The multipart file parts, in request order.
        max_bytes: Per-file byte budget.

    Yields:
        One :class:`SourceImage` per upload, in the same order.

    Raises:
        NoImagesError: The request carried no usable file parts.
        ImageTooLargeError: A file exceeded *max_bytes*.
    """
    directory, images = persist_uploads(uploads, max_bytes)
    try:
        yield images
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@asynccontextmanager
async def spooled_uploads_async(
    uploads: list[UploadFile], max_bytes: int
) -> AsyncIterator[list[SourceImage]]:
    """:func:`spooled_uploads` for the async handlers: the copy runs off the loop."""
    directory, images = await persist_uploads_async(uploads, max_bytes)
    try:
        yield images
    finally:
        shutil.rmtree(directory, ignore_errors=True)
