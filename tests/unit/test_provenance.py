"""Generated answers must not claim a source or a resolution the input does not have."""

from __future__ import annotations

import pytest

from satquery.models.prompts.provenance import correct_provenance, short_sensor
from satquery.schemas.manifest import InputManifest


def _manifest(sensor: str | None, gsd: float | None) -> InputManifest:
    base = InputManifest.model_construct(sensor_guess=sensor, gsd_m=gsd)
    return base


S2 = _manifest("Sentinel-2 (BigEarthNet-v2, 12-band)", 10.0)
UNKNOWN_VHR = _manifest(None, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "The image, sourced from GoogleEarth, shows a dense urban area.",
            "The image from Sentinel-2 shows a dense urban area.",
        ),
        (
            "This high-resolution image from GoogleEarth shows a river.",
            "This image from Sentinel-2 shows a river.",
        ),
        (
            "The image, sourced from GoogleEarth, features a high-resolution view of an airport.",
            "The image from Sentinel-2 features a view of an airport.",
        ),
        ("Water covers 3.06% of the scene.", "Water covers 3.06% of the scene."),
    ],
)
def test_sentinel_answers_lose_the_false_source_and_resolution(text: str, expected: str) -> None:
    assert correct_provenance(text, [S2]) == expected


def test_unknown_sensor_drops_the_source_but_keeps_unmeasured_resolution() -> None:
    text = "The image, sourced from GoogleEarth, shows a high-resolution view of houses."
    assert correct_provenance(text, [UNKNOWN_VHR]) == (
        "The image shows a high-resolution view of houses."
    )


def test_short_sensor() -> None:
    assert short_sensor(S2) == "Sentinel-2"
    assert short_sensor(UNKNOWN_VHR) is None
    assert short_sensor(None) is None


def test_generic_sensor_is_not_named_as_a_source() -> None:
    generic = _manifest("Generic RGB", None)
    assert short_sensor(generic) is None
    assert correct_provenance("The image, sourced from GoogleEarth, shows a port.", [generic]) == (
        "The image shows a port."
    )


def test_platform_names_are_not_numeric_claims() -> None:
    from satquery.evidence.citation_validator import validate
    from satquery.evidence.fact_sheet import FactSheet

    result = validate("This image from Sentinel-2 and Landsat 8 shows a river.", FactSheet())
    assert result.uncited_numeric_spans == []
