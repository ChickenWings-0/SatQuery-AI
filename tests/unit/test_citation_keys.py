"""Key-aware citation matching.

``validate()`` resolved every number by *value*: any fact within tolerance was a
citation. The ``[key]`` the model attaches — the thing the adapter is trained to
emit, and the stronger signal — was stripped before validation ever ran. So
``0.75 [sar_backscatter_analyzer.sigma0_vv_db_mean]`` passed if any unitless
fact was near 0.75, and landed in the trace as grounded under the wrong key.
"""

from __future__ import annotations

import pytest

from satquery.evidence.citation_validator import validate
from satquery.evidence.fact_sheet import FactSheet
from satquery.models.prompts.builder import strip_citation_markers
from satquery.training.corpus_builder import fact_sheet_from

NDVI = "spectral_index_analyzer.ndvi_mean"
VV = "sar_backscatter_analyzer.sigma0_vv_db_mean"
PCT = "change_statistics.changed_area_pct"


@pytest.fixture
def sheet() -> FactSheet:
    return fact_sheet_from({NDVI: 0.75, VV: -8.4, PCT: 7.4})


def _check(text: str, sheet: FactSheet) -> tuple[list[str], list[str], list[str]]:
    marked = strip_citation_markers(text, sheet)
    result = validate(marked.text, sheet, markers=marked.markers)
    return (
        [citation.source for citation in result.citations],
        result.uncited_numeric_spans,
        result.uncited_reasons,
    )


def test_a_right_number_with_the_wrong_key_is_uncited(sheet: FactSheet) -> None:
    """The audit's case. 0.75 *is* the NDVI mean; the model attributed it to VV."""
    cited, uncited, reasons = _check(f"The mean NDVI is 0.75 [{VV}].", sheet)
    assert cited == []
    assert uncited == ["0.75"]
    assert reasons == ["KEY_VALUE_MISMATCH"]


def test_a_right_number_with_the_right_key_cites_that_key(sheet: FactSheet) -> None:
    cited, uncited, _ = _check(f"The mean NDVI is 0.75 [{NDVI}].", sheet)
    assert cited == ["step:1/scalars.ndvi_mean"]
    assert uncited == []


def test_an_unknown_key_is_uncited_even_when_the_value_matches(sheet: FactSheet) -> None:
    """Inventing a key is inventing the measurement behind it."""
    cited, uncited, reasons = _check("Mean NDVI 0.75 [spectral_index_analyzer.made_up].", sheet)
    assert cited == []
    assert uncited == ["0.75"]
    assert reasons == ["UNKNOWN_KEY"]


def test_a_bare_number_still_resolves_by_value(sheet: FactSheet) -> None:
    """No marker, no key to check: the value search is the fallback, unchanged."""
    cited, uncited, _ = _check("Change covers 7.4 % of the scene.", sheet)
    assert cited == ["step:1/scalars.changed_area_pct"]
    assert uncited == []


def test_a_marker_binds_to_the_span_immediately_before_it_not_an_earlier_one(
    sheet: FactSheet,
) -> None:
    """Two numbers, one marker: the marker belongs to its neighbour."""
    cited, uncited, reasons = _check(f"NDVI 0.75 and VV of -8.4 dB [{VV}].", sheet)
    assert cited == ["step:1/scalars.ndvi_mean", "step:1/scalars.sigma0_vv_db_mean"]
    assert uncited == []
    assert reasons == []
    # And the same marker cannot be claimed twice.
    cited, uncited, reasons = _check(f"Values 0.75 and 0.75 [{NDVI}].", sheet)
    assert uncited == []


def test_a_marker_too_far_from_the_number_does_not_bind(sheet: FactSheet) -> None:
    """A whole clause between number and key is not an attribution."""
    _, uncited, reasons = _check(f"NDVI reached 0.75, which is high for the season [{VV}].", sheet)
    # The number resolves by value; the far marker is just a stray key.
    assert uncited == []
    assert reasons == []


def test_marker_offsets_survive_whitespace_collapse(sheet: FactSheet) -> None:
    """Adjacent markers and double spaces must not shift later offsets."""
    text = f"  NDVI 0.75 [{NDVI}] [{NDVI}]  and  VV -8.4 dB [{VV}] ."
    marked = strip_citation_markers(text, sheet)
    assert marked.text == "NDVI 0.75 and VV -8.4 dB ."
    for offset, _ in marked.markers:
        assert 0 <= offset <= len(marked.text)
    cited, uncited, _ = _check(text, sheet)
    assert cited == ["step:1/scalars.ndvi_mean", "step:1/scalars.sigma0_vv_db_mean"]
    assert uncited == []


def test_templated_answers_keep_the_value_search(sheet: FactSheet) -> None:
    """``markers=None`` is exactly the old behaviour; nothing templated changes."""
    result = validate("Mean NDVI 0.75 and 42% water.", sheet, markers=None)
    assert [c.source for c in result.citations] == ["step:1/scalars.ndvi_mean"]
    assert result.uncited_numeric_spans == ["42%"]
    assert result.uncited_reasons == ["NO_MATCH"]
