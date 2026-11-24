"""Detect-all grounding: tolerant JSON parsing, tiling and duplicate merging."""

from __future__ import annotations

from satquery.models.prompts.box_format import NormalisedBox, parse
from satquery.tools.text_grounding import (
    _to_frame,
    detect_grid,
    merge_boxes,
    target_noun,
    tile_origins,
)


def test_a_truncated_json_list_still_yields_its_complete_entries() -> None:
    text = (
        '```json\n[\n\t{"bbox_2d": [10, 20, 30, 40], "label": "aircraft"},\n'
        '\t{"bbox_2d": [50, 60, 70, 80], "label": "aircraft"},\n\t{"bbox_2d": [90, 9'
    )
    boxes = parse(text)
    assert [(b.x_min, b.y_min, b.x_max, b.y_max) for b in boxes] == [
        (10, 20, 30, 40),
        (50, 60, 70, 80),
    ]


def test_grid_grows_with_the_source() -> None:
    assert detect_grid(936, 791) == 1
    assert detect_grid(1565, 1492) == 2
    assert detect_grid(2812, 2429) == 3
    assert detect_grid(None, None) == 1


def test_tiles_cover_the_canvas_and_overlap() -> None:
    tiles = tile_origins(1344, 3)
    assert len(tiles) == 9
    xs = sorted({x for x, _, _ in tiles})
    extent = tiles[0][2]
    assert xs[0] == 0 and xs[-1] + extent == 1344
    assert xs[1] < xs[0] + extent  # neighbours overlap
    assert tile_origins(448, 1) == [(0, 0, 448)]


def test_a_tile_box_maps_back_to_the_whole_frame() -> None:
    box = NormalisedBox(0, 0, 1000, 1000, "ship")
    mapped = _to_frame(box, x0=672, y0=0, extent=672, side=1344, noun="ship")
    assert (mapped.x_min, mapped.y_min, mapped.x_max, mapped.y_max) == (500, 0, 1000, 500)


def test_merge_drops_an_object_seen_twice_at_a_tile_edge() -> None:
    whole = NormalisedBox(100, 100, 200, 200, "aircraft")
    cut = NormalisedBox(150, 100, 200, 200, "aircraft")  # the half the other tile saw
    elsewhere = NormalisedBox(600, 600, 650, 650, "aircraft")
    assert merge_boxes([cut, whole, elsewhere]) == [whole, elsewhere]


def test_target_noun_prefers_the_object_slot() -> None:
    assert target_noun({"object_class": "aircraft"}, "") == "aircraft"
    assert target_noun({"target_class": "built_up"}, "") == "building"
    assert target_noun({"referring_expression": "the ships"}, "") == "ship"
    assert target_noun({}, "count them") == "object"
