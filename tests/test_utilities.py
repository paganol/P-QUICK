from pathlib import Path

from pquick.utilities import filter_pointing_files_by_mission_length, parse_mission_length


def test_parse_mission_length_named_ranges():
    assert parse_mission_length("full") == (91, 974)
    assert parse_mission_length("survey1") == (91, 270)
    assert parse_mission_length("survey2") == (271, 456)
    assert parse_mission_length("survey3") == (457, 636)
    assert parse_mission_length("survey4") == (637, 807)
    assert parse_mission_length("survey5") == (808, 974)


def test_parse_mission_length_explicit_range():
    assert parse_mission_length("91-99") == (91, 99)
    assert parse_mission_length("OD91-OD99") == (91, 99)


def test_filter_pointing_files_by_mission_length():
    files = [
        Path("inputs/pointings/processed_od_0090.npz"),
        Path("inputs/pointings/processed_od_0091.npz"),
        Path("inputs/pointings/processed_od_0099.npz"),
        Path("inputs/pointings/processed_od_0100.npz"),
    ]
    out = filter_pointing_files_by_mission_length(files, "91-99")
    assert [p.name for p in out] == [
        "processed_od_0091.npz",
        "processed_od_0099.npz",
    ]
