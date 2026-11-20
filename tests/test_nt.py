import pytest

from mujocobiobuzz.nt import controls


def test_wheel_powers_reach_their_motors_as_written() -> None:
    got = controls({"sim/wheels/frontLeft": 0.5, "sim/wheels/backRight": -0.25})
    assert got == {"Front Left": 0.5, "Back Right": -0.25}


def test_intake_speed_turns_the_intake_and_both_gecko_wheels() -> None:
    got = controls({"/intake/speed": 1.0})
    assert got == {"intake": 1.0, "left_intake_servo": 1.0, "right_intake_servo": 1.0}


def test_feeding_turns_the_windmill() -> None:
    assert controls({"/launcher/feeding": True}) == {"windmillServo": 1.0}
    assert controls({"/launcher/feeding": False}) == {"windmillServo": 0.0}


def test_launcher_target_is_its_share_of_free_speed() -> None:
    # 6000 rpm at 28 ticks a revolution is 2800 ticks a second.
    assert controls({"/launcher/target_tps": 1400.0})["launcher"] == pytest.approx(0.5)
    assert controls({"/launcher/target_tps": 9999.0})["launcher"] == 1.0


def test_a_topic_not_yet_seen_sets_nothing() -> None:
    assert controls({}) == {}
