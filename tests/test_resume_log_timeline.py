import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_training_data import choose_duplicate_segments


def reward_rows(run_id, start_episode, episodes):
    return pd.DataFrame(
        {
            "Run_ID": [run_id] * len(episodes),
            "Run_Start_Episode": [start_episode] * len(episodes),
            "Episode": episodes,
            "Timestep": [0] * len(episodes),
            "Rewards": [1.0] * len(episodes),
            "Cumulative_Rewards": [1.0] * len(episodes),
        }
    )


def test_timeline_policy_does_not_keep_old_future_after_resume():
    original = reward_rows("20260512_120000_000000", 0, range(10))
    resumed = reward_rows("20260512_130000_000000", 8, [8, 9])

    cleaned, duplicates = choose_duplicate_segments(
        pd.concat([original, resumed], ignore_index=True),
        policy="timeline",
    )

    selected = cleaned[["Run_ID", "Episode"]].drop_duplicates()
    assert selected[selected["Episode"] < 8]["Run_ID"].eq(original["Run_ID"].iloc[0]).all()
    assert selected[selected["Episode"] >= 8]["Run_ID"].eq(resumed["Run_ID"].iloc[0]).all()
    assert set(cleaned["Episode"]) == set(range(10))
    assert set(duplicates["Episode"]) == {8, 9}


def test_timeline_policy_keeps_raw_logs_safe_when_resume_from_zero_is_wrong():
    original = reward_rows("20260512_120000_000000", 0, range(100))
    accidental_restart = reward_rows("20260512_130000_000000", 0, [0])
    raw = pd.concat([original, accidental_restart], ignore_index=True)

    cleaned, _ = choose_duplicate_segments(raw, policy="timeline")

    assert len(raw) == 101
    assert cleaned["Episode"].tolist() == [0]
    assert cleaned["Run_ID"].iloc[0] == "20260512_130000_000000"
