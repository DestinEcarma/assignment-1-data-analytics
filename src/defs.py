import numpy as np
import pandas as pd

np.random.seed(69)

NUM_PATIENTS = 400
EVALUATION_INTERVAL = 3  # Per Month
EVALUATION_DURATION = 4  # Years
EVALUATION_RANGE = np.arange(0, EVALUATION_DURATION * 12, EVALUATION_INTERVAL)


def simulate_entry() -> pd.DataFrame:
    gender = np.random.choice(["M", "F"], NUM_PATIENTS)

    pain_scores = np.clip(np.random.normal(5, 3.0, NUM_PATIENTS), 0, 9).astype(
        int
    )

    urgency_scores = np.clip(
        np.random.normal(6, 2.0, NUM_PATIENTS),
        0,
        9,
    ).astype(int)

    nocturnal_frequency = np.clip(
        np.random.normal(3, 1.5, NUM_PATIENTS), 0, 20
    ).astype(int)

    # 0 means never treated
    treatment_time = np.random.choice(EVALUATION_RANGE, NUM_PATIENTS)

    return pd.DataFrame(
        {
            "id": np.arange(NUM_PATIENTS),
            "gender": gender,
            "pain": pain_scores,
            "urgency": urgency_scores,
            "nocturnal frequency": nocturnal_frequency,
            "treatment time": treatment_time,
        }
    )


def simulate_evaluations(patients: pd.DataFrame) -> [pd.DataFrame]:
    treatment = [[] for _ in EVALUATION_RANGE]

    for i in range(NUM_PATIENTS):
        for t in EVALUATION_RANGE:
            if t == 0:
                evaluation = patients.loc[i].copy()
            else:
                evaluation = treatment[t // EVALUATION_INTERVAL - 1][i].copy()

            if t > evaluation["treatment time"]:
                evaluation["pain"] = np.clip(
                    evaluation["pain"] - np.random.normal(0, 1.0), 0, 9
                ).astype(int)

                evaluation["urgency"] = np.clip(
                    evaluation["urgency"] - np.random.normal(0, 0.5), 0, 9
                ).astype(int)

                evaluation["nocturnal frequency"] = np.clip(
                    evaluation["nocturnal frequency"]
                    - np.random.normal(0, 0.5),
                    0,
                    20,
                ).astype(int)

            treatment[t // EVALUATION_INTERVAL].append(evaluation)

    return [pd.DataFrame(t) for t in treatment]


# Exported data
patients_entry = simulate_entry()

patients_evaluations = simulate_evaluations(patients_entry)


del simulate_entry, simulate_evaluations
