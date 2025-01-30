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

    treatment_time = np.random.choice(
        list(EVALUATION_RANGE) + [None], NUM_PATIENTS
    )

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
    evaluations = []

    for i in range(NUM_PATIENTS):
        for t in EVALUATION_RANGE:
            if t == 0:
                evaluation = patients.loc[i].copy()
            else:
                evaluation = evaluations[-1].copy()

            if (
                evaluation["treatment time"] is not None
                and t > evaluation["treatment time"]
            ):
                pain_std = (-0.25, 1.0)
                urgency_std = (-0.25, 0.5)
                nocturnal_std = (-0.25, 0.5)
            else:
                pain_std = (0.25, 1.0)
                urgency_std = (0.25, 0.5)
                nocturnal_std = (0.25, 0.5)

            evaluation["pain"] = np.clip(
                evaluation["pain"] + np.random.normal(pain_std[0], pain_std[1]),
                0,
                9,
            ).astype(int)

            evaluation["urgency"] = np.clip(
                evaluation["urgency"]
                + np.random.normal(urgency_std[0], urgency_std[1]),
                0,
                9,
            ).astype(int)

            evaluation["nocturnal frequency"] = np.clip(
                evaluation["nocturnal frequency"]
                + np.random.normal(nocturnal_std[0], nocturnal_std[1]),
                0,
                20,
            ).astype(int)

            evaluation["time"] = t
            evaluations.append(evaluation)

    return pd.DataFrame(evaluations)


def simulate_risk_set(patients_evaluations: pd.DataFrame) -> pd.DataFrame:
    risk_set = {}

    for t in patients_evaluations["treatment time"].dropna().unique():
        treated = patients_evaluations[
            (patients_evaluations["time"] == t)
            & (patients_evaluations["treatment time"] == t)
        ]

        untreated = patients_evaluations[
            (patients_evaluations["time"] == t)
            & (patients_evaluations["treatment time"] != t)
        ]

        risk_set[t.astype(int)] = (treated, untreated)

    return pd.DataFrame(risk_set)


# Exported data
patients_entry = simulate_entry()

patients_evaluations = simulate_evaluations(patients_entry)

patients_risk_set = simulate_risk_set(patients_evaluations)


del simulate_entry, simulate_evaluations
