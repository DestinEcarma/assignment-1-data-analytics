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

    return pd.DataFrame(
        {
            "id": np.arange(NUM_PATIENTS),
            "gender": gender,
            "pain": pain_scores,
            "urgency": urgency_scores,
            "nocturnal frequency": nocturnal_frequency,
        }
    )


def simulate_evaluations(patients_entry: pd.DataFrame) -> list[pd.DataFrame]:
    evaluations = []

    for i in range(NUM_PATIENTS):
        treatment_time = np.random.choice(
            list(EVALUATION_RANGE)
            + [None for _ in range(EVALUATION_INTERVAL)],
        )

        for t in EVALUATION_RANGE:
            evaluation = patients_entry.loc[i].copy()

            if treatment_time is not None and t >= treatment_time:
                if t < treatment_time + EVALUATION_INTERVAL:
                    pain_ls = (4.75, 3.0)
                    urgency_ls = (5.75, 2.0)
                    nocturnal_ls = (3, 1.5)
                else:
                    pain_ls = (4.5, 3.0)
                    urgency_ls = (5.5, 2.0)
                    nocturnal_ls = (3, 1.5)
            else:
                pain_ls = (5, 3.0)
                urgency_ls = (6, 2.0)
                nocturnal_ls = (3, 1.5)

            pain_scores = np.clip(
                np.random.normal(pain_ls[0], pain_ls[1], NUM_PATIENTS), 0, 9
            ).astype(int)

            urgency_scores = np.clip(
                np.random.normal(urgency_ls[0], urgency_ls[1], NUM_PATIENTS),
                0,
                9,
            ).astype(int)

            nocturnal_frequency = np.clip(
                np.random.normal(
                    nocturnal_ls[0], nocturnal_ls[1], NUM_PATIENTS
                ),
                0,
                20,
            ).astype(int)

            evaluation["pain"] = pain_scores[i]
            evaluation["urgency"] = urgency_scores[i]
            evaluation["nocturnal frequency"] = nocturnal_frequency[i]
            evaluation["treatment time"] = treatment_time
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


def generate_rs_binary_variables(
    patients_entry: pd.DataFrame,
    patients_risk_set: pd.DataFrame,
) -> pd.DataFrame:
    percentile = 33

    current = {
        "pain": "pain current",
        "urgency": "urgency current",
        "nocturnal frequency": "nocturnal frequency current",
    }

    baseline = {
        "pain": "pain baseline",
        "urgency": "urgency baseline",
        "nocturnal frequency": "nocturnal frequency baseline",
    }

    variables = list(current.values()) + list(baseline.values())

    rs_binary_variables = {}

    def generate_binary_variables(
        data: pd.DataFrame, variables: list[str]
    ) -> pd.DataFrame:
        binary_variables = {}

        for v in variables:
            lower_percentile = np.percentile(data[v], percentile)
            upper_percentile = np.percentile(data[v], 100 - percentile)

            binary_lower = (data[v] <= lower_percentile).astype(int)
            binary_middle = (
                (data[v] > lower_percentile) & (data[v] <= upper_percentile)
            ).astype(int)
            binary_upper = (data[v] > upper_percentile).astype(int)

            binary_variables[f"{v} lower"] = binary_lower
            binary_variables[f"{v} middle"] = binary_middle
            binary_variables[f"{v} upper"] = binary_upper

        return pd.DataFrame(binary_variables)

    for i, (t, c) in patients_risk_set.items():
        treated = (t.rename(columns=current)).merge(
            patients_entry.rename(columns=baseline),
            on=["id", "gender"],
        )

        treated = pd.concat(
            [
                treated,
                generate_binary_variables(treated, variables),
            ],
            axis=1,
        )

        controlled = (c.rename(columns=current)).merge(
            patients_entry.rename(columns=baseline),
            on=["id", "gender"],
        )
        controlled = pd.concat(
            [
                controlled,
                generate_binary_variables(controlled, variables),
            ],
            axis=1,
        )

        rs_binary_variables[i] = (treated, controlled)

    return pd.DataFrame(rs_binary_variables)


# Exported data
patients_entry = simulate_entry()

patients_evaluations = simulate_evaluations(patients_entry)

patients_risk_set = simulate_risk_set(patients_evaluations)

patients_rs_binary_variables = generate_rs_binary_variables(
    patients_entry, patients_risk_set
)

del simulate_entry, simulate_evaluations, generate_rs_binary_variables
