import numpy as np
import pandas as pd

np.random.seed(69)

NUM_PATIENTS = 100
SIZE = NUM_PATIENTS * 2

GENDER = np.random.choice(["M", "F"], size=SIZE)


def baseline() -> [dict]:
    PAIN_SCORES = np.random.randint(0, 10, size=SIZE)
    URGENCY_SCORES = np.random.randint(0, 10, size=SIZE)
    NOCTURNAL_FREQUENCY_SCORES = np.random.randint(0, 21, size=SIZE)

    TREATED_INDICES = np.random.permutation(SIZE)[:NUM_PATIENTS]

    return [
        {
            "id": i,
            "gender": GENDER[i],
            "group": "treated" if i in TREATED_INDICES else "untreated",
            "pain": PAIN_SCORES[i],
            "urgency": URGENCY_SCORES[i],
            "nocturnal frequency": NOCTURNAL_FREQUENCY_SCORES[i],
        }
        for i in range(SIZE)
    ]


def at_treatment() -> [dict]:
    PAIN_SCORES = np.random.randint(0, 10, size=SIZE)
    URGENCY_SCORES = np.random.randint(0, 10, size=SIZE)
    NOCTURNAL_FREQUENCY_SCORES = np.random.randint(0, 21, size=SIZE)

    TREATED_INDICES = np.random.permutation(SIZE)[:NUM_PATIENTS]

    return [
        {
            "id": i,
            "gender": GENDER[i],
            "group": "treated" if i in TREATED_INDICES else "untreated",
            "pain": PAIN_SCORES[i],
            "urgency": URGENCY_SCORES[i],
            "nocturnal frequency": NOCTURNAL_FREQUENCY_SCORES[i],
        }
        for i in range(SIZE)
    ]


# Exported data
patients_baseline = pd.DataFrame(baseline())

patients_at_treatment = pd.DataFrame(at_treatment())

del pd
