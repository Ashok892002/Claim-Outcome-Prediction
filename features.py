
import pandas as pd

def add_features(data):
    data = data.copy()

    data["amount_per_year_of_age"] = (
        data["billed_amount"] / (data["patient_age"] + 1)
    )

    data["is_high_billed"] = (
        data["billed_amount"] > 600
    ).astype(int)

    data["payer_cpt"] = (
        data["payer"].astype(str) + "_" +
        data["cpt_code"].astype(str)
    )

    return data

