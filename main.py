import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION PARAMETERS
# =========================

DAYS = 28
SM_TARGET = 42
THRESHOLD = 35

ET = 2.0
RAIN_FACTOR = 0.005

NOISE_STD = 0.8
IRRIGATION_EFF = 0.90

FC = 45
MAD = 35
OPT_LOW = 38
OPT_HIGH = 43

CROP_COEFFICIENTS = {
    "Rice": 1.30,
    "Wheat": 0.95,
    "Cotton": 1.15,
    "Soybean": 1.05,
    "Maize": 1.20
}

WATER_COST_PER_UNIT = 0.05
POWER_RATE = 12.0
PUMP_POWER_KW = 0.5
PUMP_HOURS_PER_UNIT = 0.02


# =========================
# DATA LOADING
# =========================

def load_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        "soil_moisture_%": "soil_moisture",
        "rainfall_mm": "rainfall",
        "temperature_c": "temperature"
    })

    drop_cols = [
        "farm_id", "region", "soil_ph", "humidity_%", "sunlight_hours",
        "irrigation_type", "fertilizer_type", "pesticide_usage_ml",
        "sowing_date", "harvest_date", "total_days",
        "yield_kg_per_hectare", "sensor_id", "timestamp",
        "latitude", "longitude", "ndvi_index", "crop_disease_status"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    df["crop_type"] = df["crop_type"].str.strip().str.title()

    return df


# =========================
# MODEL COMPONENTS
# =========================

def stress_probability(sm, drought_index, water_excess, base_risk=0.05):
    dry_risk = max(0, (SM_TARGET - sm) / 10)
    wet_risk = max(0, (sm - FC) / 10)
    memory = np.tanh(drought_index / 2.5)

    p = (
        0.40 * dry_risk +
        0.20 * wet_risk +
        0.25 * memory +
        0.15 * water_excess +
        base_risk
    )

    p += np.random.normal(0, 0.03)
    return np.random.rand() < np.clip(p, 0.03, 0.85)


def waste_calc(sm_before, water):
    excess = max(0, (sm_before + water) - FC)
    return excess * 0.7 + water * 0.04


def noisy(x):
    return x + np.random.normal(0, NOISE_STD)


# =========================
# SIMULATION MODEL
# =========================

def run_model(df, mode="threshold", crop_type=None):

    crop_df = df if crop_type is None else df[df["crop_type"] == crop_type].copy()

    if len(crop_df) == 0:
        raise ValueError(f"No records found for crop type: {crop_type}")

    Kc = CROP_COEFFICIENTS.get(crop_type, 1.0)
    sm = crop_df["soil_moisture"].iloc[0]
    drought_index = 0

    out = []

    for d in range(DAYS):
        row = crop_df.iloc[d % len(crop_df)]
        rain = row["rainfall"]

        ETo = ET + np.random.normal(0, 0.2)
        ETc = Kc * ETo

        sm_before = sm

        # -----------------
        # THRESHOLD MODEL
        # -----------------
        if mode == "threshold":
            water = 5 if sm < THRESHOLD else 0

        # -----------------
        # RULE MODEL
        # -----------------
        elif mode == "rule":
            if sm < 32:
                water = 5.5
            elif sm < 38:
                water = 3.0
            else:
                water = 1.2

            water *= np.random.uniform(0.85, 1.15)

        # -----------------
        # OPTIMIZATION MODEL
        # -----------------
        else:
            error = SM_TARGET - sm
            water = 0.60 * error + 0.40 * ETc - rain * RAIN_FACTOR

            if sm >= OPT_HIGH:
                water = 0
            elif sm >= SM_TARGET:
                water *= 0.20

            water += np.random.normal(0, 0.05)
            water = np.clip(water, 0, 3.5)

        # UPDATE SOIL MOISTURE
        sm = sm + water * IRRIGATION_EFF + rain * RAIN_FACTOR - ETc
        sm = noisy(sm)
        sm = np.clip(sm, 0, 100)

        drought_index = 0.85 * drought_index + (1 if sm < MAD else 0)
        water_excess = max(0, (sm - FC) / 10)

        stress = stress_probability(sm, drought_index, water_excess)
        waste = waste_calc(sm_before, water)

        out.append([d + 1, crop_type, Kc, ETo, ETc, sm, water, int(stress), waste])

    return pd.DataFrame(out, columns=[
        "Day", "Crop", "Kc", "ETo", "ETc",
        "Soil_Moisture", "Irrigation", "Stress", "Waste"
    ])


# =========================
# EVALUATION
# =========================

def evaluate_model(df_model):

    total_irrigation = df_model["Irrigation"].sum()

    stable_days = (
        (df_model["Soil_Moisture"] >= OPT_LOW) &
        (df_model["Soil_Moisture"] <= OPT_HIGH)
    ).sum()

    wue = stable_days / (total_irrigation + 1e-6)
    root_zone_stability = stable_days / len(df_model)

    waste_index = df_model["Waste"].sum() / (total_irrigation + 1e-6)

    irrigation_frequency = (df_model["Irrigation"] > 1.0).sum()
    stress_days = (df_model["Soil_Moisture"] < MAD).sum()

    water_cost = total_irrigation * WATER_COST_PER_UNIT

    energy_used = (
        total_irrigation * PUMP_HOURS_PER_UNIT * PUMP_POWER_KW
    )
    energy_cost = energy_used * POWER_RATE

    total_cost = water_cost + energy_cost

    return {
        "WUE": round(wue, 3),
        "Root_Zone_Stability": round(root_zone_stability, 3),
        "Water_Waste_Index": round(waste_index, 3),
        "Irrigation_Frequency": int(irrigation_frequency),
        "Stress_Days": int(stress_days),
        "Water_Cost": round(water_cost, 2),
        "Energy_Cost": round(energy_cost, 2),
        "Total_Cost": round(total_cost, 2)
    }


# =========================
# MAIN SIMULATION LOOP
# =========================

def main(data_path):

    df = load_data(data_path)

    all_simulation_summaries = []
    all_daily_results_list = []

    N_SIMULATIONS = 30

    for sim in range(1, N_SIMULATIONS + 1):

        np.random.seed(sim)

        all_results = []

        for crop in sorted(df["crop_type"].unique()):

            th = run_model(df, "threshold", crop)
            ru = run_model(df, "rule", crop)
            op = run_model(df, "opt", crop)

            th["Model"] = "Threshold"
            ru["Model"] = "Rule-Based"
            op["Model"] = "Optimization"

            all_results.extend([th, ru, op])

        results = pd.concat(all_results, ignore_index=True)
        results["Simulation"] = sim

        all_daily_results_list.append(results)

        summary_rows = []

        for crop in results["Crop"].unique():
            for model in ["Threshold", "Rule-Based", "Optimization"]:

                model_df = results[
                    (results["Crop"] == crop) &
                    (results["Model"] == model)
                ]

                eval_result = evaluate_model(model_df)

                summary_rows.append([
                    sim, crop, model,
                    eval_result["WUE"],
                    eval_result["Root_Zone_Stability"],
                    eval_result["Water_Waste_Index"],
                    eval_result["Irrigation_Frequency"],
                    eval_result["Stress_Days"],
                    eval_result["Water_Cost"],
                    eval_result["Energy_Cost"],
                    eval_result["Total_Cost"]
                ])

        sim_summary = pd.DataFrame(summary_rows, columns=[
            "Simulation", "Crop", "Model",
            "WUE", "Root_Zone_Stability", "Water_Waste_Index",
            "Irrigation_Frequency", "Stress_Days",
            "Water_Cost", "Energy_Cost", "Total_Cost"
        ])

        all_simulation_summaries.append(sim_summary)

    # =========================
    # FINAL SUMMARY (FIXED)
    # =========================

    all_simulation_results = pd.concat(all_simulation_summaries, ignore_index=True)

    cols = [
        "WUE",
        "Root_Zone_Stability",
        "Water_Waste_Index",
        "Irrigation_Frequency",
        "Stress_Days",
        "Water_Cost",
        "Energy_Cost",
        "Total_Cost"
    ]

    summary = all_simulation_results.groupby("Model")[cols].mean()

    print("\n=== FINAL MODEL COMPARISON (AVERAGED OVER SIMULATIONS) ===")
    print(summary.round(3))


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    main("data/Smart_Farming_Crop_Yield_2024.csv")
    import sys

    data_path = sys.argv[1]
    main(data_path)
