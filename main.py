import argparse

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DAYS = 28
N_SIMULATIONS = 30

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

WATER_COST_PER_UNIT = 0.05
POWER_RATE = 12.0
PUMP_POWER_KW = 0.5
PUMP_HOURS_PER_UNIT = 0.02

CROP_COEFFICIENTS = {
    "Rice": 1.30,
    "Maize": 1.20,
    "Cotton": 1.15,
    "Sugarcane": 1.25,
    "Potato": 1.10,
    "Wheat": 0.95,
}

MODEL_COLORS = {
    "Threshold": "#2a78d6",
    "Rule-Based": "#1baf7a",
    "Optimization": "#eda100",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"rainfall_mm": "rainfall"})

    required_columns = {"crop_type", "soil_moisture", "rainfall", "region"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    df = df[["crop_type", "soil_moisture", "rainfall", "region"]].copy()
    df["crop_type"] = df["crop_type"].astype(str).str.strip().str.title()
    df["region"] = df["region"].astype(str).str.strip().str.title()
    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")

    df = df.dropna(subset=["crop_type", "soil_moisture", "rainfall"])
    df = df[df["region"] == "East"].copy()

    if df.empty:
        raise ValueError("No valid records found for the East region.")

    return df


def stress_probability(sm, drought_index, water_excess, base_risk=0.05):
    dry_risk = max(0, (SM_TARGET - sm) / 10)
    wet_risk = max(0, (sm - FC) / 10)
    memory = np.tanh(drought_index / 2.5)

    probability = (
        0.40 * dry_risk
        + 0.20 * wet_risk
        + 0.25 * memory
        + 0.15 * water_excess
        + base_risk
    )
    probability += np.random.normal(0, 0.03)

    return np.random.rand() < np.clip(probability, 0.03, 0.85)


def waste_calc(sm_before, water):
    excess = max(0, (sm_before + water) - FC)
    return excess * 0.7 + water * 0.04


def noisy(value):
    return value + np.random.normal(0, NOISE_STD)


def run_model(data, mode, crop_type):
    crop_df = data[data["crop_type"] == crop_type].copy()

    if crop_df.empty:
        raise ValueError(f"No records found for crop type: {crop_type}")

    kc = CROP_COEFFICIENTS.get(crop_type, 1.0)
    soil_moisture = crop_df["soil_moisture"].iloc[0]
    drought_index = 0
    results = []

    for day in range(DAYS):
        row = crop_df.iloc[day % len(crop_df)]
        daily_rain = row["rainfall"] / 365
        eto = ET + np.random.normal(0, 0.2)
        etc = kc * eto
        sm_before = soil_moisture

        if mode == "threshold":
            water = 5 if soil_moisture < THRESHOLD else 0

        elif mode == "rule":
            if soil_moisture < 32:
                moisture_level = "Low"
            elif soil_moisture < 38:
                moisture_level = "Medium"
            else:
                moisture_level = "High"

            if daily_rain < 1:
                rainfall_level = "None"
            elif daily_rain < 5:
                rainfall_level = "Light"
            else:
                rainfall_level = "Heavy"

            if moisture_level == "Low" and rainfall_level == "None":
                water = 5.5
            elif moisture_level == "Medium" and rainfall_level == "Light":
                water = 3.0
            else:
                water = 1.2

            water *= np.random.uniform(0.85, 1.15)

        elif mode == "opt":
            error = SM_TARGET - soil_moisture
            expected_rain_effect = daily_rain * RAIN_FACTOR

            water = 0.60 * error + 0.40 * etc - expected_rain_effect

            if soil_moisture >= OPT_HIGH:
                water = 0
            elif soil_moisture >= SM_TARGET:
                water *= 0.20

            water += np.random.normal(0, 0.05)
            water = np.clip(water, 0, 3.5)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        soil_moisture = (
            soil_moisture
            + water * IRRIGATION_EFF
            + daily_rain * RAIN_FACTOR
            - etc
        )
        soil_moisture = np.clip(noisy(soil_moisture), 0, 100)

        drought_index = 0.85 * drought_index + (1 if soil_moisture < MAD else 0)
        water_excess = max(0, (soil_moisture - FC) / 10)
        stress = stress_probability(
            soil_moisture, drought_index, water_excess
        )
        waste = waste_calc(sm_before, water)

        results.append(
            [
                day + 1,
                crop_type,
                kc,
                eto,
                etc,
                soil_moisture,
                water,
                int(stress),
                waste,
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "Day",
            "Crop",
            "Kc",
            "ETo",
            "ETc",
            "Soil_Moisture",
            "Irrigation",
            "Stress",
            "Waste",
        ],
    )


def evaluate_model(model_df):
    total_irrigation = model_df["Irrigation"].sum()
    stable_days = (
        (model_df["Soil_Moisture"] >= OPT_LOW)
        & (model_df["Soil_Moisture"] <= OPT_HIGH)
    ).sum()

    water_cost = total_irrigation * WATER_COST_PER_UNIT
    energy_used = total_irrigation * PUMP_HOURS_PER_UNIT * PUMP_POWER_KW
    energy_cost = energy_used * POWER_RATE

    return {
        "WUE": round(stable_days / (total_irrigation + 1e-6), 3),
        "Root_Zone_Stability": round(stable_days / len(model_df), 3),
        "Water_Waste_Index": round(
            model_df["Waste"].sum() / (total_irrigation + 1e-6), 3
        ),
        "Irrigation_Frequency": int((model_df["Irrigation"] > 1.0).sum()),
        "Stress_Days": int((model_df["Soil_Moisture"] < MAD).sum()),
        "Water_Cost": round(water_cost, 2),
        "Energy_Cost": round(energy_cost, 2),
        "Total_Cost": round(water_cost + energy_cost, 2),
    }


def simulate(data):
    summaries = []
    daily_results = []

    crops = sorted(data["crop_type"].unique())
    models = {
        "Threshold": "threshold",
        "Rule-Based": "rule",
        "Optimization": "opt",
    }

    for simulation in range(1, N_SIMULATIONS + 1):
        np.random.seed(simulation)
        simulation_results = []

        for crop in crops:
            for model_name, mode in models.items():
                result = run_model(data, mode, crop)
                result["Model"] = model_name
                result["Simulation"] = simulation
                simulation_results.append(result)

        results = pd.concat(simulation_results, ignore_index=True)
        daily_results.append(results)

        for crop in crops:
            for model_name in models:
                model_df = results[
                    (results["Crop"] == crop)
                    & (results["Model"] == model_name)
                ]
                metrics = evaluate_model(model_df)
                summaries.append(
                    {
                        "Simulation": simulation,
                        "Crop": crop,
                        "Model": model_name,
                        **metrics,
                    }
                )

    all_results = pd.concat(daily_results, ignore_index=True)
    all_summaries = pd.DataFrame(summaries)

    average_summary = (
        all_summaries.groupby(["Crop", "Model"], as_index=False)
        .mean(numeric_only=True)
        .round(3)
    )

    average_summary["Score"] = (
        average_summary["WUE"] * 6
        + average_summary["Root_Zone_Stability"] * 5
        - average_summary["Water_Waste_Index"] * 6
        - average_summary["Irrigation_Frequency"] * 0.5
        - average_summary["Stress_Days"] * 3
        - average_summary["Total_Cost"] * 0.3
    )

    return all_results, average_summary


def create_dashboard(all_daily_results, average_summary, crops):
    def run_dashboard(selected_crop):
        crop_results = average_summary[
            average_summary["Crop"] == selected_crop
        ].copy()

        best_algo = crop_results.loc[crop_results["Score"].idxmax()]
        crop_results["Best"] = crop_results["Model"].apply(
            lambda model: "Recommended" if model == best_algo["Model"] else ""
        )
        display_df = crop_results.drop(
            columns=["Crop", "Simulation"], errors="ignore"
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        for model in ["Threshold", "Rule-Based", "Optimization"]:
            subset = all_daily_results[
                (all_daily_results["Crop"] == selected_crop)
                & (all_daily_results["Model"] == model)
            ]
            average_plot = subset.groupby("Day")["Soil_Moisture"].mean()

            ax.plot(
                average_plot.index,
                average_plot.values,
                label=model,
                color=MODEL_COLORS[model],
                linewidth=2,
            )

        ax.axhspan(OPT_LOW, OPT_HIGH, alpha=0.08, color="green")
        ax.axhline(
            OPT_LOW,
            color="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Optimal range",
        )
        ax.axhline(
            OPT_HIGH, color="green", linestyle="--", linewidth=1, alpha=0.5
        )
        ax.axhline(
            MAD, color="red", linestyle=":", linewidth=1.5, label="MAD"
        )

        ax.set_xlabel("Day")
        ax.set_ylabel("Average soil moisture (%)")
        ax.set_title("Average Soil Moisture Trends", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        summary = f"""### Recommended Algorithm

**{best_algo["Model"]}** — Score: **{best_algo["Score"]:.4f}**

WUE: {best_algo["WUE"]} · Root Zone Stability: {best_algo["Root_Zone_Stability"]} · Stress Days: {best_algo["Stress_Days"]} · Total Cost: ₱{best_algo["Total_Cost"]}"""

        return summary, display_df, fig

    with gr.Blocks(
        title="Irrigation Scheduling Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# Irrigation Scheduling Dashboard\n"
            "East Region · 30 simulations per crop"
        )

        crop_selector = gr.Dropdown(
            choices=crops,
            value=crops[0],
            label="Select crop",
        )

        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown()
                gr.Markdown(
                    "### Algorithm Scores\n"
                    "Higher composite scores indicate better overall performance."
                )
                table_output = gr.Dataframe(interactive=False)

            with gr.Column():
                gr.Markdown(
                    "### Soil Moisture Simulation\n"
                    "Average daily soil moisture across 30 simulations."
                )
                plot_output = gr.Plot()

        crop_selector.change(
            fn=run_dashboard,
            inputs=crop_selector,
            outputs=[summary_output, table_output, plot_output],
        )
        app.load(
            fn=run_dashboard,
            inputs=crop_selector,
            outputs=[summary_output, table_output, plot_output],
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Run the irrigation scheduling dashboard."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the irrigation dataset CSV file.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio sharing link.",
    )
    args = parser.parse_args()

    data = load_data(args.data)
    all_daily_results, average_summary = simulate(data)

    print("\n=== Average Crop-Specific Evaluation ===")
    print(average_summary.to_string(index=False))

    dashboard = create_dashboard(
        all_daily_results,
        average_summary,
        sorted(data["crop_type"].unique()),
    )
    dashboard.launch(share=args.share)


if __name__ == "__main__":
    main()
