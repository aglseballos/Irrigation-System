# Implementation of Threshold-, Rules-, and Optimization-based Irrigation Scheduling Algorithms for Water Use Optimization (Proof of Concept)
### By Karen Althea Aquino, Angel Grace Seballos, and Bryan Anthony Al Shidhani

## Overview
This project simulates and compares three irrigation strategies:
- Threshold-based
- Rule-based
- Optimization-based

The system simulates soil moisture dynamics over time and evaluates performance using:

- Water Use Efficiency (WUE)
- Root Zone Stability
- Water Waste Index
- Irrigation Frequency
- Stress Days

Link to Colab: https://colab.research.google.com/drive/1N4sO6CeiZFVjhUYBdIdZ3-8vH0DA844j?usp=sharing

## How to Run

1. Clone the repository:

   git clone https://github.com/aglseballos/Irrigation-System.git

   cd Irrigation-System
3. Install dependencies: pip install -r requirements.txt
4. Run the simulation: python main.py data/irrigation_prediction.csv 
   
## Output
The application launches an interactive Gradio dashboard for comparing three irrigation scheduling approaches—Threshold-Based, Rule-Based, and Optimization-Based—using East-region records from the supplied dataset.

After running 30 simulations per crop, the dashboard provides:
- A crop selector for viewing Rice, Wheat, Maize, Cotton, Potato, Sugarcane, and other available crops.
- A recommended irrigation algorithm for the selected crop, based on a composite performance score.
- A comparison table containing Water Use Efficiency (WUE), Root Zone Stability, Water Waste Index, Irrigation Frequency, Stress Days, Water Cost, Energy Cost, and Total Cost.
- A line chart showing average daily soil-moisture levels over 28 simulated days for all three algorithms.
- Reference markers for the optimal soil-moisture range and the Management Allowed Depletion (MAD) threshold.
- The terminal also prints the complete average crop-by-model evaluation table after the simulations finish.


