# Implementation of Threshold-, Rules-, and Optimization-based Irrigation Scheduling Algorithms for Water Use Optimization (Proof of Concept)

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

Link to Colab: https://colab.research.google.com/drive/1ZCwUBQMlcLV86qU51Ey-ZWHXPa-hrSV6?usp=sharing

## How to Run

1. Clone the repository:

   git clone https://github.com/aglseballos/Irrigation-System.git

   cd Irrigation-System
3. Install dependencies: pip install -r requirements.txt
4. Run the simulation: python main.py data/Smart_Farming_Crop_Yield_2024.csv 
   
## Output
The system prints a comparison table of all algorithms.
=== FINAL MODEL COMPARISON (AVERAGED OVER SIMULATIONS) ===
|Model| WUE | Root_Zone_Stability | Water_Waste_Index | ... | Water_Cost | Energy_Cost | Total_Cost                                                        ...                                     
| --------|---------|---------| --------|---------|---------| --------|---------|
|Optimization | 0.413      |          0.717         |      0.04 | ...  |     2.787  |      6.689 |      9.475
|Rule-Based    |0.331       |         0.611          |     0.08  |...   |    2.815   |     6.757  |     9.572
|Threshold     |0.076        |        0.114           |    0.04  |...    |   2.553     |   6.128   |    8.681
