# ml_thesis_acd_on_energy_data
Master Thesis Machine Learning 2022
Learning causal relations in power system time series
------------------------------------
The code files of this thesis are structured as follows:

01 Data_Preparation.ipynb


- proprocess energy data and create input files needed for training

- time series and weather data comes from https://open-power-system-data.org/ and additional price data for CH and FR from https://www.smard.de

- calculate basic statistics and create data plots



02 Train_Models.ipynb


- implementations for two baselines: Multivariate Linear Regression (MLR) and Long Short-Term Memory (LSTM)

- import ACD_code.py containing the implementation of Amortized Causal discovery, which is (slightly) adapted from the official implementation on https://github.com/loeweX/AmortizedCausalDiscovery

- train all models (ACD/MLR/LSTM)



03 Evaluation_and_Plotting.ipynb


- make predictions on test set for all models

- calculate errors based only on the time series of interest (Load, Solar, Wind, Prices)

- plot predictions and graphs

