# COVID-19 (2019-nCov, SARS-CoV-2) Data Analysis

**Affan Shoukat, 2020**

This repository provides multiple scripts for various projects on COVID19. All script files are located in the `scripts` folder.
 


----
#### Project: Transmission and Reproduction Number Estimation

We estimated <img src="https://render.githubusercontent.com/render/math?math=R_0"> using the initial cluster of patients with symptom onset dates. Most other studies on <img src="https://render.githubusercontent.com/render/math?math=R_0"> estimation rely on reported cases from the official government agencies. Our estimates are slightly higher than estimated in other papers. Our report is published [on the NCCID website](https://nccid.ca/publications/nccid-special-post-transmissibility-of-the-initial-cluster-of-covid-19-patients-in-wuhan-china/). 

Files used in this project:

1. `data_fitting`: This script (a Jupyter Notebook) fits an exponential model to the incidence/prevalence, taking into account all reported data up to January 26th. It saves the fitted data in a file which is used by the R script `epi-estim-analysis.R` to compute temporal R0. 

2. `data_fitting_initial41`: This script (a Jupyter Notebook) fits an exponential model to the incidence/prevalence **of the 41 cases** for which we know symptom on-set. The script saves the fitted data in a file which is used by the R script `epi-estim-analysis.R` to compute temporal R0.  The data for symptom-onset of 41 cases were collected from Huang et al. in the Lancet. 


3. `epi-estim-analysis.R`: This script uses the `EpiEstim` R package to calculate temporal R0 values using data from either the `data_fitting` or `data_fitting_initial41` output. 

4. `contact_analysis.jl`: This script calculates the percent reduction required in contacts to bring the basic reproduction to 1. 


#### Project: Vaccine Effectivness and Strategies. 
A dynamical system modelling study to evaluate the surge capacity and peak incidence in the USA, parameterized by latest epidemiological estimates of COVID19 and USA demographics. 

The file `vaccine_ncov.jl` implements the ODE model for vaccine effectiveness of COVID19. See below for reproducibility steps. This model is self-contained in a single file and does not depend on any other files.

1. Install Julia 1.3+ and download the following packages: `DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra, StatsPlots, Query, Distributions, Statistics, Random, DelimitedFiles`. See [here for package installation instructions](https://datatofish.com/install-package-julia/).

2. Launch Julia and run `include("vaccine_ncov.jl")`. (You will either need to launch Julia from the present directory or include the full path). This will being into scope all the functions defined to run the model. 

3. Run the model by `sol = run_single()`. This function is initialize `ModelParameters` and pass it to `run_model(p::ModelParameters, nsims)` which solves the ODE system defined in `Model!()`. The `nsims` argument is the number of simulations to run. The function `run_single()` returns an array of solutions. 

4. Plot the results of `sol = run_single()` by calling the function `plots(sol)`. 

5. (optional) To evaluate different parameters or testing and calibration purposes, change values manually inside the function `run_single()`. (Remember to reinclude the new function for the new parameters to take effect. Use of `Revise.jl` is highly recommended since you can not redefine the structure `ModelParameters`). 

6. (optional) Helper functions include `getclass(i, sol)` which extracts the solution of the i'th compartment of the differential equation. For example, `getclass(1, sol)` will get the solution curve for S₁ compartment. 

7. (optional) The function `run_scenarios()` iterively goes through the scenarios we were interested in studying and calls `run_model()` for each set of parameters.