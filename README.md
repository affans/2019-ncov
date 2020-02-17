# COVID-19 (2019-nCov) Data Analysis

----
### Description of the files. 
All script files are located in the `scripts` folder. 

1. `analysis_one`: This script fits an exponential model to the incidence/prevalence, taking into account all data up to January 26th (the fitting is not good after the 26th because of large number of reported cases). It saves the fitted data in a file which is used by the R script `epi-estim-analysis.R` to compute temporal R0. This script also generates `n` epidemic curves using negative binomial distributions.

2. `analysis_two`: This script fits an exponential model to the incidence/prevalence **of the 41 cases** for which we know symptom on-set. The script saves the fitted data in a file which is used by the R script `epi-estim-analysis.R` to compute temporal R0. This script also generates `n` epidemic curves using negative binomial distributions. The data for symptom-onset of 41 cases were collected from Huang et al. in the Lancet. 

3. `epi-estim-analysis.R`: This script uses the `EpiEstim` R package to calculate temporal R0 values using data from either the `analysis_one` or `analysis_two` output. 

4. `contact_analysis.jl`: This script calculates the percent reduction required in contacts to bring R0 down to 1. 

5. `vaccine_ncov.jl`: ODE model for vaccine effectiveness and cost-effectiveness of 2019-ncov, COVID19