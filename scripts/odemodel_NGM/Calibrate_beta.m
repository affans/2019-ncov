% ## 2019-nCov Vaccination/COVID19
% ## Chad Wells, 2020
% Script to calculate the R0 based on the diff eq model in ncov_ode_model.jl


% Calibrate beta
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
% Demographics for model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55

Amin=[0 20 50 65]; % Age classes of 0-19, 20-49, 50-64, and 65+
% Calculate contact matrix and population sizes
[M,M2,P] = DemoUSA(Amin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Paramaters for calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%R0E - The basic reproductive number wanted for the model
%P - Population vector for the different age classes
%sigma - the rate from incubation period to symptomatic infectious period
%h - Age dependent hospitalization
%gamma - the rate of recovery
%delta - the rate to hospitalization
%M - The contact matrix for the community
%kappa - Relative infectivity of cases exhibiting mild
%symptoms
%theta - the proportion of cases exhibiting mild symptoms
% lb - the lower specified bound for searching beta
% ub - the lower specified bound for searching beta
% NS - Number of points used in the linear search

% Calculation
R0E=[2 2.5]; % Vecotor for R0
sigma=1/5.2; % Rate to sympmatic (infectious period)
gamma=1/(2*(7.5-5.2)); % Rate to recovery
delta=1/3.5; % Rate to hospitalization
kappa=0.5; %Realtive infectivity
theta=[0.8 0.8 0.4 0.2]; % proportion of mild infections
htemp=[0.0208    0.0214    0.0260    0.0415]; % Proportion hospitalized
h=htemp./(1-theta); % proportion severe infections hospitalized (Note: This has to change w.r.t. theta)
lb=0.01; % lower boudn for search
ub=0.1; % upper boudn for search
NS=5001; % How fine of scale you want the search
betaE=CalcR0(R0E,P,sigma,h,gamma,delta,M,kappa,theta,lb,ub,NS) % Run the calibration