% ## 2019-nCov Vaccination/COVID19
% ## Chad Wells, 2020
% Script to calculate the R0 based on the diff eq model in ncov_ode_model.jl

function betaE=CalcR0(R0E,P,sigma,h,gamma,delta,M,kappa,theta,lb,ub,NS)
%% betaE=CalcR0(R0E,N,sigma,h,gamma,delta,M,xi,s)
% Calibrates the value for beta in the model for a given R0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%R0E - The basic reproductive number wanted for the model
%P - Population vector for the different age classes
%sigma - the rate from incubation period to symptomatic infectious period
%h - Age dependent hospitalization
%gamma - the rate of recovery
%delta - the rate to hospitalization
%M - The contact matrix for the community
%kappa - Relative infectivity of cases exhibiting mild
%symptoms
%theta - vector of proportion of cases exhibiting mild symptoms 
% lb - the lower specified bound for searching beta
% ub - the lower specified bound for searching beta
% NS - Number of points used in the linear search
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% betaE - Estimated value of beta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcuation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

betav=linspace(lb,ub,NS); % Vector spanning the search criteria for the value of beta
R0=zeros(NS,1); % The values of R0 for the corresponding betaV

for rr=1:NS
    beta=betav(rr); 
    % Construction of the F matrix
    F=[];
    for ii=1:4
        F=[F; 0 0 0 0 beta*M(ii,1)*P(ii)/P(1)  beta*M(ii,2)*P(ii)/P(2) beta*M(ii,3)*P(ii)/P(3) beta*M(ii,4)*P(ii)/P(4) beta*M(ii,1)*P(ii)/P(1)  beta*M(ii,2)*P(ii)/P(2) beta*M(ii,3)*P(ii)/P(3) beta*M(ii,4)*P(ii)/P(4) kappa*beta*M(ii,1)*P(ii)/P(1)  kappa*beta*M(ii,2)*P(ii)/P(2) kappa*beta*M(ii,3)*P(ii)/P(3) kappa*beta*M(ii,4)*P(ii)/P(4)];
    end
    % Fill reamining parts with zero
    F=[F; zeros(12,length(F(1,:)))];
    
    % Contruct the V matrix
    V=zeros(size(F));
    % The rate from the incubation period
    for ii=1:4
        V(ii,ii)=sigma; 
        V(4+ii,ii)=-(1-theta(ii)).*sigma.*(1-h(ii));
        V(8+ii,ii)=-(1-theta(ii)).*sigma.*h(ii);
        V(12+ii,ii)=-theta(ii).*sigma;
    end
    % The rates for the periods in which these cases are infectious
    for ii=5:8
        V(ii,ii)=gamma;
        V(ii+4,ii+4)=delta;
        V(ii+8,ii+8)=gamma;
    end
     %Compute R0 using eignevalues
    R0(rr)=max(abs(eig(F*inv(V)))); % Determine the maximal spectral radius of all the eigen values
end
% Interpolation to determine that value of betaE
betaE=pchip(R0,betav,R0E);
% Determine the it falls within the bounds specified for the search
for jj=1:length(betaE)
    if((betaE(jj)>ub)||(betaE(jj)<lb)) 
        betaE(jj)=NaN;
    end
end
end

   