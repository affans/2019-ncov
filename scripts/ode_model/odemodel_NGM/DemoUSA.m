% ## 2019-nCov Vaccination/COVID19
% ## Chad Wells, 2020
% Script to calculate the R0 based on the diff eq model in ncov_ode_model.jl


function [M,M2,P] = DemoUSA(Amin)
% Returns contact matrix for community and home absed on specified
% compression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Amin - The minimum age of the different classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%M - contact matrix (Size: AxA)
%M2 - contact matrix home (Size: AxA)
%P -Population size

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Full population size at 5 yr increments starting 0-4 to 100+
PFull=[19810275
20195642
20879527
21097221
21873579
23561756
22136018
21563587
19714301
20747135
20884564
21940985
20331651
17086893
13405423
9267066
6127308
3898995
1958772
592809
93927];
% Specify the age vector based on the Amin input
AA=5.*[0:(length(PFull)-1)];
% Allocate space for aggregation of age classes
Findx=cell(length(Amin),1); % Index cell array for age groups
F80indx=cell(length(Amin),1); % Index cell array for age groups up-to 80 (used in contact matrix)
% Find the index for the specified age range
hh=1; % This is used if there is only a single age class (which would not be relevant)
for ii=1:length(Amin)-1
    gg=find(AA==Amin(ii)); % Find the index for the specified minimum
    hh=find(AA==Amin(ii+1)); % Find the index for the next specified minimum
    Findx{ii}=[gg:(hh-1)]; % The index is from gg to the index before hh
    F80indx{ii}=[gg:(hh-1)]; % The index is from gg to the index before hh
end
Findx{ii+1}=hh:length(PFull); % The index for the last class goes from hh to the end
gg=find(AA==Amin(end)); % Find the index for the minimum of the vector A min
hh=find(AA==80); % Find the index for the age class 80 (as contact matrix ends at 75-79)
F80indx{ii+1}=[gg:(hh-1)]; % % The index is from gg to the index before hh (i.e. before age class 80 --> 75-79)

% Full contact matrix for the USA for all locations
MFull=[2.598237	1.101286	0.499396	0.315998	0.411961	0.715457	1.057365	0.988814	0.497488	0.322391	0.336978	0.26598	0.173599	0.135957	0.073886	0.038853
0.989686	5.386372	1.224101	0.347044	0.193902	0.509275	0.892744	1.068942	0.846197	0.347779	0.241652	0.197511	0.170506	0.12128	0.051215	0.039004
0.304842	1.888934	8.284524	0.973481	0.347356	0.301111	0.537831	0.869836	1.047914	0.564866	0.319458	0.16629	0.104552	0.098895	0.063186	0.052481
0.173684	0.432369	3.06756	11.10614	1.599837	0.783971	0.636821	0.892186	1.12534	1.009137	0.533794	0.23687	0.09507	0.067864	0.033226	0.021371
0.262619	0.21439	0.329929	2.645996	4.257321	1.742612	1.146007	1.040654	0.915094	1.079188	0.738022	0.412893	0.121239	0.052891	0.04983	0.040059
0.579676	0.34356	0.209946	0.882349	2.024331	3.414656	1.729206	1.333241	1.128603	0.952613	0.905713	0.498146	0.162274	0.055466	0.02694	0.018639
0.685994	0.925715	0.70581	0.511232	0.993349	1.651464	2.724437	1.697023	1.31631	1.026176	0.798953	0.542883	0.215146	0.088143	0.040122	0.037413
0.673234	1.087263	0.907901	0.762107	0.712048	1.280754	1.57267	2.780799	1.83112	1.187005	0.862649	0.468816	0.253787	0.147221	0.074796	0.029849
0.344231	0.743115	1.012538	1.172505	0.867901	1.128282	1.445997	1.662298	2.523069	1.465606	1.041925	0.393184	0.213727	0.11748	0.074063	0.032296
0.369636	0.561619	0.752136	1.656516	0.894872	0.95449	1.148589	1.325174	1.421437	2.031986	1.079285	0.497942	0.183696	0.092167	0.074005	0.066027
0.307299	0.676821	1.043991	1.429576	1.02141	1.296546	1.161537	1.154061	1.522421	1.697469	1.930457	0.891297	0.310827	0.127652	0.079973	0.067437
0.536572	0.736502	0.751744	0.996978	0.775486	1.224096	1.225258	0.977184	1.088083	0.956104	1.215903	1.598582	0.546926	0.215659	0.095427	0.064091
0.447587	0.435037	0.339826	0.509585	0.424671	0.623602	0.717327	0.777598	0.634482	0.549384	0.551756	0.716844	1.069015	0.37754	0.188201	0.075498
0.274332	0.432112	0.360245	0.237979	0.277204	0.353273	0.498051	0.526944	0.541686	0.324366	0.354407	0.439389	0.481989	0.900066	0.231848	0.091894
0.107388	0.324594	0.320266	0.345067	0.15107	0.233391	0.230088	0.393683	0.548842	0.407598	0.325744	0.275529	0.480619	0.471458	0.683924	0.216669
0.210146	0.282074	0.419783	0.336161	0.138282	0.15386	0.23403	0.294429	0.344528	0.430889	0.383853	0.221516	0.166169	0.238682	0.250445	0.396258];
% Full contact matrix for the USA for only at home
M2Full=[0.619699	0.519309	0.273171	0.122099	0.172915	0.317096	0.556542	0.562403	0.213493	0.08079	0.061797	0.029987	0.020785	0.009464	0.002771	0.004486
0.319301	0.985416	0.487797	0.161959	0.043457	0.199983	0.51032	0.645337	0.461914	0.127765	0.05226	0.02669	0.013276	0.010203	0.003921	0.00339
0.177181	0.519932	1.52392	0.402288	0.059383	0.039347	0.198347	0.482	0.554877	0.213346	0.07943	0.024422	0.012561	0.012496	0.00828	0.003763
0.085271	0.163466	0.458029	1.290057	0.19609	0.048415	0.049507	0.244721	0.439034	0.36974	0.186676	0.058342	0.016116	0.015776	0.005813	0.002674
0.147842	0.070406	0.084661	0.357683	1.208625	0.208847	0.05697	0.027084	0.136703	0.319342	0.190628	0.108003	0.019465	0.005588	0.004129	0.003219
0.39766	0.178023	0.043948	0.0829	0.231941	0.992952	0.22154	0.030563	0.014188	0.070238	0.159581	0.116132	0.043054	0.008038	0.001035	0.004178
0.482799	0.547389	0.270238	0.053308	0.063373	0.197624	0.880607	0.191623	0.083306	0.017345	0.031822	0.050587	0.04698	0.007488	0.003611	0.002553
0.453281	0.742789	0.601557	0.233327	0.030012	0.0321	0.145678	0.917033	0.176858	0.037459	0.022987	0.016383	0.029149	0.016191	0.00615	0.001933
0.212172	0.500954	0.651967	0.44861	0.103033	0.026289	0.089226	0.186577	0.7325	0.120203	0.036477	0.006614	0.023745	0.025323	0.011007	0.00493
0.136259	0.27871	0.453793	0.618342	0.338379	0.085128	0.030916	0.093554	0.165242	0.757664	0.142921	0.033596	0.014121	0.009896	0.009165	0.01458
0.19468	0.178425	0.332587	0.412934	0.356065	0.222836	0.088399	0.053726	0.097245	0.178225	0.732492	0.161746	0.032856	0.007849	0.008561	0.017963
0.291725	0.296161	0.222646	0.33221	0.30779	0.367346	0.249909	0.08112	0.042157	0.132227	0.24003	0.79826	0.165735	0.038899	0.00605	0.013891
0.311345	0.288344	0.20589	0.184573	0.138437	0.194156	0.249503	0.179676	0.10188	0.048114	0.098838	0.207235	0.664002	0.103621	0.021957	0.003799
0.213625	0.322172	0.297049	0.180858	0.117946	0.110641	0.180064	0.240919	0.255591	0.06754	0.05972	0.090417	0.144707	0.600705	0.08574	0.009134
0.087535	0.280315	0.256362	0.212743	0.042	0.077569	0.069271	0.156575	0.258791	0.150311	0.083949	0.037344	0.101479	0.134146	0.401519	0.092435
0.178671	0.22724	0.369939	0.286083	0.078802	0.068464	0.086063	0.172616	0.222537	0.262969	0.252851	0.088714	0.035285	0.066994	0.115221	0.301984];

% Allocate memory for the population calcuations
P=zeros(length(Amin),1); % population array that will be retrined
Ptemp=zeros(length(MFull(:,1)),length(Amin)); % Density matrix used in compression of contact matrix

Mtemp=zeros(length(Amin),length(MFull(:,1))); % temp matrix used in the compression
M2temp=zeros(length(Amin),length(MFull(:,1))); % temp matrix used in the compression

% Compress contact matrix
for ii=1:length(P)
    P(ii)=sum(PFull([Findx{ii}])); % Aggregate population based on the age classes   
    Ptemp([F80indx{ii}],ii)=PFull([F80indx{ii}])./sum(PFull([F80indx{ii}])); % Density of populations based on the age classes up to 80 for compression of contact matrix
    Mtemp(ii,:)=sum(MFull([F80indx{ii}],:),1); % Aggregate number of contacts (i.e. the ii row is our number of contacts in our compressed age class)
    M2temp(ii,:)=sum(M2Full([F80indx{ii}],:),1); % Aggregate number of contacts (i.e. the ii row is our number of contacts in our compressed age class)
end

M=Mtemp*Ptemp; % Compute the average number of coantacts for the different ages classes based on the density in each age class
M2=M2temp*Ptemp; % Compute the average number of coantacts for the different ages classes based on the density in each age class

M=(M+M')./2; % Make the cotnact matrix symetric
M2=(M2+M2')./2; % Make the cotnact matrix symetric

end
