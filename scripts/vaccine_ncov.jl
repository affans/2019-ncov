## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra
using StatsPlots, Query, Distributions, Statistics, Random, DelimitedFiles

∑(x) = sum(x) # synctactic sugar
heaviside(x) = x <= 0 ? 0 : 1

@with_kw mutable struct ModelParameters
    # default parameter values, some are overwritten in the main function. 
    ## parameters for transmission dynamics. 
    β::Float64 = 0.037 # (input) 0.037: R0 ~ 2.5, next generation method
    σ::Float64 = 1/5.2 # (sampled)incubation period 5.2 days mean, LogNormal 
    q::NTuple{4, Float64} = (0.05, 0.05, 0.05, 0.05) # (fixed) proportion of self-quarantine
    h::NTuple{4, Float64} = (0.02075, 0.02140, 0.025, 0.03885) ## (sampled), default values mean of distribution 
    f::Float64 = 0.05  ## (input) default 5% self-isolation 
    c::NTuple{4, Float64} = (0.0129, 0.03875, 0.203, 0.47) ## (sampled), default values mean of distribution     
    τ::Float64 = 0.5   # (input) symptom onset to isolation, default  average 1 day
    γ::Float64 = 1/4.6 # (fixed) symptom onset to recovery, assumed fixed, based on serial interval... sampling creates a problem negative numbers
    δ::Float64 = 1/3.5 # (sampled), default value mean of distribution

    ## vaccination specific parameters
    nva::Float64 = 0.0 ## vaccine doses limit. baseline 5e6 
    ν::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) 
    ξ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) # reduction in transmission, use vaccine efficacy or 0.0 to be conservative
    ϵ::NTuple{4, Float64} = (0.64, 0.64, 0.48, 0.32)# vaccine efficacy, frailty index per age group * 80% base efficacy
    
    ## transmission parameters for vaccinated
    q̃::NTuple{4, Float64} = (0.05, 0.05, 0.05, 0.05) # fixed
    h̃::NTuple{4, Float64} = (0.02075, 0.02140, 0.025, 0.03885) ## sampled from h in sims
    f̃::Float64 = 0.05 ## same as f
    c̃::NTuple{4, Float64} = (0.0129, 0.03875, 0.203, 0.47)  ## sampled from c in sims

    ## recovery and mortality
    mH::Float64 = 0.1116  ## prob of death in hospital
    μH::Float64 = 1/12.4  ## length of hospital stay before death (CDC)
    ψH::Float64 = 1/10    ## length of hospital stay before recovery
    mC::Float64 = 0.1116  ## prob of death in ICU 
    μC::Float64 = 1/7     ## length of ICU stay before death 
    ψC::Float64 = 1/13.25 ## length of ICU before recovery  
end

function contact_matrix()
    ## contact matrix for general population and in household. 
    M = ones(Float64, 4, 4)
    M[1, :] = [9.786719237, 3.774315965, 1.507919769, 0.603940171]
    M[2, :] = [3.774315965, 9.442271327, 3.044332992, 0.702042998]
    M[3, :] = [1.507919769, 3.044332992, 2.946427003, 0.760366544]
    M[4, :] = [0.603940171, 0.702042998, 0.760366544, 1.247911075]
    Mbar = ones(Float64, 4, 4)
    Mbar[1, :] = [2.039302567, 1.565307565, 0.5035389324, 0.3809355428]
    Mbar[2, :] = [1.565307565, 1.509696249, 0.444748829, 0.2389607652]
    Mbar[3, :] = [0.5035389324, 0.444748829, 1.03553314, 0.1908134302]
    Mbar[4, :] = [0.3809355428, 0.2389607652, 0.1908134302, 0.6410794914]
    return M, Mbar
end

function Model!(du, u, p, t)
    # model v4, with age structure, feb 20
    # 4 age groups: 0 - 18, 19 - 49, 50 - 65, 65+
    S₁, S₂, S₃, S₄, 
    E₁, E₂, E₃, E₄, 
    F₁, F₂, F₃, F₄,         
    Iₙ₁, Iₙ₂, Iₙ₃, Iₙ₄, 
    Qₙ₁, Qₙ₂, Qₙ₃, Qₙ₄, 
    Iₕ₁, Iₕ₂, Iₕ₃, Iₕ₄, 
    Qₕ₁, Qₕ₂, Qₕ₃, Qₕ₄, 
    V₁, V₂, V₃, V₄, 
    Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄,    
    Ĩₙ₁, Ĩₙ₂, Ĩₙ₃, Ĩₙ₄,
    Q̃ₙ₁, Q̃ₙ₂, Q̃ₙ₃, Q̃ₙ₄,
    Ĩₕ₁, Ĩₕ₂, Ĩₕ₃, Ĩₕ₄,
    Q̃ₕ₁, Q̃ₕ₂, Q̃ₕ₃, Q̃ₕ₄,
    H₁, H₂, H₃, H₄, 
    C₁, C₂, C₃, C₄, 
    N₁, N₂, N₃, N₄,     
    # internal incidence equations
    Z₁, Z₂, Z₃, Z₄,
    CX₁, CX₂, CX₃, CX₄,
    CY₁, CY₂, CY₃, CY₄,
    DX₁, DX₂, DX₃, DX₄,
    DY₁, DY₂, DY₃, DY₄,
    Dᵥ, Wᵥ = u
    
    # set up the vectors for syntactic sugar 
    S = (S₁, S₂, S₃, S₄)
    E = (E₁, E₂, E₃, E₄)
    F = (F₁, F₂, F₃, F₄)
    Iₙ = (Iₙ₁, Iₙ₂, Iₙ₃, Iₙ₄)
    Qₙ = (Qₙ₁, Qₙ₂, Qₙ₃, Qₙ₄)
    Iₕ = (Iₕ₁, Iₕ₂, Iₕ₃, Iₕ₄) 
    Qₕ = (Qₕ₁, Qₕ₂, Qₕ₃, Qₕ₄)
    V = (V₁, V₂, V₃, V₄)
    Ẽ = (Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄)
    Ĩₙ = (Ĩₙ₁, Ĩₙ₂, Ĩₙ₃, Ĩₙ₄)
    Q̃ₙ = (Q̃ₙ₁, Q̃ₙ₂, Q̃ₙ₃, Q̃ₙ₄)
    Ĩₕ = (Ĩₕ₁, Ĩₕ₂, Ĩₕ₃, Ĩₕ₄)
    Q̃ₕ = (Q̃ₕ₁, Q̃ₕ₂, Q̃ₕ₃, Q̃ₕ₄)
    H = (H₁, H₂, H₃, H₄)
    C = (C₁, C₂, C₃, C₄)
    N = (N₁, N₂, N₃, N₄)
    # internal incidence euqations
    Z = (Z₁, Z₂, Z₃, Z₄)
    CX = (CX₁, CX₂, CX₃, CX₄)
    CY = (CY₁, CY₂, CY₃, CY₄)
    DX = (DX₁, DX₂, DX₃, DX₄)
    DY = (DY₁, DY₂, DY₃, DY₄)
    
    
    #get the contact matrix 
    M, M̃ = contact_matrix()
    
    # constants 
    Nᵥ = 150e6
    #nva::NTuple{4, Float64} = (0.10*Nᵥ, 0.10*Nᵥ, 0.25*Nᵥ, 0.55*Nᵥ)
    pop = (81982665,129596376,63157200,52431193)

    @unpack β, ξ, ν, σ, q, h, f, τ, γ, δ, ϵ, q̃, h̃, f̃, c, c̃, mH, μH, ψH, mC, μC, ψC = p
    for a = 1:4
        # susceptibles
        #println("$(dot(M[a, :], Iₙ))")
        du[a] = -β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + dot(M[a, :], (1 .- ξ).*(Ĩₙ./pop)) + dot(M[a, :], (1 .- ξ).*(Ĩₕ./pop))) - 
                    β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₙ./pop)) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₕ./pop))) -        
                    ν[a]*heaviside(Nᵥ - Dᵥ)  ## removed S[a]
        # exposed E
        du[a+4]  = β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + dot(M[a, :], (1 .- ξ).*(Ĩₙ./pop)) + dot(M[a, :], (1 .- ξ).*(Ĩₕ./pop))) +
                    β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₙ./pop)) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₕ./pop))) - 
                    σ*E[a] - ν[a]*heaviside(Nᵥ - Dᵥ)*E[a]  ## removed E[a]
        #vaccinated, but exposed,  F
        du[a+8] = -σ*F[a] + ν[a]*heaviside(Nᵥ - Dᵥ)*E[a]  ## removed E[a]
        # In class
        du[a+12] = (1 - q[a])*(1 - h[a])*σ*(E[a] + F[a]) - (1 - f)*γ*Iₙ[a] - f*τ*Iₙ[a]
        # Qn class
        du[a+16] = q[a]*(1 - h[a])*σ*(E[a] + F[a]) + f*τ*Iₙ[a] - γ*Qₙ[a] 
        # Ih class
        du[a+20] = (1 - q[a])*h[a]*σ*(E[a] + F[a]) - δ*Iₕ[a]
        # Qh class 
        du[a+24] = q[a]*h[a]*σ*(E[a] + F[a]) - δ*Qₕ[a]
        # vaccine class V 
        du[a+28] = -β*(1 - ϵ[a])*V[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + dot(M[a, :], (1 .- ξ).*(Ĩₙ./pop)) + dot(M[a, :], (1 .- ξ).*(Ĩₕ./pop))) - 
                    β*(1 - ϵ[a])*V[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₙ./pop)) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₕ./pop))) +
                    ν[a]*heaviside(Nᵥ - Dᵥ) ## removed S[a]
        # Ẽ class
        du[a+32] = β*(1 - ϵ[a])*V[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + dot(M[a, :], (1 .- ξ).*(Ĩₙ./pop)) + dot(M[a, :], (1 .- ξ).*(Ĩₕ./pop))) +
                   β*(1 - ϵ[a])*V[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₙ./pop)) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₕ./pop))) - 
                   σ*Ẽ[a]
        # Ĩn class
        du[a+36] = (1 - q̃[a])*(1 - h̃[a])*σ*Ẽ[a] - (1 - f̃)*γ*Ĩₙ[a] - f̃*τ*Ĩₙ[a]
        # Q̃n class
        du[a+40] = q̃[a]*(1 - h̃[a])*σ*Ẽ[a] + f̃*τ*Ĩₙ[a] - γ*Q̃ₙ[a]
        # Ĩh class
        du[a+44] = (1 - q̃[a])*h̃[a]*σ*Ẽ[a] - δ*Ĩₕ[a]
        # Q̃h class
        du[a+48] = q̃[a]*h̃[a]*σ*Ẽ[a] - δ*Q̃ₕ[a]
        # Ha class
        du[a+52] = (1 - c[a])*δ*Iₕ[a] + (1 - c[a])*δ*Qₕ[a] + (1 - c̃[a])*δ*Ĩₕ[a] + (1 - c̃[a])*δ*Q̃ₕ[a] - 
                    (mH*μH + (1 - mH)*ψH)*H[a]                    
        # Ca class
        du[a+56] = c[a]*δ*Iₕ[a] + c[a]*δ*Qₕ[a] + c̃[a]*δ*Ĩₕ[a] + c̃[a]*δ*Q̃ₕ[a] - 
                    (mC*μC + (1 - mC)*ψC)*C[a]
        # Na class 
        du[a+60] = -mC*μC*C[a] - mH*μH*H[a]     

        # Z class to collect cumulative incindece 
        du[a+64] =  β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + dot(M[a, :], (1 .- ξ).*(Ĩₙ./pop)) + dot(M[a, :], (1 .- ξ).*(Ĩₕ./pop))) +
                    β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₙ./pop)) + dot(M̃[a, :], (1 .- ξ).*(Q̃ₕ./pop))) 

        # CX, CY, DX, DY class to calculate hosp, icu cumulative incidence 
        # X hosp, Y ICU
        # split this into alive(CX/CY), dead(DX/DY) as well 
        du[a+68] = ((1 - mH)*ψH)*H[a]  ## CX 
        du[a+72] = ((1 - mC)*ψC)*C[a]  ## CY
        du[a+76] = (mH*μH)*H[a]  ## DX
        du[a+80] = (mC*μC)*C[a]  ## DY
                    
    end
    #du[85] = heaviside(Nᵥ - Dᵥ)*(ν[1]*(S[1] + E[1]) + ν[2]*(S[2] + E[2]) + ν[3]*(S[3] + E[3]) + ν[4]*(S[4] + E[4]))
    du[85] = heaviside(Nᵥ - Dᵥ)*(1 + E[1]) + ν[2]*(1 + E[1]) + ν[3]*(1 + E[1]) + ν[4]*(1 + E[1])
    du[86] = heaviside(Nᵥ - Dᵥ)*(ν[1]*(E[1]) + ν[2]*(E[2]) + ν[3]*(E[3]) + ν[4]*(E[4]))
end

## ODE callback 
condition(u,t,integrator) = t == 70.0
isoutofdomain(u, t, integrator) = u[1] <= 0 || u[2] <= 0 || u[3] <= 0 || u[4] <= 0
function vaccine!(integrator)
    nva = integrator.p.nva
    integrator.p.ν = (0.1*nva/7, 0.1*nva/7, 0.25*nva/7, 0.55*nva/7) # vaccination rate #10 10 25 55            
end
function zerod!(integrator)
    for i = 1:4 
        if integrator.u[i] <= 0 
            integrator.u[i] = 0
        end
    end
end
cb = DiscreteCallback(condition,vaccine!)
cb2= DiscreteCallback(isoutofdomain,zerod!)
cbset = CallbackSet(cb, cb2)

function run_model(p::ModelParameters, nsims=1)
    ## set up ODE time and initial conditions
    tspan = (0.0, 500.0)
    u0 = zeros(Float64, 86) ## total compartments
    sols = []    
    for mc = 1:nsims
        ## reset the initial conditions
       
        u0[1] = u0[61] = 81982665
        u0[2] = u0[62] = 129596376
        u0[3] = u0[63] = 63157200
        u0[4] = u0[64] = 52431193
        # initial infected person
        u0[13] = 1

        # sample the parameters needed per simulation
        p.ν = (0.0, 0.0, 0.0, 0.0)  ## reset vaccine rate. if vaccine is on, the integrator will properly set it based on nva.
        p.δ = 1/(rand(Uniform(2, 5)))
        p.σ = 1/(rand(LogNormal(log(5.2), 0.1)))     
        p.h = (rand(Uniform(0.0085, 0.033)), rand(Uniform(0.0088, 0.034)), rand(Uniform(0.01, 0.042)), rand(Uniform(0.017, 0.066)))
        p.c = (rand(Uniform(0.013, 0.015)), rand(Uniform(0.04, 0.044)), rand(Uniform(0.2, 0.22)), rand(Uniform(0.46, 0.50)))
        p.h̃ = @. (1 - p.ϵ) * p.h
        p.c̃ = @. (1 - p.ϵ) * p.c        
        

        ## solve the ODE model
        print("...sim: $mc, params: $(p.τ), $(p.f), $(p.nva), $(p.ν) \r")
        prob = ODEProblem(Model!, u0, tspan, p)
        sol = solve(prob, Rodas4(autodiff=false), dt=1, adaptive=false, callback=cbset)  ## WORKS
        push!(sols, sol)     
    end
    println("\n simulation scenario finished")
    return sols
end

function run_single()
    # A function that runs a single simulation for testing/calibrating purposes. 
    ## setup parameters (overwrite some default ones if needed)
    p = ModelParameters() 
    p.nva = 5e6
    p.β = 0.037  ## 0.028: R0 ~ 2.0, next generation method
    p.τ = 0.5
    p.f = 0.05
    p.f̃ = 0.05
    sol = run_model(p, 1)[1] ## the [1] is required since runsims returns an array of solutions
    return sol
end

function run_scenarios()
    ## setup parameters  (overwrite some default ones if needed)
  
    p = ModelParameters()   
    ## go over these scenarios for f and taus
    βs = (0.037, 0.044, 0.052) #r0 2.5, 3, 3.5
    fs = (0.05, 0.1, 0.2)
    τs = (0.5, 1)  
    for v in (true, false), β in βs, f in fs, τ in τs
        p.nva = v ? 5e6 : 0.0
        p.β = β
        p.τ = τ  
        p.f = f
        p.f̃ = f
        fldrname = savestr(p); println("working on: $fldrname")
        sols = run_model(p, 100)
        savesims(sols, fldrname)
    end  
end


function savestr(p::ModelParameters)
    ## setup folder name based on model parameters
    taustr = replace(string(p.τ), "." => "")
    fstr = replace(string(p.f), "." => "")
    if p.β == 0.037
        r0 = "r25"
    elseif p.β == 0.044
        r0 = "r30"
    elseif p.β == 0.052
        r0 = "r35"
    else 
        r0 = "undef"
    end

    if p.nva > 0 
        fldrname = "./$r0/wivac/tau$(taustr)_f$(fstr)/"
    else
        fldrname = "./$r0/novac/tau$(taustr)_f$(fstr)/"
    end
    mkpath(fldrname)
end

function savesims(sols, prefix="./")
    nsims = length(sols)
    sus1, sus2, sus3, sus4 = saveclass(1, sols), saveclass(2, sols), saveclass(3, sols), saveclass(4, sols)
    ci1, ci2, ci3, ci4 = saveclass(65, sols), saveclass(66, sols), saveclass(67, sols), saveclass(68, sols)
    cx1, cx2, cx3, cx4 = saveclass(69, sols), saveclass(70, sols), saveclass(71, sols), saveclass(72, sols)
    cy1, cy2, cy3, cy4 = saveclass(73, sols), saveclass(74, sols), saveclass(75, sols), saveclass(76, sols)
    dx1, dx2, dx3, dx4 = saveclass(77, sols), saveclass(78, sols), saveclass(79, sols), saveclass(80, sols)
    dy1, dy2, dy3, dy4 = saveclass(81, sols), saveclass(82, sols), saveclass(83, sols), saveclass(84, sols)
    
    vars = (sus1, sus2, sus3, sus4, ci1, ci2, ci3, ci4,  cx1, cx2, cx3, cx4, cy1, cy2, cy3, cy4, dx1, dx2, dx3, dx4, dy1, dy2, dy3, dy4)
    fns = ("sus1", "sus2", "sus3", "sus4", "ci1", "ci2", "ci3", "ci4", "cx1", "cx2", "cx3", "cx4", "cy1", "cy2", "cy3", "cy4", "dx1", "dx2", "dx3", "dx4", "dy1", "dy2", "dy3", "dy4")
    fns = string.(prefix, "/", fns, ".csv") ## append the csv
    writedlm.(fns, vars, ',')

end


## helper functions to analyze the ODE solution
function saveclass(class, sols)
    nsims = length(sols)
    res = Array{Array{Float64,1}, 1}(undef, nsims)
    for mc = 1:nsims
        res[mc] = getclass(class, sols[mc])
    end
    return hcat(res...)
end

function getclass(x, sol) 
    p = [sol.u[i][x] for i = 1:length(sol.u)]
    return p 
end


function plots(sol)
    ## plots a summary of the solution (susceptibles, infected, H/Q, Exposed, etc...)
    ## susceptibles
    s1 = getclass(1, sol)
    s2 = getclass(2, sol)
    s3 = getclass(3, sol)
    s4 = getclass(4, sol)

    startpop = (s1[1], s2[1], s3[1], s4[4])
    tspop = sum(startpop)

    n1 = getclass(61, sol)
    n2 = getclass(62, sol)
    n3 = getclass(63, sol)
    n4 = getclass(64, sol)
    n = @. n1 + n2 + n3 + n4 
    println("total number of N (totaldeaths) at start: $(n[1]), peak: $(maximum(n)),  end $(n[end])")

    s = @. (s1 + s2 + s3 + s4)
    println("total number of susceptibles at start: $(s[1]), peak: $(maximum(s)),  end $(s[end])")

    e1 = getclass(5, sol)
    e2 = getclass(6, sol)
    e3 = getclass(7, sol)
    e4 = getclass(8, sol)
    e = sum([getclass(i, sol) for i = 5:8])
    ẽ = sum([getclass(i, sol) for i = 33:36])
    println("total number of exposed at start: $(e[1]), peak: $(maximum(e)),  end $(e[end])")
    println("total number of exposed (vaccinated) at start: $(ẽ[1]), peak: $(maximum(ẽ)),  end $(ẽ[end])")

    ## cumulative exposed Z class
    zexp = sum([getclass(i, sol) for i = 65:68])
    
    ## all the infected classes In, Qn, Ih, Qh class numbers 13 to 28
    totali = [getclass(i, sol) for i = 13:28]
    totali = sum(totali) #/ tspop
    println("total number of I (all classes) at start: $(totali[1]), peak: $(maximum(totali)),  end $(totali[end])")

    # separate the infected classes (nonvaccine)
    i_n = sum([getclass(i, sol) for i = 13:16])
    i_h = sum([getclass(i, sol) for i = 21:24])
    q_n = sum([getclass(i, sol) for i = 17:20])
    q_h = sum([getclass(i, sol) for i = 25:28])
    iclass = (i_n + i_h) #./ n
    qclass = (q_n + q_h) #./ n

    ## vaccinated in, ih, qn, qh
    i_n_tilde = sum([getclass(i, sol) for i = 37:40])
    i_h_tilde = sum([getclass(i, sol) for i = 45:48])
    q_n_tilde = sum([getclass(i, sol) for i = 41:44])
    q_h_tilde = sum([getclass(i, sol) for i = 49:52])
    iclass_tilde = (i_n_tilde + i_h_tilde) #./ n
    qclass_tilde = (q_n_tilde + q_h_tilde) #./ n

    ## uses the Z class to calculate incidence
    incidence = calculate_incidence(sol)
    
    ## prevalence of hospital/icu
    hosp = sum([getclass(i, sol) for i = 53:56])
    icu = sum([getclass(i, sol) for i = 57:60])
    total_hosp_icu = hosp + icu
    ratio_symp_hosp = total_hosp_icu ./ totali

    ## cumulative incidence hosp/icu CX, CY, DX, DY
    chospicu = sum([getclass(i, sol) for i = 69:84])
    
    ## vaccination numbers
    totalvacV = sum([getclass(i, sol) for i = 29:32])  ## this is V class. 
    totalvacF = sum([getclass(i, sol) for i = 9:12])   ## F class
    totalvac=totalvacV+totalvacF
    collectedvac = getclass(85, sol)
    wastedvac = getclass(86, sol)

    ## CHANGE in suspectibles, PER WEEK (used to calibrate the vaccine rate. not really needed anymore)
    sus = [getclass(i, sol) for i = 1:4]   ## susceptibles left
    nweek = Int(floor(length(sol.t)/7))
    ws = fill(0.0, 4, nweek)
    for a=1:4
        ## calculate the change in susceptibles everyday
        sinc = sus[a] - circshift(sus[a], -1)
        sinc[end] = sinc[(end - 1)]

        for i = 1:nweek
            ws[a, i] = sum(sinc[(7*i - 6):(7*i)])
        end
    end
    # plotting the above.
    # l = @layout [a b; c d]
    # p1 = bar(1:nweek, ws[1, :], label="sus[1]")
    # p2 = bar(1:nweek, ws[2, :], label="sus[2]")
    # p3 = bar(1:nweek, ws[3, :], label="sus[3]")
    # p4 = bar(1:nweek, ws[4, :], label="sus[4]")
    # display(plot(p1, p2, p3, p4, layout = l, linewidth=3, size=(1000, 600)))    

    l = @layout [a b; c d; e f; g h]
    #p1 = plot(1:length(mean_inc), [prob, szv])
    p1 = plot(sol.t, s, label="susceptibles")
    p2 = plot(sol.t, zexp, label="c exp")

    p3 = plot(sol.t, [e, ẽ], label=["exposed" "exposed (vaccinated)"])
    p4 = plot(sol.t, [iclass, qclass], label=["i class" "q class"])

    p5 = plot(sol.t, [iclass_tilde, qclass_tilde], label=["i class (vac)" "q class(vac)"])
    p6 = plot(sol.t, incidence, label="incidence")

    #p5 = plot(sol.t, [hosp, icu], label=["hosp" "icu"])
    #p6 = plot(sol.t, [chosp, cicu], label=["chosp" "cicu"])
    
    p7 = plot(sol.t, ẽ, label=["e vac"])    
    p8 = plot(sol.t, [totalvac, collectedvac, wastedvac], label=["V+F" "Dv" "Wv"])
    
    #p9 = groupedbar(rand(10,3), bar_position = :dodge)
    #p6 = plot(sol.t, [chosp, cicu], label=["chosp" "cicu"])
    #p7 = plot(sol.t, ratio_symp_hosp, legend=false)
   
    #Plots.scalefontsizes(0.1) #Or some other factor
    plot(p1, p2, p3, p4, p5, p6, p7, p8, layout = l, linewidth=3, size=(1000, 600))
end


function calculate_incidence(sol)
    # calculates incidence using the Z class
    cuminc = sum([getclass(i, sol) for i = 65:68])
    ## calculate incidence
    #_tmp = circshift(cuminc, 1)
    #_tmp[1] = 0
    inc = circshift(cuminc, -1) - cuminc
    inc[end] = inc[(end - 1)]
   
    #println("hello")
    return(inc)
end

## NEGATIVE BINOMIAL SIMULATIONS 
## These functions are not really neccessary anymore. R0 is determined by NextGen Matrix
# define a function calculate the coefficient of variation of y using window size w
function cv_window(y, w)
    ny = length(y)
    w2 = Int(floor(w/2))
    
    ran = (w2+1):(ny-w)
    println("offset datarange:$ran")
    s = zeros(Float64, ny)
    m = zeros(Float64, ny)
        
    for i in ran        
        s[i] = std(y[(i - w2):(i + w2)])
        m[i] = mean(y[(i - w2):(i + w2)])
    end
    cv = s./m
    return (s, m)
end

function simulate_negbinomial(sol)
    # we use the output of the ODE to simulate epi curves, which then go into R estim package for R0
    # this is not really neccessary, can use next generation method for R0 
    # or at the minimum only use the solution curve from ODE to estimate R0 instead of all 500 random curves. 
    # calculate the coefficient of variance
    win = 8
    fvr = calculate_incidence(sol)[1:35] ## take the first hundred days for now
    sr, mr = cv_window(fvr, win)
    cvr = sr./mr
    cvr[(length(fvr) - win + 1):end] .= cvr[(length(fvr) - win)]

    ## Simulating negative binomial, we remove all the NANs and start where we have a a positive cv
    ## compare with the negative binomial we get from the R code. (eventually removing that part)
    nanrows = findall(x -> isnan(x), cvr)
    deleteat!(cvr, nanrows);

    ## need to delete the equivalent amount from fvr
    deleteat!(fvr, nanrows);

    mean_inc = fvr
    szv = @. 1/(cvr^2 - 1/mean_inc)
    #for numerical stability
    szv[isinf.(szv)] .= 100
    szv[szv .<= 0] .= 50

    prob = @.  (szv./(szv.+mean_inc))
    #prob[prob .== 0] .= 0.001

    nb = NegativeBinomial.(szv, prob)  

    sims = rand.(nb, 500)
    sims = transpose(hcat(sims...) )

    # summarize the data and average out the simulations. 
    # need to put in a dataframe to be melted
    simdf = DataFrame(sims)
    insertcols!(simdf, 1, :time => 1:length(fvr))
    insertcols!(simdf, 2, :x0 => fvr)
    

    # melt the data
    simdf_melt = stack(simdf, [Symbol("x$i") for i = 0:500], [:time], variable_name=:mc, value_name=:value);
    insertcols!(simdf_melt, 2, :seed => sort(repeat(1:501, length(fvr)))) ## notice the 501 cuz its 500 sims + the original vector

    simdf_summary = simdf_melt |> @groupby(_.time) |>
        @map({time=key(_), m = mean(_.value), md = median(_.value), qlow = quantile(_.value, 0.025), qhi = quantile(_.value, 0.975)}) |> DataFrame


    ## the ribbon function here ADDS and SUBTRACTS the value from the main series data, so account for that.
    @df simdf_summary plot(:time, [:m], linewidth=3,label="mean of NB sims", ribbon=(simdf_summary.m - simdf_summary.qlow, simdf_summary.qhi - simdf_summary.m))

    ## plot the actual incidence
    display(plot!(fvr, linewidth=2, linestyle=:dash, label="model output"))

    #l = @layout [a  b]
    #p1 = plot(1:length(mean_inc), [prob, szv])
    #p2 = plot(1:length(mean_inc), sims, legend=false)

    #plot(p1, p2,  layout = l)
    #@. dd.cv[24:27]^2 - 1/dd.cases[24:27]
    return(simdf_melt)
end

## latin hypercube sampling, not used.
function lhsu(xmin, xmax, nsample)
    # s=lhsu(xmin,xmax,nsample)
    # LHS from uniform distribution
    # Input:
    #   xmin    : min of data (1,nvar)
    #   xmax    : max of data (1,nvar)
    #   nsample : no. of samples
    # Output:
    #   s       : random sample (nsample,nvar)
    nvar=length(xmin)
    ran = rand(nsample, nvar)
    s = zeros(Float64, nsample, nvar)
    for j=1:nvar
        idx=randperm(nsample)
        P =(idx - ran[:,j]) / nsample
        s[:,j] = xmin[j] .+ P .* (xmax[j] - xmin[j])
    end
    return s
end

