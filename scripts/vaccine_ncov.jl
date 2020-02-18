## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra

## original model (with no age structure)
## not neccessary anymore
@with_kw mutable struct Model1_Parameters
    β::Float64   = 0.0
    M::Float64   = 0.0    
    ξ::Float64   = 1/(10 - 4.6)
    ν::Float64    = 0.0                    
    σ::Float64    = 1/5.2                  ## incubation rate
    τ::Float64    = 0.0                    ## contact tracing efficacy
    q::Float64    = 0.0                    ## weight for symptomatic cases being quarantined  (calculated!)
    h::Float64    = 0.0                    ## weight for symptomatic cases being hospitalized (calculated!)
    γ::Float64    = 1/(2*(7.5 - 5.2))      ## rate of symptom onset to recovery
    ω::Float64    = 1/4.6                  ## rate of symptom onset to self-quarantine
    δ::Float64    = 1/10                   ## rate of symptom onset to hospitalization
    hq::Float64   = 0.0                    ## weight for quarantine to recovery (calculated!)
    ρ::Float64    = 1/14                   ## rate of quarantine to recovery 
    ζ::Float64    = 0.0                    ## rate of quarantine to hospitalization
    m::Float64    = 0.0                    ## weight for mortality in hospital for unvaccinated (calculated!)
    μ::Float64    = 1/(23 - 10)            ## rate of hospitalization to death
    ψ::Float64    = 1/10                   ## rate of hospitalization to recovery
    ϵ::Float64    = 0.0                    ## vaccine efficacy (AGE DEPENDENT)
    qbar::Float64  = 0.0                   ## weight for symptomatic (vaccinated) cases being quarantined  (calculated!)
    hbar::Float64  = 0.0                   ## weight for symptomatic (vaccinated) cases being hospitalized (calculated!)
    hqbar::Float64 = 0.0                   ## weight for quarantine (vaccinated) to recovery (calculated!) 
    mbar::Float64  = 0.0                   ## weight for mortality (vaccinated) in hospital for unvaccinated (calculated!)  
    pm::Float64    = 6/47                  ## intermediate value to calculate m 
    ph::Float64    = 0.05                  ## intermediate value to calculate h 
    pq::Float64    = 0.14                  ## intermediate value to calculate q 
    phq::Float64   = 0.26                   ## intermediate value to calculate hq
end

function m_weight(par::Model1_Parameters)
    @unpack ψ, pm, μ = par
    val = (pm*ψ)/(pm*ψ + (1 - ψ)μ)
    par.m = val     
end 

function hq_weight(par::Model1_Parameters)
    @unpack ph, pq, phq, ω, γ, δ, ζ, ρ = par
    hval = (ph*ω*γ)/(δ*((1 - pq)*ω + pq*γ) + ph*ω*(γ - δ))
    qval = (pq*(hval*δ + (1 - hval)*γ))/((1 - pq)*ω + pq*γ)
    hqval = (phq*ρ)/(phq*ρ + (1 - phq)*ζ)
    par.h = hval
    par.q = qval
    par.hq = hqval
end

function model1_setup_parameters() 
    p = Model1_Parameters()
    
    # set beta  
    p.β = 0.001 
    p.M = 0.1111  #try 3.43e-7

    ## setup m, h, q, hq weights
    m_weight(p); hq_weight(p); 
    
    ## set the vaccinated rates to be the same for now
    p.qbar = p.q 
    p.hbar = p.h 
    p.hqbar = p.hq 
    p.mbar = p.m

    ## temporary set rates 
    p.ν = 0.05
    p.ζ = 0.01
    p.ϵ = 0.5
    
    # check for 0 entries    
    for x in propertynames(p)
        getproperty(p, x) == 0 && println("$x is set to zero")
    end

    return p
end

function Model1!(du,u,p,t) 
    Nv = 1e9
    S,E,F,I,O,Q,G,H,V, Ev,Iv,Ov,Qv,Gv,Hv,Dv = u
    @unpack β, M, ξ, ν, σ, τ, q, h, γ, ω, δ, hq, ρ, ζ, m, μ, ψ, ϵ, qbar, hbar, hqbar, mbar = p

    du[1] = dS = -β*S*M*(I+(1 - ξ)*Iv) - ν*S*(1 - Dv/Nv)
    du[2] = dE = β*S*M*(I+(1 - ξ)*Iv) - σ*E - τ*E*(M*(H+Hv+Q+Qv)) - ν*E*(1 - Dv/Nv)
    du[3] = dF = -σ*F - τ*F*(M*(H+Hv+Q+Qv)) + ν*E*(1 - Dv/Nv)
    du[4] = dI = σ*(E + F) - (1 - q - h)*γ*I - q*ω*I - h*δ*I - τ*I*(M*(H+Hv+Q+Qv))
    du[5] = dO = τ*(F + E)*(M*(H+Hv+Q+Qv)) - σ*O
    du[6] = dQ = q*ω*I + σ*O - (1 - hq)*ρ*Q - hq*ζ*Q + τ*I*(M*(H+Hv+Q+Qv))
    du[7] = dG = hq*ζ*Q - m*μ*G - (1 - m)*ψ*G
    du[8] = dH = h*δ*I - m*μ*H - (1 - m)*ψ*H

    du[9] = dV = ν*S*(1 - Dv/Nv) - β*(1 - ϵ)*V*M*(I+(1 - ξ)*Iv)
    
    du[10] = dEv = β*(1-ϵ)*V*M*(I+(1 - ξ)*Iv) - σ*Ev - τ*Ev*(M*(H+Hv+Q+Qv)) 
    du[11] = dIv = σ*Ev - (1 - qbar - hbar)*γ*Iv - qbar*ω*Iv - hbar*δ*Iv - τ*Iv*(M*(H+Hv+Q+Qv))
    du[12] = dOv = τ*Ev*(M*(H+Hv+Q+Qv)) - σ*Ov
    du[13] = dQv = qbar*ω*Iv + σ*Ov - (1 - hqbar)*ρ*Qv - hqbar*ζ*Qv + τ*Iv*(M*(H+Hv+Q+Qv))
    du[14] = dGv = hqbar*ζ*Qv - mbar*μ*Gv - (1 - mbar)*ψ*Gv
    du[15] = dHv = hbar*δ*Iv - mbar*μ*Hv - (1 - mbar)*ψ*Hv
    du[16] = (1 - Dv/Nv)*(μ*S + μ*E)
end

function run_model1()
    tspan = (0.0,100.0)
    u0 = Float64.([10000, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p = model1_setup_parameters()
    prob = ODEProblem(Model1!, u0, tspan, p)
    sol = solve(prob)
    plot(sol)
    return sol
end
