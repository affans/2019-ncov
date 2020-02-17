## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV

@with_kw struct myPara1
    β::Float64  
    M::Float64 
    ξ::Float64 
    ν::Float64 
    σ::Float64 
    τ::Float64 
    q::Float64 
    h::Float64 
    γ::Float64 
    ω::Float64 
    δ::Float64 
    hq::Float64
    ρ::Float64
    ζ::Float64
    m::Float64
    μ::Float64
    ϕ::Float64
    ϵ::Float64 
    qbar::Float64
    hbar::Float64
    hqbar::Float64
    mbar::Float64
end

function Model!(du,u,p,t) 
    Nv = 1e9
    S,E,F,I,O,Q,G,H,V, Ev,Iv,Ov,Qv,Gv,Hv,Dv = u
    @unpack β, M, ξ, ν, σ, τ, q, h, γ, ω, δ, hq, ρ, ζ, m, μ, ϕ, ϵ, qbar, hbar, hqbar, mbar = p

    du[1] = dS = -β*S*M*(I+(1 - ξ)*Iv) - ν*S*(1 - Dv/Nv)
    du[2] = dE = β*S*M*(I+(1 - ξ)*Iv) - σ*E - τ*E*(M*(H+Hv+Q+Qv)) - ν*E*(1 - Dv/Nv)
    du[3] = dF = -σ*F - τ*F*(M*(H+Hv+Q+Qv)) + ν*E*(1 - Dv/Nv)
    du[4] = dI = σ*(E + F) - (1 - q - h)*γ*I - q*ω*I - h*δ*I - τ*I*(M*(H+Hv+Q+Qv))
    du[5] = dO = τ*(F + E)*(M*(H+Hv+Q+Qv)) - σ*O
    du[6] = dQ = q*ω*I + σ*O - (1 - hq)*ρ*Q - hq*ζ*Q + τ*I*(M*(H+Hv+Q+Qv))
    du[7] = dG = hq*ζ*Q - m*μ*G - (1 - m)*ϕ*G
    du[8] = dH = h*δ*I - m*μ*H - (1 - m)*ϕ*H

    du[9] = dV = ν*S*(1 - Dv/Nv) - β*(1 - ϵ)*V*M*(I+(1 - ξ)*Iv)
    
    du[10] = dEv = β*(1-ϵ)*V*M*(I+(1 - ξ)*Iv) - σ*Ev - τ*Ev*(M*(H+Hv+Q+Qv)) - ν*E*(1 - Dv/Nv)
    du[11] = dIv = σ*Ev - (1 - qbar - hbar)*γ*Iv - qbar*ω*Iv - hbar*δ*Iv - τ*Iv*(M*(H+Hv+Q+Qv))
    du[12] = dOv = τ*Ev*(M*(H+Hv+Q+Qv)) - σ*Ov
    du[13] = dQv = qbar*ω*Iv + σ*Ov - (1 - hqbar)*ρ*Qv - hqbar*ζ*Qv + τ*Iv*(M*(H+Hv+Q+Qv))
    du[14] = dGv = hqbar*ζ*Qv - mbar*μ*Gv - (1 - mbar)*ϕ*Gv
    du[15] = dHv = hbar*δ*Iv - mbar*μ*Hv - (1 - mbar)*ϕ*Hv
    du[16] = (1 - Dv/Nv)*(μ*S + μ*E)
end

#prob = ODEProblem(Model!, u0, tspan, myPara(0.01, 0.1))
#sol = solve(prob)
#plot(sol)
