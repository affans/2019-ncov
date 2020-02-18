## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra

@with_kw mutable struct ModelParameters
    β::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## transmission       
    ν::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## vaccination rate
    σ::Float64 = 0.0 ## incubation period
    τ::Float64 = 0.0 ## contact tracing parameter
    ϵ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## age specific vaccine efficacy
    q::Float64 = 0.0 ## ???
    h::Float64 = 0.0 ## ???
    c::Float64 = 0.0 ## ???
    γ::Float64 = 0.0 ## ???
    δ::Float64 = 0.0 ## ???
    θ::Float64 = 0.0 ## ???
    μ::Float64 = 0.0 ## ??
    ω::Float64 = 0.0  ## ?
    mh::Float64 = 0.0 ## ??
    μh::Float64 = 0.0 ## ??
    ϕh::Float64 = 0.0 ## ??
    mc::Float64 = 0.0 ## ??
    μc::Float64 = 0.0 ## ??
    ϕc::Float64 = 0.0 ## ??
end

function contact_matrix()
    ## contact matrix will go in here
    M = ones(Int64, 4, 4)
    Mbar = ones(Int64, 4, 4)
    return M, Mbar
end

function Model!(du, u, p, t)
    # simplified model v2, with age structure
    # 4 age groups: 0 - 18, 19 - 49, 50 - 65, 65+
    S₁, S₂, S₃, S₄, 
    E₁, E₂, E₃, E₄, 
    V₁, V₂, V₃, V₄, 
    F₁, F₂, F₃, F₄, 
    I₁, I₂, I₃, I₄, 
    O₁, O₂, O₃, O₄, 
    Q₁, Q₂, Q₃, Q₄, 
    H₁, H₂, H₃, H₄, 
    C₁, C₂, C₃, C₄, Dᵥ = u

    # set up the vectors for syntactic sugar 
    S = (S₁, S₂, S₃, S₄)
    E = (E₁, E₂, E₃, E₄)
    V = (V₁, V₂, V₃, V₄)
    F = (F₁, F₂, F₃, F₄)
    I = (I₁, I₂, I₃, I₄)
    O = (O₁, O₂, O₃, O₄)
    Q = (Q₁, Q₂, Q₃, Q₄)
    H = (H₁, H₂, H₃, H₄)
    C = (C₁, C₂, C₃, C₄)

    println("S: $(typeof(S)), S1 = $(typeof(S₁))")
#    error("errored out")

    #get the contact matrix 
    M, Mbar = contact_matrix()
    
    # contants 
    Nᵥ = 1e9
    Bh = 10000
    Bc = 10000
    
    #unpack the parameters 
    @unpack β, ν, σ, τ, ϵ, q, h, c, γ, δ, θ, μ, ω, mh, μh, ϕh, mc, μc, ϕc  = p
    for a = 1:4
        #susceptble S₁, S₂, S₃, S₄ 
        du[a]    = -β[a]*S[a]*(dot(M[a, :], I) + dot(Mbar[a, :], Q)) - ν[a]*S[a]*(1 - Dᵥ/Nᵥ)
        #exposed E
        du[a+4]  = β[a]*S[a]*(dot(M[a, :], I) + dot(Mbar[a, :], Q)) - σ*E[a] - τ*E[a]*(dot(M[a, :], (Q.+H.+C))) - ν[a]*E[a]*(1 - Dᵥ/Nᵥ)
        #vaccinated V
        du[a+8]  = -β[a]*(1 - ϵ[a])*V[a]*(dot(M[a, :], I) + dot(Mbar[a, :], Q)) + ν[a]*S[a]*(1 - Dᵥ/Nᵥ)
        #vaccinated, but exposed F
        du[a+12] = β[a]*(1 - ϵ[a])*V[a]*(dot(M[a, :], I) + dot(Mbar[a, :], Q)) - σ*F[a] - τ*E[a]*(dot(M[a, :], (Q.+H.+C))) + ν[a]*E[a]*(1 - Dᵥ/Nᵥ)
        #infected class I
        du[a+16] = (1 - q)*σ*(E[a] + F[a]) - 
                        (1 - h*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c*(1 - (C[1]+C[2]+C[3]+C[4])/Bc))*γ*I[a] - 
                        h*(1 - (H[1]+H[2]+H[3]+H[4])/Bh)*δ*I[a] - 
                        c*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)*θ*I[a] - 
                        τ*I[a]*(dot(M[a, :], (Q.+H.+C))) - 
                        μ*I[a]*((H[1] + C[1] + H[2] + C[2] + H[3] + C[3] + H[4] + C[4])/(Bh + Bc))
        #Oa class
        du[a+20] = τ*(F[a] + E[a])*(dot(M[a, :], (Q.+H.+C))) - ω*O[a]
        #infected but quarantined Q class
        du[a+24] = q*σ*(E[a] + F[a]) + σ*O[a] + τ*I[a]*(dot(M[a, :], (Q.+H.+C))) -                        
                        h*(1 - (H[1]+H[2]+H[3]+H[4])/Bh)*δ*Q[a] - 
                        c*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)*θ*Q[a] - 
                        (1 - h*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c*(1 - (C[1]+C[2]+C[3]+C[4])/Bc))*γ*Q[a] - 
                        μ*Q[a]*((H[1] + C[1] + H[2] + C[2] + H[3] + C[3] + H[4] + C[4])/(Bh + Bc))
        #hospitalized H class
        du[a+28] = h*δ*(1 - (H[1]+H[2]+H[3]+H[4])/Bh)*(I[a] + Q[a]) - (mh*μh + (1 - mh)*ϕh)*H[a]
        #C class 
        du[a+32] = c*θ*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)*(I[a] + Q[a]) - (mc*μc + (1 - mc)*ϕc)*C[a] 
    end
    du[37] = ν[1]*(1 - Dᵥ/Nᵥ)*(S[1] + E[1]) + 
             ν[2]*(1 - Dᵥ/Nᵥ)*(S[2] + E[2]) + 
             ν[3]*(1 - Dᵥ/Nᵥ)*(S[3] + E[3]) +
             ν[4]*(1 - Dᵥ/Nᵥ)*(S[4] + E[4]) 
end

function main()
    tspan = (0.0,100.0)
    u0 = zeros(Float64, 37) ## 37 compartments
    u0[1] = 10000
    p = ModelParameters()
    prob = ODEProblem(Model!, u0, tspan, p)
    sol = solve(prob)
    #plot(sol)
    return sol
end

function _testparameters(p::ModelParameters) 
    error("not implemented")    
end

function _summary(p::ModelParameters)
    error("not implemented")
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
