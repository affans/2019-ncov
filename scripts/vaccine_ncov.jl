## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra

@with_kw mutable struct ModelParameters
    β::Float64 = 0.0 # transmission
    ξ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## reduction in transmission due to meeting vaccinated individual
    ν::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) # vaccination rate 
    σ::Float64 = 0.1923076923 # incubation period, 5.2 days average, real: gamma distributed. 
    q::NTuple{4, Float64} = (0.41, 0.41, 0.41, 0.41) # proportion of self-quarantine
    h::NTuple{4, Float64} = (0.009462204218, 0.0287817404, 0.1670760276, 0.4851364693) # Model weight for going to hospital
    f::Float64 = 0.0 
    γ::Float64 = 0.2173913043  # rate : symptom onset to recovery        
    τ::Float64 = 0.0  # rate : symptom onset to quarantine
    δ::Float64 = 0.0
    ϵ::NTuple{4, Float64} = (0.5, 0.5, 0.5, 0.5) ## vaccine efficacy, why is this age dependent?
    q̃::NTuple{4, Float64} = (0.41, 0.41, 0.41, 0.41) # proportion of self-quarantine (vaccinated)
    h̃::NTuple{4, Float64} = (0.009462204218, 0.0287817404, 0.1670760276, 0.4851364693) 
    f̃::Float64 = 0.0 
    c::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0)
    c̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0)
    mH::Float64 = 0.0 ##  Model weight for hospital death
    μH::Float64 = 0.111 # Hosptial admission to death
    ψH::Float64 = 0.1 # ICU admission to death   
    mC::Float64 = 0.0 ##  Model weight for hospital death
    μC::Float64 = 0.111 # Hosptial admission to death
    ψC::Float64 = 0.1 # ICU admission to death   
end

function contact_matrix()
    ## contact matrix will go in here
    M = ones(Float64, 4, 4)
    #M[1, :] = [9.786719237, 3.774315965, 1.507919769, 0.603940171]
    #M[2, :] = [3.774315965, 9.442271327, 3.044332992, 0.702042998]
    #M[3, :] = [1.507919769, 3.044332992, 2.946427003, 0.760366544]
    #M[4, :] = [0.603940171, 0.702042998, 0.760366544, 1.247911075]
    Mbar = ones(Float64, 4, 4)
    Mbar[1, :] = [2.039302567, 1.565307565, 0.5035389324, 0.3809355428]
    Mbar[2, :] = [1.565307565, 1.509696249, 0.444748829, 0.2389607652]
    Mbar[3, :] = [0.5035389324, 0.444748829, 1.03553314, 0.1908134302]
    Mbar[4, :] = [0.3809355428, 0.2389607652, 0.1908134302, 0.6410794914]

    return M, Mbar
end

∑(x) = sum(x) # synctactic sugar
heaviside(x) = x <= 0 ? 0 : 1

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
    
    #println("S: $(typeof(S)), S1 = $(typeof(S₁))")

    #get the contact matrix 
    M, M̃ = contact_matrix()
    
    # constants 
    Nᵥ = 1e9
    Bh = 33955
    Bc = 5666
    pop = (42000000,66000000,39000000,16000000)
 
    # println("sum $(∑(H))")
    # if isapprox(∑(H), Bh, atol=1)
    #    println("sum reached at time $t)")
    # end
    # if isapprox(∑(C), Bc, atol=1)
    #     println("sum reached at time $t)")
    #  end
    #totalpopsize = ∑(S .+ E .+ F .+ I .+ Q .+ H .+ C .+ V)
    #println("total size: $totalpopsize")

    @unpack β, ξ, ν, σ, q, h, f, τ, γ, δ, ϵ, q̃, h̃, f̃, c, c̃, mH, μH, ψH, mC, μC, ψC = p
    for a = 1:4
        # susceptibles
        #println("$(dot(M[a, :], Iₙ))")
        du[a] = -β*S[a]/pop[a]*(dot(M[a, :], Iₙ) + dot(M[a, :], Iₕ) + dot(M[a, :], (1 .- ξ).*Ĩₙ) + dot(M[a, :], (1 .- ξ).*Ĩₕ)) - 
                    β*S[a]/pop[a]*(dot(M̃[a, :], Qₙ/pop[a]) + dot(M̃[a, :], Qₕ/pop[a]) + dot(M[a, :], (1 .- ξ).*Q̃ₙ) + dot(M[a, :], (1 .- ξ).*Q̃ₕ)) -        
                    ν[a]*S[a]*heaviside(Nᵥ - Dᵥ)
        # exposed E
        du[a+4]  = β*S[a]/pop[a]*(dot(M[a, :], Iₙ) + dot(M[a, :], Iₕ) + dot(M[a, :], (1 .- ξ).*Ĩₙ) + dot(M[a, :], (1 .- ξ).*Ĩₕ)) +
                    β*S[a]/pop[a]*(dot(M̃[a, :], Qₙ) + dot(M̃[a, :], Qₕ) + dot(M[a, :], (1 .- ξ).*Q̃ₙ) + dot(M[a, :], (1 .- ξ).*Q̃ₕ)) - 
                    σ*E[a] - ν[a]*E[a]*heaviside(Nᵥ - Dᵥ)
        #vaccinated, but exposed,  F
        du[a+8] = -σ*F[a] + ν[a]*E[a]*heaviside(Nᵥ - Dᵥ)
        # In class
        du[a+12] = (1 - q[a])*(1 - h[a])*σ*(E[a] + F[a]) - (1 - f)*γ*Iₙ[a] - f*τ*Iₙ[a]
        # Qn class
        du[a+16] = q[a]*(1 - h[a])*σ*(E[a] + F[a]) + f*τ*Iₙ[a] - γ*Qₙ[a] 
        # Ih class
        du[a+20] = (1 - q[a])*h[a]*σ*(E[a] + F[a]) - δ*Iₕ[a]
        # Qh class 
        du[a+24] = q[a]*h[a]*σ*(E[a] + F[a]) - δ*Qₕ[a]
        # vaccine class V 
        du[a+28] = -β*(1 - ϵ[a])*V[a]/pop[a]*(dot(M[a, :], Iₙ) + dot(M[a, :], Iₕ) + dot(M[a, :], (1 .- ξ).*Ĩₙ) + dot(M[a, :], (1 .- ξ).*Ĩₕ)) - 
                    β*(1 - ϵ[a])*V[a]/pop[a]*(dot(M̃[a, :], Qₙ) + dot(M̃[a, :], Qₕ) + dot(M[a, :], (1 .- ξ).*Q̃ₙ) + dot(M[a, :], (1 .- ξ).*Q̃ₕ)) +
                    ν[a]*S[a]*heaviside(Nᵥ - Dᵥ)
        # Ẽ class
        du[a+32] = β*(1 - ϵ[a])*V[a]/pop[a]*(dot(M[a, :], Iₙ) + dot(M[a, :], Iₕ) + dot(M[a, :], (1 .- ξ).*Ĩₙ) + dot(M[a, :], (1 .- ξ).*Ĩₕ)) +
                   β*(1 - ϵ[a])*V[a]/pop[a]*(dot(M̃[a, :], Qₙ) + dot(M̃[a, :], Qₕ) + dot(M[a, :], (1 .- ξ).*Q̃ₙ) + dot(M[a, :], (1 .- ξ).*Q̃ₕ)) - 
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
    end
    du[49] = heaviside(Nᵥ - Dᵥ)*(ν[1]*(S[1] + E[1]) + ν[2]*(S[2] + E[2]) + ν[3]*(S[3] + E[3]) + ν[4]*(S[4] + E[4]))
    du[50] = heaviside(Nᵥ - Dᵥ)*(ν[1]*(E[1]) + ν[2]*(E[2]) + ν[3]*(E[3]) + ν[4]*(E[4]))
end

function main()
    tspan = (0.0, 1000.0)
    u0 = zeros(Float64, 66) ## 50 compartments
    u0[1] = 42000000
    u0[2] = 66000000
    u0[3] = 39000000
    u0[4] = 16000000

    u0[13] = 10

    u0[61] = 42000000 ## N[1] N[2] N[3] N[4]
    u0[62] = 66000000
    u0[63] = 39000000
    u0[64] = 16000000
    p = ModelParameters()
    p.β = 0.05
    #p.σ = 2  ## the smaller this parameter, the longer it takes to reach equilibrium 10000 for 0.01
    #p.γ = 0
    #p.τ = 0
    prob = ODEProblem(Model!, u0, tspan, p)
    alg = Tsit5()
    sol = solve(prob, Rodas4(autodiff=false), dt=0.5, adaptive=false)  ## WORKS
    #sol = solve(prob, Midpoint(), dt=0.01, adaptive=true)  ## WORKS but too many points
    #sol = solve(prob, Midpoint(), saveat=1, adaptive=true) 
    #sol = solve(prob)
    
    return sol
end

## helper functions to analyze the ODE solution
function getclass(x, sol) 
    p = [sol.u[i][x] for i = 1:length(sol.u)]
    return p 
end

function plotclass(c, sol)
    p = [sol.u[i][c] for i = 1:length(sol.u)]
    plot(p)
end

function _testparameters(p::ModelParameters) 
    error("not implemented")    
end

function _summary(p::ModelParameters)
    error("not implemented")
end

#sol = main()
