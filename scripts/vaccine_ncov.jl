## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra

@with_kw mutable struct ModelParameters
    β::Float64 = 0.0 ## transmission   
    ξ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## vaccination rate
    ν::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## vaccination rate
    σ::Float64 = 0.0 ## incubation period
    q::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    γ::Float64 = 0.0
    τ::Float64 = 0.0
    h::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    c::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    δ::Float64 = 0.0
    θ::Float64 = 0.0
    μ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    ω::Float64  = 0.0
    mH::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    μH::Float64 = 0.0
    ϕH::Float64 = 0.0
    mC::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    μC::Float64 = 0.0
    ϕC::Float64 = 0.0
    ϵ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## vaccine efficacy
    ## vaccination parameters
    q̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    h̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    c̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    μₜ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
end


function contact_matrix()
    ## contact matrix will go in here
    M = ones(Int64, 4, 4)
    Mbar = ones(Int64, 4, 4)
    return M, Mbar
end

∑(x) = sum(x) # synctactic sugar

function Model!(du, u, p, t)
    # model v3, with age structure, feb 19, 1:20pm latest file
    # 4 age groups: 0 - 18, 19 - 49, 50 - 65, 65+
    S₁, S₂, S₃, S₄, 
    E₁, E₂, E₃, E₄, 
    F₁, F₂, F₃, F₄,     
    I₁, I₂, I₃, I₄, 
    Q₁, Q₂, Q₃, Q₄, 
    H₁, H₂, H₃, H₄, 
    C₁, C₂, C₃, C₄, 
    V₁, V₂, V₃, V₄, 
    Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄, 
    Ĩ₁, Ĩ₂, Ĩ₃, Ĩ₄,
    Q̃₁, Q̃₂, Q̃₃, Q̃₄,
    N₁, N₂, N₃, N₄, 
    Dᵥ, Wᵥ = u

    # set up the vectors for syntactic sugar 
    S = (S₁, S₂, S₃, S₄)
    E = (E₁, E₂, E₃, E₄)
    F = (F₁, F₂, F₃, F₄)
    I = (I₁, I₂, I₃, I₄)
    Q = (Q₁, Q₂, Q₃, Q₄)
    H = (H₁, H₂, H₃, H₄)
    C = (C₁, C₂, C₃, C₄)
    V = (V₁, V₂, V₃, V₄)
    Ẽ = (Ẽ₁, Ẽ₂, Ẽ₃, Ẽ₄)
    Ĩ = (Ĩ₁, Ĩ₂, Ĩ₃, Ĩ₄)
    Q̃ = (Q̃₁, Q̃₂, Q̃₃, Q̃₄)
    N = (N₁, N₂, N₃, N₄)
    

    println("S: $(typeof(S)), S1 = $(typeof(S₁))")
#    error("errored out")

    #get the contact matrix 
    M, Mbar = contact_matrix()
    
    # contants 
    Nᵥ = 1e9
    Bh = 10000
    Bc = 10000
    
    #unpack the parameters 
    @unpack β, ξ, ν, σ, q, γ, τ, h, c, δ, θ, μ, ω, mH, μH, ϕH, mC, μC, ϕC, ϵ, q̃, h̃, c̃, μₜ = p
    for a = 1:4
        #susceptble S₁, S₂, S₃, S₄ 
        du[a]    = -β*S[a]/N[a]*(dot(M[a, :], I) + dot(M[a, :], (1 .- ξ).*Ĩ) + dot(Mbar[a, :], Q) + dot(Mbar[a, :], (1 .- ξ).*Q̃)) 
                    - ν[a]*S[a]*(1 - Dᵥ/Nᵥ)
        #exposed E
        du[a+4]  = β*S[a]/N[a]*(dot(M[a, :], I) + dot(M[a, :], (1 .- ξ).*Ĩ) + dot(Mbar[a, :], Q) + dot(Mbar[a, :], (1 .- ξ).*Q̃)) 
                    - σ*E[a] - ν[a]*E[a]*(1 - Dᵥ/Nᵥ)
        #vaccinated, but exposed,  F
        du[a+8] = -σ*F[a] + ν[a]*E[a]*(1 - Dᵥ/Nᵥ)
        # infected class I
        du[a+12] = (1 - q[a])*σ*(E[a] + F[a]) - 
                    γ*(γ/(γ + τ))*I[a]*(1 - h[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                    τ*(τ/(γ + τ))*I[a]*(1 - h[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                    δ*h[a]*I[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - 
                    θ*c[a]*I[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc) - 
                    μ[a]*I[a]*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc)) 
        # quarantined class Q
        du[a+16] = q[a]*σ*(E[a] + F[a]) - 
                    δ*h[a]*Q[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - 
                    θ*c[a]*Q[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc) - 
                    γ*Q[a]*(1 - h[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                    μ[a]*Q[a]*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc)) + 
                    τ*(τ/(γ + τ))*I[a]*(1 - h[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc))  
        # hospitalized class H
        du[a + 20] = h[a]*δ*(1 - ∑(H)/Bh)*(I[a] + Q[a]) - (mH[a]*μH + (1 - mH[a])*ϕH)*H[a]
        # ICU class C 
        du[a + 24] = c[a]*θ*(1 - ∑(C)/Bc)*(I[a] + Q[a]) - (mC[a]*μC + (1 - mC[a])*ϕC)*C[a]
        # vaccinated V
        du[a + 28] = -β*(1 - ϵ[a])*V[a]/N[a]*(dot(M[a, :], I) + dot(M[a, :], (1 .- ξ).*Ĩ) + dot(Mbar[a, :], Q) + dot(Mbar[a, :], (1 .- ξ).*Q̃)) + 
                    ν[a]*S[a]*(1 - Dᵥ/Nᵥ)                        
        # exposed, vaccinated Ẽ
        du[a + 32] = β*(1 - ϵ[a])*V[a]/N[a]*(dot(M[a, :], I) + dot(M[a, :], (1 .- ξ).*Ĩ) + dot(Mbar[a, :], Q) + dot(Mbar[a, :], (1 .- ξ).*Q̃))  - 
                    σ*Ẽ[a] 
        # infected, vaccinated I tilde
        du[a + 36] = (1 - q̃[a])*σ*Ẽ[a] - 
                        γ*(γ/(γ + τ))*Ĩ[a]*(1 - h̃[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c̃[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                        τ*(τ/(γ + τ))*Ĩ[a]*(1 - h̃[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c̃[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                        δ*h̃[a]*Ĩ[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - 
                        θ*c̃[a]*Ĩ[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc) -                         
                        μₜ[a]*Ĩ[a]*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc))   
        # quarantined, vaccinated Q tilde               
        du[a + 40] = q̃[a]*σ*Ẽ[a] + τ*(τ/(γ + τ))*Ĩ[a]*(1 - h̃[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c̃[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                        γ*Q̃[a]*(1 - h̃[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - c̃[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc)) - 
                        δ*h̃[a]*Q̃[a]*(1 - (H[1]+H[2]+H[3]+H[4])/Bh) - 
                        θ*c̃[a]*Q̃[a]*(1 - (C[1]+C[2]+C[3]+C[4])/Bc) -                  
                        μₜ[a]*Q̃[a]*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc))                         
        # Na 
        du[a + 44] = -mC[a]*μC*C[a] - mH[a]*μH*H[a] - 
                        μ[a]*(I[a] + Q[a])*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc)) - 
                        μₜ[a]*(Ĩ[a] + Q̃[a])*((1 - ω)*(∑(H)/Bh) + ω*(∑(C)/Bc))
    end
    du[49] = ν[1]*(1 - Dᵥ/Nᵥ)*(S[1] + E[1]) + 
             ν[2]*(1 - Dᵥ/Nᵥ)*(S[2] + E[2]) + 
             ν[3]*(1 - Dᵥ/Nᵥ)*(S[3] + E[3]) +
             ν[4]*(1 - Dᵥ/Nᵥ)*(S[4] + E[4]) 
    du[50] = ν[1]*(1 - Dᵥ/Nᵥ)*(E[1]) + 
             ν[2]*(1 - Dᵥ/Nᵥ)*(E[2]) + 
             ν[3]*(1 - Dᵥ/Nᵥ)*(E[3]) +
             ν[4]*(1 - Dᵥ/Nᵥ)*(E[4])              
end

function main()
    tspan = (0.0,100.0)
    u0 = zeros(Float64, 50) ## 50 compartments
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

sol = main()
