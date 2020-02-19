## 2019-nCov Vaccination/COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra

@with_kw mutable struct ModelParameters
    β::Float64 = 0.0 ## transmission to be calibrated
    ξ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## reduction in transmission due to meeting vaccinated individual
    ν::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) # vaccination rate 
    σ::Float64 = 0.1923076923 # incubation period, 5.2 days average, real: gamma distributed. 
    q::NTuple{4, Float64} = (0.41, 0.41, 0.41, 0.41) # proportion of self-quarantine
    γ::Float64 = 0.2173913043  # rate of symptom onset to recovery
    τ::Float64 = 0.0  # Early case finding rate
    h::NTuple{4, Float64} = (0.009462204218, 0.0287817404, 0.1670760276, 0.4851364693) # Model weight for going to hospital
    c::NTuple{4, Float64} = (0.003658999482, 0.01112979289, 0.06460768383, 0.1876004839) # Model weight for going to ICU
    δ::Float64 = 0.1428571429 # rate of symptom onset to hospitalization
    θ::Float64 = 0.125 # Rate from onset to ICU
    μ::NTuple{4, Float64} = (0.0002255096518, 0.0006942592474, 0.004412973677, 0.01639611258) ## age dependent death rate, General Pop
    ω::Float64  = 1.0   # Proportion of deaths in ICU relative to hosptial and ICU
    mH::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ##  Model weight for hospital death
    μH::Float64 = 0.111 # Hosptial admission to death
    ϕH::Float64 = 0.1 # ICU admission to death   
    mC::NTuple{4, Float64} = (0.3950958655, 0.3950958655, 0.3950958655, 0.3950958655) ## Model weight for ICU death
    μC::Float64 = 0.111
    ϕC::Float64 = 0.1
    ϵ::NTuple{4, Float64} = (0.5, 0.5, 0.5, 0.5) ## vaccine efficacy, why is this age dependent?
    ## vaccination parameters
    q̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    h̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    c̃::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
    μₜ::NTuple{4, Float64} = (0.0, 0.0, 0.0, 0.0) ## 
end


function contact_matrix()
    ## contact matrix will go in here
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
    Bh = 33955
    Bc = 5666
    
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
    u0[1] = 163000000
    u0[5] = 10
    u0[45] = 42000000 ## N[1] N[2] N[3] N[4]
    u0[46] = 66000000
    u0[47] = 39000000
    u0[48] = 16000000
    p = ModelParameters()
    p.β = 1.0
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
