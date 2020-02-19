


## test functions, please ignore.
function HINDAWI!(du,u,p,t) 
    # model with logistic S growth, nonlinear incidence 
    # https://www.hindawi.com/journals/complexity/2019/9876013/
    S, I = u   
    r = 10
    K = 100
    β = 0.1 
    α = 0.01
    Φ = 2.3
    λ = 68.66
    ϵ = 0.5

    du[1] = dS = r*S*(1 - S/K) - (β*S*I)/(1 + α*I)
    du[2] = dI = (β*S*I)/(1 + α*I) - Φ*I - λ*I/(1 + ϵ*I)
    #du[3] = σ*I + λ*I/(1 + ϵ*I) - μ*R
end

function _runhindawi()
    tspan = (0.0,1.0)
    u0 = Float64.([50, 10]) 
    prob = ODEProblem(HINDAWI!, u0, tspan)    
    sol = solve(prob)
    plot(sol)
end

function testSIR!(du,u,p,t) 
    S₁, S₂, S₃, I₁, I₂, I₃, R = u
    β,γ = p
    M = (0.01, 0.04, 0.05)
    S = (S₁, S₂, S₃)
    I = (I₁, I₂, I₃)
    for i = 1:3
        du[i] = -β*S[i]*(dot(M, I))           
        du[i + 3] = β*S[i]*(dot(M, I)) - γ*I[i] 
    end
    #du[1] = dS₁ = -β*S₁*(dot(M, I))
    #du[2] = dS₂ = -β*S₂*(dot(M, I))
    #du[3] = dS₃ = -β*S₃*(dot(M, I))    
    #du[4] =  β*S₁*(dot(M, I)) - γ*I₁
    #du[5] =  β*S₂*(dot(M, I)) - γ*I₂
    #du[6] =  β*S₃*(dot(M, I)) - γ*I₃   
     du[7] = dR = γ*(I₁ + I₂ + I₃)
end


function _runtest()
    tspan = (0.0,100.0)
    u0 = Float64.([100, 100, 100, 1, 0, 0, 0])    
    prob = ODEProblem(testSIR!, u0, tspan, [0.01, 0.001])    
    sol = solve(prob)
    plot(sol)
end