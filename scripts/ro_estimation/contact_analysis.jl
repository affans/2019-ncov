## contact analysis 2019-nCov 
## Affan Shoukat, Feb 2020

## This script calculates the percent reduction required in contacts to bring R0 down to 1. 

using Distributions, Random, Plots

function sample_serial_interval(n = 1)
    k = 4.86592 # shape
    θ = 1.54133 # scale
    dist = Gamma(k, θ)
    rand(dist, n)
end

function contact_matrix()
    CM = Array{Array{Float64, 1}, 1}(undef, 15)
    CM[1] = [0.2287, 0.1153, 0.0432, 0.0254, 0.0367, 0.0744, 0.1111, 0.0992, 0.0614, 0.0391, 0.0414, 0.0382, 0.032, 0.0234, 0.0305]
    CM[2] = [0.0579, 0.4287, 0.0848, 0.0213, 0.0175, 0.042, 0.0622, 0.0772, 0.0723, 0.0333, 0.0262, 0.0205, 0.0192, 0.0138, 0.0231]
    CM[3] = [0.0148, 0.0616, 0.5082, 0.0885, 0.0219, 0.0212, 0.032, 0.0542, 0.0717, 0.0464, 0.0256, 0.0147, 0.0098, 0.0096, 0.0198]
    CM[4] = [0.01, 0.0202, 0.0758, 0.5002, 0.0804, 0.037, 0.0287, 0.0443, 0.0549, 0.0663, 0.0341, 0.0185, 0.0087, 0.007, 0.0139]
    CM[5] = [0.0191, 0.0155, 0.0203, 0.1107, 0.2624, 0.1342, 0.0932, 0.0668, 0.0604, 0.0754, 0.0616, 0.0334, 0.0157, 0.0092, 0.0221]
    CM[6] = [0.0376, 0.0339, 0.02, 0.0465, 0.126, 0.1847, 0.1238, 0.0895, 0.0742, 0.069, 0.076, 0.0497, 0.0283, 0.0141, 0.0267]
    CM[7] = [0.0554, 0.0623, 0.0444, 0.0362, 0.0536, 0.1071, 0.1592, 0.1372, 0.1001, 0.0677, 0.0586, 0.0426, 0.0328, 0.0175, 0.0253]
    CM[8] = [0.0438, 0.0648, 0.0501, 0.0354, 0.0472, 0.0778, 0.1034, 0.1613, 0.1343, 0.0838, 0.0658, 0.0374, 0.0381, 0.0275, 0.0293]
    CM[9] = [0.0462, 0.0473, 0.0616, 0.0669, 0.0691, 0.0679, 0.0929, 0.1046, 0.1441, 0.1045, 0.0772, 0.0306, 0.0287, 0.0204, 0.0380]
    CM[10] = [0.0237, 0.0254, 0.0409, 0.09, 0.0658, 0.0763, 0.0867, 0.0984, 0.1122, 0.1369, 0.0962, 0.0512, 0.0301, 0.0157, 0.0505]
    CM[11] = [0.0184, 0.0356, 0.0487, 0.0585, 0.0766, 0.098, 0.0804, 0.0811, 0.0922, 0.1018, 0.1109, 0.0818, 0.0439, 0.0183, 0.0538]
    CM[12] = [0.0231, 0.0294, 0.0309, 0.0416, 0.0543, 0.0766, 0.0896, 0.0776, 0.0933, 0.0834, 0.1025, 0.1293, 0.0687, 0.0395, 0.0602]
    CM[13] = [0.0312, 0.0245, 0.0294, 0.0295, 0.0424, 0.0676, 0.0955, 0.0923, 0.0879, 0.0696, 0.0722, 0.1013, 0.1063, 0.0659, 0.0844]
    CM[14] = [0.0212, 0.0357, 0.0251, 0.0234, 0.0346, 0.066, 0.0779, 0.0979, 0.0975, 0.0637, 0.0673, 0.0741, 0.0945, 0.1021, 0.119]
    CM[15] = [0.0202, 0.0276, 0.0508, 0.0539, 0.0315, 0.0513, 0.055, 0.0639, 0.086, 0.089, 0.0677, 0.0594, 0.0755, 0.084, 0.1842]
    return CM
end

function negative_binomials()
    ##Age group's mean
    AgeMean = Vector{Float64}(undef, 15)
    AgeSD = Vector{Float64}(undef, 15)

    AgeMean = [10.21, 14.81, 18.22, 17.58, 13.57, 13.57, 14.14, 14.14, 13.83, 13.83, 12.3, 12.3, 9.21, 9.21, 6.89]
    AgeSD = [7.65, 10.09, 12.27, 12.03, 10.6, 10.6, 10.15, 10.15, 10.86, 10.86, 10.23, 10.23, 7.96, 7.96, 5.83]

    nbinoms = Vector{NegativeBinomial{Float64}}(undef, 15)
    for i = 1:15
        p = 1 - (AgeSD[i]^2-AgeMean[i])/(AgeSD[i]^2)
        r = AgeMean[i]^2/(AgeSD[i]^2-AgeMean[i])
        nbinoms[i] =  NegativeBinomial(r, p)
    end
    return nbinoms    
end

function sample_contacts(bval, creduction = 0)
    # defaul beta value => 0.024, R0 = 2.35
    CM = contact_matrix()  # don't really need contacts.
    NB = negative_binomials()

    # sample 1000 serial intervals 
    # each serial interval is a proxy for the infectious period of an individual
    n_mc = 1000

    si = Int.(round.(sample_serial_interval(n_mc)))
    #β = 0.01 
    β = bval

    ## sample the age group in each serial interval 
    ig = rand(1:15, n_mc)

    # sample the number of total contacts in each serial interval 
    cnt_meet = sum.(rand.(NB[ig], si))

    # reduce the number of contacts 
    cnt_meet = Int.(round.((1 - creduction) .* cnt_meet))

    cnt_inf = map(cnt_meet) do x
        cnt_inf = 0
        for m in 1:x
            if rand() < β
                cnt_inf += 1
            end
        end
        return cnt_inf
    end

    res = zeros(Int64, n_mc, 3)
    res[:, 1] = si
    res[:, 2] = cnt_meet
    res[:, 3] = cnt_inf
    return res
end

function beta_table(βs)    
    res = zeros(Float64, length(βs), 2)
    res[:, 1] = βs
    res[:, 2] = map(x -> mean(sample_contacts(x, 0)[:, 3]), βs)
    return res
end

function min_reduction(β)
    ## go over these reduction values
    lvals = 0.0:0.01:0.9 
    mm = Float64[]
    for δ in lvals 
        m = mean(sample_contacts(β, δ)[:, 3]) # get the mean 
        push!(mm, m)
    end
    minred = findfirst(x -> x <= 1, mm)
    println("minimum reduction needed: $(lvals[minred])")
    ## push the minimum value needed to the top of mm 
    ## so that we don't have to return a tuple
    pushfirst!(mm, lvals[minred])
    
    # return mm, lvals[minred]
end

function min_reduction_overbetas()
    βs = 0.021:0.001:0.05
    btable = beta_table(βs)
    mm = map(x -> min_reduction(x), βs) 
    mm = transpose(foldl(hcat, mm))
    #convert to a dataframe
    # where rows are betas and columns are reduction values (but don't bother with column  names)
    # remember the first element of `mm` is the reduction parameter, which after the transpose gets its own column
    df = DataFrame(mm)
    # rename the column for reduction 
    rename!(df, :x1 => :reduction)

    insertcols!(df, 1, :beta => btable[:, 1])
    insertcols!(df, 2, :r0 => btable[:, 2])
    return df
end


