using Distributions, StaticArrays


agedist = Distributions.Categorical(@SVector [0.190674655, 0.41015843, 0.224428346, 0.17473857]) 
const agebraks = @SVector [0:19, 20:49, 50:64, 65:99]


humans = zeros(Int64, 10000)


@inbounds for i = 1:length(humans) 
    humans[i] =  rand(agedist)
end

function contact()
    nbs = negative_binomials()
    cm = contact_matrix()
    whocontactwho = zeros(Int64, 4, 4)
    for ag in humans
        cnt = rand(nbs[ag])  ## get number of contacts/day
        gpw = Int.(round.(cm[ag]*cnt)) 
        #println("cnt: $cnt, gpw: $gpw")
        # let's stratify the human population in it's age groups. 
        # this could be optimized by putting it outside the contact_dynamic2 function and passed in as arguments               
        # enumerate over the 15 groups and randomly select contacts from each group
        for (i, g) in enumerate(gpw)
            # i : age group being met, i = 1 2 3 4
            # g:  totalcontacts
            whocontactwho[ag, i] += g
        end
    end
    return whocontactwho
end

function contact_matrix()
    CM = Array{Array{Float64, 1}, 1}(undef, 4)
    CM[1]=[0.5712, 0.3214, 0.0722, 0.0353]
    CM[2]=[0.1830, 0.6253, 0.1423, 0.0494]
    CM[3]=[0.1336, 0.4867, 0.2723, 0.1074]    
    CM[4]=[0.1290, 0.4071, 0.2193, 0.2446]
    return CM
end
function negative_binomials() 
    ## the means/sd here are calculated using _calc_avgag
    means = [15.30295, 13.7950, 11.2669, 8.0027]
    sd = [11.1901, 10.5045, 9.5935, 6.9638]
    nbinoms = Vector{NegativeBinomial{Float64}}(undef, 4)
    for i = 1:4
        p = 1 - (sd[i]^2-means[i])/(sd[i]^2)
        r = means[i]^2/(sd[i]^2-means[i])
        nbinoms[i] =  NegativeBinomial(r, p)
    end
    return nbinoms   
end
