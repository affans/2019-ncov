
## board 
r = 1:8
board = isodd.(r .+ r')

## first example
## create a empty heatmap "scene"
eh = heatmap(fill(0, 100, 100), scale_plot = true, show_axis = false, resolution = (500, 500))
typeof(eh) 
hm = last(eh) ## get the attributes 
N = 50
## record inside this scene
record(eh, "output.mp4", 1:N; framerate = 24) do i
    hp = reshape(rand(1:100, 10000), (100,100)) ## create random 100x100 matrix
    hm[1] = hp  ## assign matrix to the hm[1] plot property.
    yield()
end


## for my purposes 
function t(mats) 
    N = 250
    mscene = Scene(resolution = (1000,1000), scale_plot=true)
        
    #mymaparr = [colorant"green", colorant"red", colorant"blue", colorant"black", 
    #colorant"red", colorant"lightsalmon",  colorant"magenta", colorant"purple", colorant"yellow", colorant"black"]
    mymaparr = [colorant"white", colorant"red", colorant"green"]
    record(mscene, "output.mp4", 1:N, framerate=24) do i
        heatmap!(mscene, mats[:, :, i], scale_plot = true, show_axis = false, colormap=mymaparr, colorrange=(0, 9) )
        update!(mscene)
    end
    #show(mscene)   
end


heatmap!(scene, mats[:, :, 1], scale_plot = true, show_axis = false, colormap=mymaparr, colorrange=(0, 9))

mymap = Dict(0 => colorant"green", 1 => colorant"red", 2 => colorant"blue", 3 => colorant"azure", 
4 => colorant"red", 5 => colorant"lightsalmon", 6 => colorant"magenta", 7 => colorant"magenta", 8 => colorant"grey60", 9=> colorant"black")

for c in unique(mats[:, :, 1])
    println(mymaparr[c])
end

x = heatmap(mats[:, :, 1], scale_plot = true, show_axis = false, 
    colormap=[colorant"green", colorant"red", colorant"blue", colorant"azure", 
    colorant"red", colorant"lightsalmon", colorant"magenta", colorant"magenta", colorant"grey60", colorant"black"])
    # #push!(heatmaps, x)

    
#scene = Scene(resolution = (500, 500))

heatmaps = []
for i = 1:500
    x = heatmap(mats[:, :, i], scale_plot = true, show_axis = false, 
    colormap=[colorant"green", colorant"teal", colorant"blue", colorant"azure", 
    colorant"red", colorant"lightsalmon", colorant"magenta", colorant"magenta", colorant"grey60", colorant"black"])
    push!(heatmaps, x)
end

