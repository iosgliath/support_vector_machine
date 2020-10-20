using CSV, DataFrames, LinearAlgebra, Distributions, Random, StatsBase, Plots, Combinatorics

mutable struct SVM
    x::Matrix
    y::Vector
    c::Float64
    tol::Float64
    max_iter::Integer
    max_passes::Integer
    kernel::String
    degree::Integer
    γ::Float64
    α::Vector
    b::Float64
    sv_pos::Vector
    k::Matrix
end

mutable struct BinaryModel
    x_train::Array{Float64,2}
    y_train::Vector{Int64}
    x_test::Array{Float64,2}
    y_test::Vector{Int64}
    classpos::Int64
    classneg::Int64
    β::SVM
    accuracy::Float64
end

function svm(
             x::Matrix,
             y::Vector;
             c::Float64 = 0.6,
             kernel::String = "rbf",
             max_iter::Integer = 1000,
             max_passes::Integer = 1000,
             tol::Float64 = 1e-5,
             degree::Integer = 4,
             γ::Float64 = 0.1,
             #α::Vector,
             b::Float64 = 0.0
             )

    n = size(x, 1)
    α = zeros(n)
    k = zeros(n, n)
    sv_pos = collect(1:n)

    return SVM(x, y, c, tol, max_iter, max_passes, kernel, degree, γ, α, b, sv_pos, k)
end

function splitTestTrain(x::Array{Float64, 2}, y::Vector{Int64}, α::Float64)
    bernoulli(p) = rand() < p
    train = Any[]; test = Any[];
    for i = 1:size(x, 1)
        if bernoulli(α) == true
            append!(train,i)
        else
            append!(test,i)
        end
    end
    # code not optimized
    # + care, train could be 0 lenght!
    xtrain = convert(Array, x[train,:])
    ytrain = vec(convert(Array, y[train,:]))
    xtest = convert(Array, x[test,:])
    ytest = vec(convert(Array, y[test,:]))
    return xtrain, ytrain, xtest, ytest
end

function fit(
            x_train::Array{Float64, 2},
            x_test::Array{Float64, 2},
            y_train::Vector{Int64},
            y_test::Vector{Int64},
            max_i::Int64,
            max_p::Int64,
            k::String,
            c::Float64,
            γ::Float64
            )

    β = svm(x_train, y_train, max_iter = max_i, max_passes = max_p, kernel = k, c = c, γ=γ)
    smo!(β)
    predictions = predict(x_test, β)
    #print("\nModel computeAccuracy : $(computeAccuracy(y_test, predictions))")
    return β, predictions
end

function smo!(β::SVM)
    """
    http://cs229.stanford.edu/materials/smo.pdf
    If your not an expert in lagrangian arithmetics (I'm not),
    this is a great ressource to get started with SMO
    """
    m = size(β.x, 1)
    for i = 1:m
        # does rbf betzeen all rows of features into columns of K
        β.k[:,i] = kernel(β.x, β.x[i,:], β)
    end

    passes = 0
    while passes < β.max_passes

        Δα = 0

        for i = 1:m

            Ei = computeError(i, β)

            if (β.y[i] * Ei < -β.tol && β.α[i] < β.c) || (β.y[i] * Ei > β.tol && β.α[i] > 0)

                j = rand(1:m)
                if j == i
                    j = rand(1:m)
                end

                L, H = computeBounds(i, j, β)
                L == H && continue

                η = 2.0 * β.k[i, j] - β.k[i, i] - β.k[j, j]
                η >= 0 && continue

                Ej = computeError(j, β)

                α_io, α_jo = β.α[i], β.α[j]

                β.α[j] -= (β.y[j] * (Ei - Ej)) / η
                β.α[j] = clamp(β.α[j], L, H)

                abs(β.α[j] - α_jo) < β.tol && continue

                β.α[i] = β.α[i] + β.y[i] * β.y[j] * (α_jo - β.α[j])

                b1 = β.b - Ei - β.y[i] * (β.α[i] - α_jo) * β.k[i, i] -
                     β.y[j] * (β.α[j] - α_jo) * β.k[i, j]
                b2 = β.b - Ej - β.y[j] * (β.α[j] - α_jo) * β.k[j, j] -
                     β.y[i] * (β.α[i] - α_io) * β.k[i, j]

                if 0 < β.α[i] < β.c
                    β.b = b1
                elseif 0 < β.α[j] < β.c
                    β.b = b2
                else
                    β.b = 0.5 * (b1 + b2)
                end

                Δα += 1

            end

            if Δα == 0
                passes += 1
            else
                passes = 0
            end

        end

        β.sv_pos = findall(β.α .> 0)

    end
end

function kernel(a::Matrix, v::Vector, β::SVM)
    if β.kernel == "rbf"
        n = size(a, 1)
        k = zeros(n)
        for i = 1:n
            # γ = 1/2σ^2
            k[i] = ℯ^(-β.γ*sqL2dist(a[i,:],v))
        end
        return k
    end
end

function computeBounds(i::Int64, j::Int64, β::SVM)
    # optimise αi and αj
    if β.y[i] != β.y[j]
        L = max(0, β.α[j] - β.α[i])
        H = min(β.c, β.c - β.α[i] + β.α[j])
    else
        L = max(0, β.α[i] + β.α[j] - β.c)
        H = min(β.c, β.α[i] + β.α[j])
    end
    return L, H
end

function predict(x::Vector{Float64}, β::SVM)
    k = kernel(β.x, x, β)
    return sign(k' * (β.α .* β.y) + β.b)
end

function predict(x::Array{Float64, 2}, β::SVM)
    n = size(x, 1)
    k = zeros(Int64, n)
    if n == 1
        k[1] = predict(x, β)
    else
        for i = 1:n
            k[i] = predict(x[i,:], β)
        end
    end
    return k
end

function computeError(i::Int64, β::SVM)
    return predict(β.x[i,:], β) - β.y[i]
end

function computeAccuracy(predictions::Vector{Int64}, y_test::Vector{Int64})
    pos = 0
    n = length(y_test)
    for i = 1:n
        if predictions[i] == y_test[i]
            pos +=1
        end
    end
    return pos/n
end

function βbattleground(x::Array{Float64,2}, y::Vector{Int64}, splitα::Float64, mi::Int64, mp::Int64, k::String, c::Float64, γ::Float64)
    """
    implementation of the One vs One strategy in order to use SVM in multi classification
    """
    n = length(unique(y))
    labels = [i for i in unique(y)]
    binaries = combinations(1:n, 2)
    #nmodels = trunc(Int, n * (n-1) / 2) == length(binaryMaps)

    βs = Array{BinaryModel, 1}(undef, length(binaries))

    idx = 1
    for binary in binaries
        c1 = binary[1]
        c2 = binary[2]

        sidx = findall(x->x ∈ [c1, c2], y)

        xi = x[sidx,:]
        d = Dict([ labels[c1] => 1, labels[c2] => -1])
        yi = getindex.(Ref(d), y[sidx])

        βs[idx] = binaryβ(xi, yi, c1, c2, splitα, mi, mp, k, c, γ)

        idx += 1
    end

    return βs, labels
end

function binaryβ(x::Array{Float64,2}, y::Vector{Int64}, classpos::Int64, classneg::Int64, splitα::Float64, mi::Int64, mp::Int64, k::String, c::Float64, γ::Float64)
    x_train, y_train, x_test, y_test = splitTestTrain(x, y, splitα)
    β, preds = fit(x_train, x_test, y_train, y_test, mi, mp, k, c, γ)
    acc = computeAccuracy(preds, y_test)
    model = BinaryModel(x_train, y_train, x_test, y_test, classpos, classneg, β, acc)
    return model
end

function countmemb(itr)
    #https://stackoverflow.com/questions/39101839/the-number-of-occurences-of-elements-in-a-vector-julia
    d = Dict{eltype(itr), Int}()
    for val in itr
        if isa(val, Number) && isnan(val)
            continue
        end
        d[val] = get!(d, val, 0) + 1
    end
    return d
end

function kaloskagathing(βs::Array{BinaryModel, 1}, x_test::Array{Float64, 2}, classIdx::Array{Int64, 1})

    """
    'Kalos kagathos or kalokagathos (Ancient Greek: καλὸς κἀγαθός [kalòs kaːɡatʰós]),
    of which kalokagathia (καλοκαγαθία) is the derived noun, is a phrase used
    by classical Greek writers to describe an ideal of gentlemanly personal conduct,
    especially in a military context.'
    https://en.wikipedia.org/wiki/Kalos_kagathos
    """
    kalokagathos = Array{Int64, 2}(undef, size(x_test, 1), length(βs))

    for i in 1:length(βs)
        p = predict(x_test, βs[i].β)

        replace!(p, 1 => βs[i].classpos)
        replace!(p, -1 => βs[i].classneg)
        kalokagathos[:,i] = p
    end

    kalokagathos = mapslices(x->countmemb(x), kalokagathos, dims=2)

    kalokagathos = findmax.(kalokagathos)

    kalokagathos = [(kalokagathos...)...]

    i = 2:2:length(kalokagathos)

    return kalokagathos[i]
end


function gridSearch(x::Array{Float64,2}, y::Vector{Int64}, splitα::Float64, mi::Int64, mp::Int64, k::String)

    crange = [ 2*10^-4, 2*10^-2, 2*10^-1, 2^-1, 2, 10, 100, 2*10^2, 2*10^2]
    γrange = [ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

    βs = Array{BinaryModel, 1}(undef, length(crange)*length(γrange))

    x_train, y_train, x_test, y_test = splitTestTrain(x, y, splitα)

    idx = 1
    for ci in crange, γi in γrange
        β, predictions = fit(x_train, x_test, y_train, y_test, mi, mp, k, ci, γi)
        acc = computeAccuracy(predictions, y_test)
        print("\nC = ", ci, " // γ = ", γi, " // acc = ", acc)
        βs[idx] = BinaryModel(x_train, y_train, x_test, y_test, 1, -1, β, acc)
        idx += 1
    end

    return βs
end

function normalise(v::Vector{Float64})
    l = minimum(v); h = maximum(v)
    δ = h-l
    w(x) = (x-l) / δ
    v = w.(v)
end

function normalise(a::DataFrame)
    for row in eachrow(a)
      row .= normalise(vec(convert(Array, row)))
    end
    return a
end

##########################################

function irisPreprocess(mapping::DataFrame)
    dict = Dict([ "setosa" => 1, "versicolor" => 2, "virginica" => 3 ])
    f = getindex.(Ref(dict), mapping.Species)
    # remove useless col: first and last + diagnosis
    select!(mapping, Not([:Species]))
    #mapping = normalise(mapping)
    insert!(mapping, 1, f, :Class)
    return mapping
end

using RDatasets
# load iris dataset
iris = dataset("datasets", "iris")

# geat features and labels
mapping = irisPreprocess(iris)
y = vec(convert(Array, mapping[:,1]))
x = convert(Array, mapping[:,2:5])

# split a first time in training and testing with split value 0.7 (keep 30% for testing)
x_train, y_train, x_test, y_test = splitTestTrain(x, y, 0.7)

# feed training into One Vs One process.
# each binary classifier will get samples from xtrain to pick from with split value 0.7
models, labels = βbattleground(x_train, y_train, 0.7, 1000, 1000, "rbf", 10.0, 0.01)
# print our labels
print(labels)

# now let's test our model against the initial test set
predictions = kaloskagathing(models, x_test, labels)
accu = computeAccuracy(predictions, y_test)

##################################


"""
Now let's use some dataset from kaggle
https://www.kaggle.com/ronitf/heart-disease-uci
"""
pwd()
cd()
cd("your file path")
kaggle_file = "heart.csv"


function loadNpreprocess()
    data = CSV.File(kaggle_file) |> DataFrame
    x = convert(Array{Float64,2}, data[:,1:13])
    y = vec(data[:,14])
    replace!(y, 0 => -1)
    return x, y
end

# get features and labels
xx, yy = loadNpreprocess()

# try the classifier with and without normalisation and see results
#
# xx[:,1] = normalise(xx[:,1])
# xx[:,4] = normalise(xx[:,4])
# xx[:,5] = normalise(xx[:,5])
# xx[:,8] = normalise(xx[:,8])
#

# feed features and labels into binary classifier
binaryModel = binaryβ(xx, yy, 1, -1, 0.5, 1000, 1000, "rbf", 0.6, 0.1)
print(binaryModel.accuracy)

# now let's try and find better params with grid search
models = gridSearch(xx, yy, 0.5, 1000, 1000, "rbf")


########################

"""

This part is more visual, using Distirbutions.jl to generate points
in 2 and 3 dimensional spaces and testing the classifier on them

"""
using Distributions


function generateRandomDataN(points::Int64)
    
    """
    I tried around messing with several point distributions, hence this corky setup
    """
    d1x = Normal(2, 1.5)
    d1y = Normal(2, 1)
    d2x = Normal(8, 1)
    d2y = Normal(3, 1.5)
    d3x = Normal(4, 1)
    d3y = Normal(6, 1)
    d4x = Normal(9, 1)
    d4y = Normal(9, 1.5)
    c1x = rand(d1x, points)
    c1y = rand(d1y, points)
    c2x = rand(d2x, points)
    c2y = rand(d2y, points)
    c3x = rand(d3x, points)
    c3y = rand(d3y, points)
    c4x = rand(d4x, points)
    c4y = rand(d4y, points)
    return c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y
end

function plotModel2D(c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, x_test, predictions, y_test)
    scatter(c1x, c1y, markercolor="grey")
    scatter!(c2x, c2y, markercolor="orange")
    scatter!(c3x, c3y, markercolor="blue")
    scatter!(c4x, c4y, markercolor="brown")

    error = predictions .== y_test

    data = hcat(x_test, error)

    scatter(
        data[:,1],
        data[:,2],
        group = data[:,3],
        title = "Testing Errors",
        xlabel = "x",
        ylabel = "y",
        #m = (0.5, [:cross :hex :star7], 12),
        bg = RGB(0.2, 0.2, 0.2)
    )
    scatter!(vcat(c1x, c2x, c3x, c4x), vcat(c1y, c2y, c3y, c4y), markercolor="white", markersize=0.1)
end

c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y = generateRandomDataN(1000)

x = hcat( vcat(c1x, c2x, c3x, c4x), vcat(c1y, c2y, c3y, c4y) )
y = vcat(ones(Int64, 1000)*1, ones(Int64, 1000)*2, ones(Int64, 1000)*3, ones(Int64, 1000)*4)

x_train, y_train, x_test, y_test = splitTestTrain(x, y, 0.5)

models, labels = βbattleground(x_train, y_train, 0.9, 1000, 1000, "rbf", 0.6, 0.001)
predictions = kaloskagathing(models, x_test, labels)
acc = computeAccuracy(predictions, y_test)

plotModel2D(c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y, x_test, predictions, y_test)


###########################


# 3D data points

function generateRandomDataN3D(points::Int64)

"""
I tried around messing with several point distributions, hence this corky setup
"""
    d1x = Normal(2, 1)
    d1y = Normal(2, 1)
    d1z = Normal(2, 1)
    d2x = Normal(8, 1)
    d2y = Normal(3, 1)
    d2z = Normal(7, 1)
    d3x = Normal(4, 1)
    d3y = Normal(6, 1)
    d3z = Normal(5, 1)
    d4x = Normal(9, 1)
    d4y = Normal(9, 1)
    d4z = Normal(8, 1)
    c1x = rand(d1x, points)
    c1y = rand(d1y, points)
    c1z = rand(d1z, points)
    c2x = rand(d2x, points)
    c2y = rand(d2y, points)
    c2z = rand(d2z, points)
    c3x = rand(d3x, points)
    c3y = rand(d3y, points)
    c3z = rand(d3z, points)
    c4x = rand(d4x, points)
    c4y = rand(d4y, points)
    c4z = rand(d4z, points)


    return c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z, c4x, c4y, c4z
end

c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z, c4x, c4y, c4z = generateRandomDataN3D(1000)

x = hcat( vcat(c1x, c2x, c3x, c4x), vcat(c1y, c2y, c3y, c4y), vcat(c1z, c2z, c3z, c4z) )
y = vcat(ones(Int64, 1000)*1, ones(Int64, 1000)*2, ones(Int64, 1000)*3, ones(Int64, 1000)*4)

x_train, y_train, x_test, y_test = splitTestTrain(x, y, 0.8)

#nclass = length(unique(y))
models, labels = βbattleground(x_train, y_train, 0.4, 1000, 1000, "rbf", 0.6, 0.001)
predictions = kaloskagathing(models, x_test, labels)
acc = computeAccuracy(predictions, y_test)


function plotModel3D(c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z, c4x, c4y, c4z, x_test, predictions, y_test)
    error = predictions .== y_test

    data = hcat(x_test, error)

    scatter(
        data[:,1],
        data[:,2],
        data[:,3],
        group = data[:,4],
        title = "Testing Errors",
        xlabel = "x",
        ylabel = "y",
        zlabel = "z",
        #m = (0.5, [:cross :hex :star7], 12),
        bg = RGB(0.2, 0.2, 0.2)
    )
    scatter!(vcat(c1x, c2x, c3x, c4x), vcat(c1y, c2y, c3y, c4y), vcat(c1z, c2z, c3z, c4z), markercolor="white", markersize=0.1)
end
plotModel3D(c1x, c1y, c1z, c2x, c2y, c2z, c3x, c3y, c3z, c4x, c4y, c4z, x_test, predictions, y_test)
