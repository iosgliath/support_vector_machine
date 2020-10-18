# support_vector_machine
Sequential minimal optimization for support vector machine. </br>
Support vector machine for binary and multi class (one vs one strategy) classification.</br>


If your not an expert in lagrangian arithmetics (I'm not), this is a great ressource to get started with SMO.</br>
 http://cs229.stanford.edu/materials/smo.pdf


The core of the process is happening in this function :</br>

```julia
function smo!(β::SVM)
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
```
