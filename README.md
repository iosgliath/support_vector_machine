# support_vector_machine
svm.jl</br>
Sequential minimal optimization for support vector machine. </br>
Support vector machine for binary and multi class (one vs one strategy) classification.</br>

Implementation in as pure Julia as the learning / work ratio would recommand it :)

If your not an expert in lagrangian arithmetics (I'm not), this is a great ressource to get started with SMO.</br>
 http://cs229.stanford.edu/materials/smo.pdf
 
 
<strong>How to use :</strong></br>

1. Binary classification
```julia
# xx are our features, yy our labels

binaryModel = binaryβ(xx, yy, 1, -1, 0.5, 1000, 1000, "rbf", 0.6, 0.1)
```
2. Multi class classification
```julia
iris = dataset("datasets", "iris")

mapping = multiClassPreprocess(iris) # some pre processing to data
y = vec(convert(Array, mapping[:,1])) # get labels
x = convert(Array, mapping[:,2:5]) # get features
x_train, y_train, x_test, y_test = splitTestTrain(x, y, 0.5) # split first time for fresh unseen data

models, labels = βbattleground(x_train, y_train, 0.9, 1000, 1000, "rbf", 0.6, 0.001) # feed xtrain into one vs one method
print(labels) # check our labels
predictions = kaloskagathing(models, x_test, labels) # check our prediction on xtest
accu = computeAccuracy(predictions, y_test)

```
3. Grid search
```julia
# xx are our features, yy our labels

binaryModel = binaryβ(xx, yy, 1, -1, 0.5, 1000, 1000, "rbf", 0.6, 0.1)
models = gridSearch(xx, yy, 0.5, 1000, 1000, "rbf")


```

