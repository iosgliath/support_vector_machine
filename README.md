# support_vector_machine
Sequential minimal optimization for support vector machine. </br>
Support vector machine for binary and multi class (one vs one strategy) classification.</br>


If your not an expert in lagrangian arithmetics (I'm not), this is a great ressource to get started with SMO.</br>
 http://cs229.stanford.edu/materials/smo.pdf


The core of the process is happening in this function :</br>

```julia
function smo!(β::SVM)
```
