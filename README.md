## CO 553 Neural Network Coursework

### Part1:

We finished all of the part1 and it has successfully passed on LabTS:)

### Part2:

We implement our claimclassifier by using Keras Neural Network Library. We initialze our model by adding 3 layer and initial parameters: 

* `keras.models.Dense` 
* `epochs=30`
* `batch_size=32,`
* `optimizer = 'SGD'`

We also implement functions as follows:

* `evaluate_architecture(self,x_val,y_val,predicted_y))` will return the auc score from the trained model and plot the roc curve.


* `plot_roc(y_val,predicted_y)` is implemented to plot the roc curve and print out auc in evaluation the network.
* `ClaimClassifierHyperParameterSearch(x_train,y_train, estimator, param_grid)` returns the best parameter of the model by `GridSearch` method.

##### Model is saved in .5h file and test passed on the local repository provided by the instructor.

### Part3:

We preprocessed our data use `LabelEncoder()` from sklearn and draw a feature correlation heatmap in order to select the features.   

In `predict_premium(self, X_raw)` , we add a `factor` to scale the premium. This factor depends on the how much more expensive `vh_value` of the contract is compared to the median `vh_value`.

Other parts are the save as part2