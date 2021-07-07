# decision tree from scratch
A program in Python to implement the ID3 decision tree algorithm, written from scratch.

The program is run from the terminal. It takes a command-line parameter that contains the name of the file containing the training data, the maximum tree depth and the name of the file containing the training data (if any). For example:
```
# with no tree depth limit, the training set and test set are both titanic.txt

python decisiontreeassignment.py titanic.txt 
```
or
```
# with maximum tree depth be 3, the training set and test set are both titanic.txt

python decisiontreeassignment.py titanic.txt
```
or
```
# with maximum tree depth be 3, the training set and test set are separate

python decisiontreeassignment.py titanic_trainingset.txt 3 titanic_testset.txt
```
  
And it outputs the decision tree and the training set accuracy with the format like:
```
The decision tree is: {'sex': {'male': {'pclass': {'1st': {'age': {'adult': 'no', 'child': 'yes'}}, '2nd': {'age': {'adult': 'no', 'child': 'yes'}}, '3rd': {'age': {'adult': 'no', 'child': 'no'}}, 'crew': 'no'}}, 'female': {'pclass': {'1st': {'age': {'adult': 'yes', 'child': 'yes'}}, '2nd': {'age': {'adult': 'yes', 'child': 'yes'}}, '3rd': {'age': {'adult': 'no', 'child': 'no'}}, 'crew': 'yes'}}}}
The accuracy for the training set is 0.7905497501135847.
```
