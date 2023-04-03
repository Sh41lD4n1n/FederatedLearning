# FederatedLearning

![](demo_image/gad-fed-learning-2-0.png)

In this project we have applied Federated learning with avaraging to train machine learning model on some number of workers. We neede to optimize equation:

$
    \min_{x \in \R^d} { f(x) = \frac{1}{M} \sum_{m=1}^{M} f_m (x)},
$

Then we tried some optimization methods(preconditioners: SGD,Adam, OASIS). Final results are on folder (Reports/Tuning_prec)



## Problem
- Task: Binary classification
- Model: Logistic Regression

    $
                \begin{cases}
                z = wX + b \\
                y = \frac{1}{1 + \exp^{-z}} \\
                \end{cases}
    $

- Optimization function

    $W = W - lr * (D_k)^{-1} * grad$

- Loss: Cross Entropy

    $
        loss = - \sum_{i \in Data} y_{i,true} log(y_{i,pred}) + (1 - y_{i,true} ) log(1-y_{i,pred})
    $

- Other metrics: Accuracy

    $
        accuracy = \frac{\# elements\ predicted\ correctly}{\# elements}
    $

## Results
Finaly we tuned both optimizers and draw final results.

### Heterogenious split

![accuracy](demo_image/het_acc.png)
![loss](demo_image/het_loss.png)
### Identical split
![accuracy](demo_image/Ident_loss.png)
![loss](demo_image/Ident_acc.png)