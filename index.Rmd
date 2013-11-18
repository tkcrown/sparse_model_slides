---
title       : Sparse Methods
subtitle    : 
author      : Ping Jin
job         : 
framework   : io2012       # {io2012, html5slides, shower, dzslides, ...}
highlighter : highlight.js  # {highlight.js, prettify, highlight}
hitheme     : tomorrow      # 
widgets     : [mathjax]            # {mathjax, quiz, bootstrap}
mode        : selfcontained # {standalone, draft}

--- 

## Outline

- <h3>Introduction to Dimension Reduction</h3>

- <h3>Linear Regression and Least Squares (Review)</h3>

- <h3>Subset Selection</h3>

- <h3>Shrinkage Method</h3>

- <h3>Beyond Lasso</h3>

--- 

## Part 1: Introduction to Dimension Reduction

- <b>Introduction to Dimension Reduction</b>
    - <b>Difference between feature selection and feature extraction</b>
    - <b>Feature Selection</b>
        - <b>Wrapper method</b>
        - <b>Filter method</b>
        - <b>Embedded method</b>
    - <b>Feature Extraction</b>
        - <b>PCA, ICA...</b>
- Linear Regression and Least Squares (Review)
- Subset Selection
- Shrinkage Method
- Beyond Lasso

---

## Difference between Feature Selection and Feature Extraction

### Feature Selection

- chooses a subset of features from the original feature set

### Feature Extraction

- transforms the original features into new ones
- e.g. projects data from high dimensions to low dimensions

### Question: 

- Why do we need two frameworks for dimension reduction?


---

## Difference between Feature Selection and Feature Extraction

### Example 1: Prostate Cancer

The data come from a study by Stamey et al.(1989). In this task, we are trying to identify a subset of features that are useful for prediction of the level of prostate-specific antigen (lpsa). Our available feature set is 
$$\{lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45\}.$$
In this case, we would like to have our result as a subset of the whole set, such as
$$\{lcavol, lweight, age, svi, lcp\},$$
which are important for prediction of lpsa.

Feature selection applies.

---

## Difference between Feature Selection and Feature Extraction

### Example 2: Embedded System with limited computation resource

Suppose we have a robot which possess many sensors, such as temperature sensor, light senor, cliff sensor, gas sensor, GPS, gyroscope and so on. And our task is to discriminate if the current situation is dangerous for the robot, according the data from these sensors. In this task, our computation resource is very limited due to 
- the CPU of the embedded system
- our battery capacity. 

Hence, we would prefer that our prediction task involves computation as little as possible. Then a compact model with less features from sensors would be preferred. 

Feature selection applies.


---

## Difference between Feature Selection and Feature Extraction

### Example 3: classification with fMRI data
fMRI data are 4D images, with one dimension being the time slot.

<center>![fMRI](assets/img/fMRI.png "fmri")</center>



---

## Difference between Feature Selection and Feature Extraction

### Example 3: classification with fMRI data

- fMRI data are 4D images, with one dimension being the time slot. 

- Suppose the dimension of images is $50 \times 50 \times 50$ for single time point and we have 200 time points

- $50 \times 50 \times 50 \times 200 = 25,000,000$ dimensions in total! This will cause great computation burdun

In this task, we are not concerned about importance of particular voxels. Our purpose is to decrease the number of dimensions without losing too much information for further prediction task. 

Feature extraction applies better.



---

## Feature Selection

### Wrapper Methods

- search the space of feature subsets
- use the training/validation accuracy of a particular classifier as the measure of utility for a candidate subset.

### Embedded Methods
- exploit the structure of speciﬁc classes of learning models to guide the feature selection process, 
- e.g. LASSO[1]. It is embedded as part of the model construction process[2].

### Filter Methods

- use some general rules/criterions to measure the feature selection results, independent of the classifiers, 
- e.g. mutual information based method[1].

---

## Feature Selection

### Comparison

|               | Wrapper       | Filter| Embedded|
| ----------- |:-------------:| -----:|-----:|
| Speed               | Low     | High  | Mid|
| Chance of Overfitting| High   | Low   | Mid|
| Classifier-Independent | No   | Yes   | No  |

--- &twocolportion w1:40% w2:60%

## Feature Extraction   

*** left

- <b>A graphical explanation</b>
    - Each data sample has two features
    - Prefer the direction with larger variance
    - Original features are transformed into new ones

*** right

![alt text](assets/img/pca.png "Principle component analysis")


---

## Part 2: Linear Regression and Least Squares (Review)

- Introduction to Dimension Reduction
- <b>Linear Regression and Least Squares (Review)</b>
    - <b>Least Square Fit</b>
    - <b>Gauss Markov</b>
    - <b>Bias-Variance tradeoff</b>
    - <b>Problems</b>
- Subset Selection
- Shrinkage Method
- Beyond Lasso

--- &twocolportion w1:58% w2:38%

## Linear Regression and Least Squares (Review)

*** left

### Least Squares Fit

$$
\begin{equation}
\begin{split}
RSS(\beta) &= (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta)\\
\frac{\partial RSS}{\partial \beta} &= -2 \mathbf{X}^T(\mathbf{y} - \mathbf{X}\beta)\\
\hat{\beta} &= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\end{split}
\end{equation}
$$

### Gauss Markov Theorem

The least squares estimates of the parameters β have the smallest variance among all linear unbiased estimates.

### Question

Is unbiased assumption necessary?

*** right

![Linear regression](assets/img/lr.png "Linear regression")

![Least Squares](assets/img/ls.png "Least squares")

---

## Part 3: Subset Selection

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- <b>Subset Selection</b>
  - <b>Best-subset selection</b>
  - <b>Forward stepwise selection</b>
  - <b>Forward stagewise selection</b>
  - <b>Problems</b>
- Shrinkage Method
- Beyond Lasso

---

## Linear Regression and Least Squares (Review)

### Bias-Variance tradeoff
$$
\begin{equation}
\begin{split}
MSE(\tilde{\theta} &= E[(\tilde{\theta} - \theta)^2])\\
&= Var(\tilde{\theta} + [E[\tilde{\theta}] - \theta])
\end{split}
\end{equation}
$$

where $\theta = \alpha^T \beta$. We can trade some bias for much less variance.

### Problems of Least Squares

- Prediction accuracy: low bias, but high variance, overfitting noise and sensitive to outlier
- Interpretation: Sometimes, especially when faced with numerous features, we may want a "big picture" of the model
- $(\mathbf{X}^T\mathbf{X})$ may be invertible and thus no closed form solution

---

## Subset Selection

### Best-subset selection


- Best subset regression finds for each $k \in \{0, 1, 2, . . . , p\}$ the subset of size $k$ that gives smallest residual sum of squares.
- An efficient algorithm, the leaps and bounds procedure (Furnival and Wilson, 1974), makes this feasible for p as large as 30 or 40.[]

---

## Subset Selection

### Forward-stepwise selection

Instead of searching all possible subsets, we can seek a good path through them.

*Forward-Stepwise Selection* builds a model sequentially, adding one variable at a time. At each step, it

- identifies the best variable (with the highest correlation with the residual error) to include in the *active set*
$$\mathbf{x_k} = argmax_{\mathbf{x}_j}(|\mathbf{x}^T_j \mathbf{r}|)$$
- then updates the least squares fit to include all the active variables

--- &twocolportion w1:45% w2:50%

## Subset Selection

### Forward-Stagewise Regression

*** left
- Initialize the fit vector $\mathbf{f} = 0$
- Compute the correlation vector 
$$\mathbf{c} = \mathbf{c}(\mathbf{f}) = \mathbf{X}^T(\mathbf{y} - \mathbf{f})$$
- $k = argmax_{j \in \{1,2,..,p\}} |\mathbf{c}_j|$
- Coefficients and fit vector are updated
$$\mathbf{f} \gets \mathbf{f} + \alpha \cdot sign(\mathbf{c}_j) \mathbf{x}_j$$
$$\beta_j \gets \beta_j + \alpha \cdot sign(\mathbf{c}_j)$$ 


***right

![Stagewise](assets/img/stagewise.png "Stagewise")


---&twocolportion w1:40% w2:55%

## Subset Selection

### Comparison
*** left
- It takes at most $p$ steps for forward-stepwise selection to get the final fit
- Forward stagewise selection is a slow fitting algorithm, at each time step we only update one $\beta_j$, which can take more than $p$ steps
- Forward stagewise is useful in high dimensional problem

*** right

<center>![comp1](assets/img/comp1.png "comp")</center>

---

## Subset Selection

### Pros

- More interpretable and compact model

### Cons

- It is a discrete process, and thus has high variance and sensitivity to the change in dataset.
- Thus may not be able to lower prediction error

---


## Ridge Regression

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- <b>Shrinkage Method</b>
    - <b>Ridge Regression</b>
        - <b>Formulations and closed form solution</b>
        - <b>Singular value decomposition</b>
        - <b>Degree of Freedom</b>
    - Lasso
- Beyond Lasso

---

## Ridge Regression

- <b>Linear regression with $l2$-regularization</b>
    - Least squares with quadratic constraints
$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2, s.t. \sum_{j = 1}^p \beta_j^2 \leq t
\end{equation}
$$
    - Its dual form
$$
\hat{\beta}^{ridge} = argmin_{\beta}\{\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2 + \lambda \sum_{j = 1}^p\beta_j^2\}
$$
    - The L2-regularization can be viewed as a Gaussian prior on the coefficients, and our estimates are the posterior means
- <b>Solution</b>
$$
\begin{equation}
\begin{split}
&RSS(\lambda) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta) + \lambda \beta^T\beta\\
&\hat{\beta}^{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
\end{split}
\end{equation}
$$    


---

## Ridge Regression

### Singular Value Decomposition (SVD)
- The SVD of $\mathbf{X}$:
$$\mathbf{X} = \mathbf{UDV}^T$$
    - $\mathbf{U}$: $N \times p$ <b>orthogonal</b> matrix with columns spanning the column space of $\mathbf{X}$
    - $\mathbf{V}$: $p \times p$ <b>orthogonal</b> matrix with columns spanning the row space of $\mathbf{X}$  
    - $\mathbf{D}$: $p \times p$ <b>diagonal</b> matrix with diagonal entries $d_1 \geq d_2 \geq ... \geq d_p \geq 0$ being the singular values of $\mathbf{X}$
- For least squares
$$
\begin{equation}
\begin{split}
\mathbf{X}\hat{\beta}^{ls} &= \mathbf{X(X^TX)^{-1}X^Ty}\\
&=\mathbf{UU^Ty}
\end{split}
\end{equation}
$$

---

## Ridge Regression

### Singular Value Decomposition (SVD)
- For ridge regression
$$
\begin{equation}
\begin{split}
\mathbf{X}\hat{\beta}^{ridge} &= \mathbf{X(X^TX + \lambda I)^{-1}X^Ty}\\
&=\sum_{j=1}^p\mathbf{u}_j\frac{d_j^2}{d_j^2 + \lambda} \mathbf{u}_j^T\mathbf{y}
\end{split}
\end{equation}
$$
    - Compared with the solution of least square, we have an additional shrinkage term, the smaller d is and the larger λ is, the more shrinkage we have. 
- The SVD of the centered matrix $X$ is another way of expressing the principal components of the variables in $X$. 


---

## Ridge Regression

### Singular Value Decomposition (SVD) 

<center>![PC](assets/img/pc.png "pc")</center>

--- &twocolportion w1:40% w2:55%

## Ridge Regression

### Degree of Freedom

*** left
- In statistics, the number of degrees of freedom is the number of values in the final calculation of a statistic that are free to vary.
- Computation
$$
\begin{equation}
\begin{split}
d(\lambda) &= tr[\mathbf{X(X^TX + \lambda I)^{-1}X^T}]\\
&=tr[\mathbf{H_{\lambda}}]\\
&=\sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} 
\end{split}
\end{equation}
$$
- The smaller $d$ is and the larger $\lambda$ is, the less degree of freedom we have

*** right

<center>![df](assets/img/df.png "df")</center>


---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - <b>Formulations</b>
        - <b>Comparisons with ridge regression and subset selection</b>
        - <b>Quadratic Programming</b>
        - <b>Least Angle Regression</b>
        - <b>Viewed as approximation for $l0$-regularization</b>
- Beyond Lasso

---


## LASSO

### Linear regression with $l1$-regularization

- Problems with $l2$-regularization
    - Interpretability and compactness: Though coefficients are shrinked, but not to zero.
    
    - Least squares with constraints
$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2, s.t. \sum_{j = 1}^p |\beta_j| \leq t
\end{equation}
$$
    - Its dual form
$$
\hat{\beta}^{ridge} = argmin_{\beta}\{\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2 + \lambda \sum_{j = 1}^p|\beta_j|\}
$$
    - The $11$-regularization can be viewed as a Laplace prior on the coefficients

---

## LASSO

<center>![lasso](assets/img/lasso.png "lasso")</center>


---

## LASSO

### Comparison

- Orthonormal Input $\mathbf{X}$
    - <b>Best subset</b>: [Hard thresholding] Only keep the top $M$ largest coefficeints of $\hat{\beta}^{ls}$
    - <b>Ridge</b>: [Pure shrinkage] does proportionally shrinkage of $\hat{\beta}^{ls}$
    - <b>Lasso</b>: [Soft thresholding] translates each coefficient of $\hat{\beta}^{ls}$ by $\lambda$, truncating at 0 


<center>![comp2](assets/img/comp2.png "comp2")</center>

---

## LASSO

### Comparison

- Non-orthonormal Input $\mathbf{X}$

<center>![comp3](assets/img/comp3.png "comp3")</center>


---

## LASSO

### Other unit circles for different $p$-norms

<center>![uc](assets/img/unit_circle.png "uc")</center>

|   |Convex| Smooth| Sparse|
|----|----|----|----|
|$q<1$|No|No|Yes|
|$q>1$|Yes|Yes|No|
|$q = 1$|Yes|No|Yes|

Here $q = 0$ is the pure variable selection procedure, as it is counting the <b>number of non-zero coefficients</b>.

--- &twocolportion w1:48% w2:48%

## LASSO

### Regularizations as priors

$|\beta_j|^q$ can be viewed as the log-prior density for $\beta_j$, these three methods are bayes estimates with different priors

- <b>Subset selection</b>: corresponds to $q = 0$
- <b>LASSO</b>: corresponds to $q = 1$, Laplace prior, $density = (\frac{1}{\tau})exp(\frac{-|\beta|}{\tau}), \tau = 1/\lambda$
- <b>Ridge regression</b>: corresponds to $q = 2$, Gaussian Prior

*** left

<center>![laplace](assets/img/laplace.png "laplace")</center>

*** right

<center>![gauss](assets/img/gauss.png "gauss")</center>


--- &vcenter   

## Thank you

