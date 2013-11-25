---
title       : Sparse Models
subtitle    : CMPUT 466/551
author      : Ping Jin (pjin1@ualberta.ca)
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

- <h3>Beyond LASSO</h3>

--- 

## Part 1: Introduction to Dimension Reduction

1. <b>Introduction to Dimension Reduction</b>
    - <b>General notations</b>
    - <b>Motivations</b>
    - <b>Feature selection and feature extraction</b>
    - <b>Feature Selection</b>
        - <b>Wrapper method</b>
        - <b>Filter method</b>
        - <b>Embedded method</b>
    - <b>Feature Extraction</b>
        - <b>PCA, ICA...</b>
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. Shrinkage Method
5. Beyond LASSO

---

## General Notations

### Dataset
- $\mathbf{X}$: columnwise centered $N \times p$ matrix
    - $N:$ # samples, $p:$ # features
    - An intercept vector $\mathbf{1}$ is added to $\mathbf{X}$, then $\mathbf{X}$ is $N \times (p+1)$ matrix
- $\mathbf{y}$: $N \times 1$ vector of labels(classification) or continous values(regression)

### Basic Model
- Linear Regression
  - Assumption: the regression function $E(Y|X)$ is linear
  $$f(X) = X\beta$$
  - $\beta$: $(p+1) \times 1$ vector of coefficients

---&twocolportion w1:53% w2:47% 

## Motivations

- Dimension reduction is about transforming data with high dimensionality into data of much lower dimensionality
    - <b>Computational efficiency</b>: less dimensions require less computations
    - <b>Accuracy</b>: lower risk of overfitting

*** left

- <b>Categories</b>
    - Feature Selection:  
        - chooses a subset of features from the original feature set
    - Feature Extraction:
        - transforms the original features into new ones, linearly or non-linearly
        - e.g. PCA, ICA, etc.

*** right

<br>

<center>![fs](assets/img/fs.png "fs")</center>


<center>![fe](assets/img/fe.png "fe")</center>


--- &twocolportion w1:50% w2:50%

## Feature Selection and Feature Extraction

*** left

### Feature Selection

- Easier to interpret
- Reduces cost: computation, budget, etc.

<br>
<br>
<br>
<center>![fs2](assets/img/fs2.png "fs2")</center>



*** right

### Feature Extraction

- More flexible. Feature selection is a spectial case of linear feature extraction

<br>
<br>
<br>


<center>![fe2](assets/img/fe2.png "fe2")</center>

---

## Feature Selection and Feature Extraction

### Example 1: Prostate Cancer

- <b>Response</b>: level of prostate-specific antigen (lpsa). 
- <b>Inputs</b>:
$$\{lcavol, lweight, age, lbph, svi, lcp, gleason, pgg45\}.$$
- <b>Task</b>:
    - predict $lpsa$ from measurements of features

Feature selection
- Cost: Measuring features cost money
- Interpretation: Doctors can see which features are important


---

## Feature Selection and Feature Extraction

### Example 2: classification with fMRI data

- fMRI data are 4D images, with one dimension being time. 

- Each image is ~ $50 \times 50 \times 50$(spatial) $\times 200$(times) $= 25M$ dimensions


Feature extraction 
- Individual voxel-times are not important 
- Cost is not correlated with #features
- Feature extraction offers more flexibility in transforming features, which potentially results in better accuracy

<center>![fMRI2](assets/img/fMRI2.png "fmri2")</center>


---

## Feature Selection Methods

### Wrapper Methods

- search the space of feature subsets
- use the training/validation accuracy of a particular classifier as the measure of utility for a candidate subset

<center>![wrapper](assets/img/wrapper.png "wrapper")</center>

---

## Feature Selection Methods

### Embedded Methods
- exploit the structure of speciﬁc classes of learning models to guide the feature selection process
- embedded as part of the model construction process
    - e.g. LASSO. 

<center>![embedded](assets/img/embedded.png "embedded")</center>

---

## Feature Selection Methods

### Filter Methods

- use some general rules/criterions to measure the feature selection results independent of the classifiers
- e.g. mutual information

<center>![filter](assets/img/filter.png "filter")</center>

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

### Principle Components Analysis

- <b>A graphical explanation</b>
    - Each data sample has three features
    - Often prefer the direction with larger variance
    - Original features are transformed into new ones
- <b>Example</b>
    - For fMRI images, we usually have millions of dimensions. PCA can project the data from millions of dimensions to only thousands of dimensions, or even less
- Other feature extraction methods: ICA, Kernel PCA , etc..

*** right

![alt text](assets/img/pca.png "Principle component analysis")


---

## Part 2: Linear Regression and Least Squares (Review)

1. Introduction to Dimension Reduction
2. <b>Linear Regression and Least Squares (Review)</b>
    - <b>Least Square Fit</b>
    - <b>Gauss Markov</b>
    - <b>Bias-Variance tradeoff</b>
    - <b>Problems</b>
3. Subset Selection
4. Shrinkage Method
5. Beyond LASSO

--- &twocolportion w1:58% w2:38%

## Linear Regression and Least Squares (Review)

*** left

### Least Squares Fit

$$
\begin{equation}
\begin{split}
RSS(\beta) &= (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta)\\
\frac{\partial RSS}{\partial \beta} &= -2 \mathbf{X}^T(\mathbf{y} - \mathbf{X}\beta) = 0
\quad \Rightarrow \quad \hat{\beta}^{ls} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
\end{split}
\end{equation}
$$

### Gauss Markov Theorem

The least squares estimates $\hat{\beta}^{ls}$ of the parameters β have the smallest variance among all linear unbiased estimates.

### Question

Is it good to be unbiased?

*** right

![Linear regression](assets/img/lr.png "Linear regression")

![Least Squares](assets/img/ls.png "Least squares")

---

## Linear Regression and Least Squares (Review)

### Bias-Variance tradeoff

$$
\begin{equation}
\begin{split}
MSE(\hat{\mathbf{y}}) &= E[(\hat{\mathbf{y}} - Y)^2]\\
&= Var(\hat{\mathbf{y}}) + [E[\hat{\mathbf{y}}] - Y]^2
\end{split}
\end{equation}
$$

where $Y = X^T\beta$. We can trade some bias for much less variance.

### Problems of Least Squares

- <b>Prediction accuracy</b>: unbiased, but high variance compared to many biased estimators, overfitting noise and sensitive to outlier
- <b>Interpretation</b>:  $\hat{\beta}$ involves all of the features.
Better to have SIMPLER linear model, that involves only a few features...
- Recall that $\hat{\beta}^{ls} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
    - $(\mathbf{X}^T\mathbf{X})$ may be <b>not invertible</b> and thus no closed form solution


---

## Part 3: Subset Selection Methods

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. <b>Subset Selection</b>
  - <b>Best-subset selection</b>
  - <b>Forward stepwise selection</b>
  - <b>Forward stagewise selection</b>
  - <b>Problems</b>
4. Shrinkage Method
5. Beyond LASSO

---

## Subset Selection Methods

### Best-subset selection


- Best subset regression finds for each $k \in \{0, 1, 2, . . . , p\}$ the subset of features of size $k$ that gives smallest RSS. 
- Then cross validation is utilized to choose the best $k$
- An efficient algorithm, the leaps and bounds procedure (Furnival and Wilson, 1974), makes this feasible for $p$ as large as 30 or 40.

<center>![best_sub](assets/img/best_sub.png "best_sub")</center>

---

## Subset Selection Methods

### Forward-STEPWISE selection

Instead of searching all possible subsets, we can seek a good path through them. 

- a <b>sequential greedy</b> algorithm.

*Forward-Stepwise Selection* builds a model sequentially, adding one variable at a time. 
- Initialization
    - Active set $\mathcal{A} = \emptyset$, $\mathbf{r} = \mathbf{y}$, $\beta = 0$
- At each step, it
    - identifies the best variable (with the highest correlation with the residual error)
$$\mathbf{k} = argmax_{j}(|correlation(\mathbf{x}_j, \mathbf{r})|)$$
    - $A = A \cup \mathbf{k}$
    - then updates the least squares fit $\beta$, $\mathbf{r}$ to include all the active variables

--- &twocolportion w1:55% w2:43%

## Subset Selection Methods

### Forward-STAGEWISE Regression

*** left
- Initialize the fit vector $\mathbf{f} = 0$
- For each time step
    - Compute the correlation vector $\mathbf{c} = (\mathbf{c}_1, ..\mathbf{c}_p)$, $\mathbf{c}_j$ represents the correlation between $\mathbf{x}_j$ and the residual error
    - $k = argmax_{j \in \{1,2,..,p\}} |\mathbf{c}_j|$
    - Coefficients and fit vector are updated
$$\mathbf{f} \gets \mathbf{f} + \alpha \cdot sign(\mathbf{c}_k) \mathbf{x}_k$$
$$\beta_k \gets \beta_k + \alpha \cdot sign(\mathbf{c}_k)$$ 
where $\alpha$ is the learning rate

***right

![Stagewise](assets/img/stagewise.png "Stagewise")


---&twocolportion w1:40% w2:55%

## Subset Selection Methods

### Comparison
*** left
- Forward-STEPWISE selection: 
    - algorithm stops in $p$ steps
- Forward-STAGEWISE selection: 
    - is a slow fitting algorithm, at each time step, only $\beta_k$ is updated. Alg can take more than $p$ steps  to stop

*** right

<center>![comp1](assets/img/comp1.png "comp")</center>

- $N = 300$ Observations, $p = 31$ features
- averaged over 50 simulations

---

## Summary of Subset Selection Methods

### Advantages

- More interpretable result
- More compact model

### Disadvantages

- It is a discrete process, and thus has high variance and sensitivity to the change in dataset.
    - If the dataset changes a little, the feature selection result may be very different
- Thus may not be able to lower prediction error

---


## Part 4: Shrinkage Method

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. <b>Shrinkage Method</b>
    - <b>Ridge Regression</b>
        - <b>Formulations and closed form solution</b>
        - <b>Singular value decomposition</b>
        - <b>Degree of Freedom</b>
    - LASSO
5. Beyond LASSO

---

## Ridge Regression

- Least squares with quadratic constraints
$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x}_{ij}\beta_j)^2, \quad s.t. \quad \sum_{j = 1}^p \beta_j^2 \leq t
\end{equation}
$$

- Simulation Experiment
    - $N = 30$
    - $\mathbf{x}_1 \sim N(0, 1)$, $\mathbf{x}_2 = \mathbf{x}^2_1$
    - $\beta \sim U(-0.5,0.5)$
    - $Y = (\mathbf{x}_1, \mathbf{x}_2) \times \beta$
    - $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}^2_1, ..., \mathbf{x}^8_1)$
    
---

---

## Ridge Regression

- Simulation Experiment

<center>![lst](assets/img/lst.png "lst")</center>


---

## Ridge Regression

- <b>Least squares with quadratic constraints</b>
$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x}_{ij}\beta_j)^2, \quad s.t. \quad \sum_{j = 1}^p \beta_j^2 \leq t
\end{equation}
$$
- <b>Its Lagrange form</b>
$$
\hat{\beta}^{ridge} = argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2 + \lambda \sum_{j = 1}^p\beta_j^2
$$
- The $l_2$-regularization can be viewed as a Gaussian prior on the coefficients, our solution as the posterior means

- <b>Solution</b>

$$
\begin{equation}
\begin{split}
&RSS(\beta) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta) + \lambda \beta^T\beta\\
&\partial RSS(\beta)/ \partial \beta = 0  \quad \Rightarrow\quad \hat{\beta}^{ridge} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
\end{split}
\end{equation}
$$    

---&twocolportion w1:50% w2:50%

## Ridge Regression

### Singular Value Decomposition (SVD)
SVD offers some additional insight into the nature of ridge regression. 

*** left

- <b>The SVD of</b> $\mathbf{X}$:
$$\mathbf{X} = \mathbf{UDV}^T$$
    - $\mathbf{U}$: $N \times p$ <b>orthogonal</b> matrix with columns spanning the column space of $\mathbf{X}$. 
        - $\mathbf{u}_j$ is the $j$th column of $\mathbf{U}$
    - $\mathbf{V}$: $p \times p$ <b>orthogonal</b> matrix with columns spanning the row space of $\mathbf{X}$. 
        - $\mathbf{v}_j$ is the $j$th column of $\mathbf{V}$  
    - $\mathbf{D}$: $p \times p$ <b>diagonal</b> matrix with diagonal entries $d_1 \geq d_2 \geq ... \geq d_p \geq 0$ being the singular values of $\mathbf{X}$

*** right

<center>![svd2](assets/img/svd2.png "svd2")</center>

<center>![svd](assets/img/svd.gif "svd")</center>

---&triple w1:50% w2:50%

## Ridge Regression

### Singular Value Decomposition (SVD)

*** left

- <b>For least squares</b>
$$
\begin{equation}
\begin{split}
\mathbf{X}\hat{\beta}^{ls} &= \mathbf{X(X^TX)^{-1}X^Ty}\\
&=\mathbf{UU^Ty} =\sum_{j=1}^p\mathbf{u}_j \mathbf{u}_j^T\mathbf{y}
\end{split}
\end{equation}
$$

*** right
- <b>For ridge regression</b>
$$
\begin{equation}
\begin{split}
\mathbf{X}\hat{\beta}^{ridge} &= \mathbf{X(X^TX + \lambda I)^{-1}X^Ty}\\
&=\sum_{j=1}^p\mathbf{u}_j\frac{d_j^2}{d_j^2 + \lambda} \mathbf{u}_j^T\mathbf{y}
\end{split}
\end{equation}
$$

*** down

- Compared with the solution of least squares, we have an additional shrinkage term 
$$\frac{d_j^2}{d_j^2 + \lambda},$$ 
the smaller $d_j$ is and the larger $\lambda$ is, the more shrinkage we have. 




---&twocolportion w1:50% w2:50%

## Ridge Regression

### Singular Value Decomposition (SVD) 

*** left

- $N = 100$, $p = 10$

<center>![ls_pc](assets/img/ls_pc.png "ls_pc")</center>

<center>![rr_pc](assets/img/rr_pc.png "rr_pc")</center>

*** right

<center>![shrink](assets/img/shrink.png "shrink")</center>



--- &twocolportion w1:40% w2:55%

## Ridge Regression

### Degree of Freedom

*** left
- The number of degrees of freedom is the number of values in the final calculation of a statistic that are free to vary. The degree of freedom of ridge estimate is related to $\lambda$, thus defined as $df(\lambda)$.
- Computation
$$
\begin{equation}
\begin{split}
df(\lambda) &= tr[\mathbf{X(X^TX + \lambda I)^{-1}X^T}]\\
&=\sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} 
\end{split}
\end{equation}
$$
- [larger $\lambda$] $\rightarrow$ [smaller $df(\lambda)$] $\rightarrow$ [more constrained model].

*** right

<center>![df](assets/img/df.png "df")</center>

---

## Ridge Regression

### Advantages

- $(\mathbf{X^TX + \lambda I})$ is always inveritible and thus the closed form solution always exist
- Ridge regression controls the complexity with regularization term via $\lambda$, which is less prone to overfitting compared with least squares fit, 
     - e.g. sometimes a wildly large coefficient on one variable can be cancelled by another wildly large coefficient of a correlated variable
- Possibly higher prediction accuracy, as the estimates of ridge regression trade a little bias for less variance

### Disadvantages

- Interpretability and compactness: Though coefficients are shrunk, but not to zero. Unlike methods that select part of the features, ridge regression may encounter efficiency issue and offer little interpretations in high dimensional problems.

---

## Part 4: Shrinkage Method - LASSO

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - <b>Formulations</b>
        - <b>Comparisons with ridge regression and subset selection</b>
        - <b>Quadratic Programming</b>
        - <b>Least Angle Regression</b>
        - <b>Viewed as approximation for $l_0$-regularization</b>
5. Beyond LASSO

---


## LASSO

### Linear regression with $l_1$-regularization

- <b>Formulations</b>

    
    - <b>Least squares with constraints</b>
$$
\begin{equation}
\hat{\beta}^{LASSO}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2, \quad s.t. \sum_{j = 1}^p |\beta_j| \leq t
\end{equation}
$$
    - <b>Its lagrange form</b>
$$
\hat{\beta}^{LASSO} = argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2 + \lambda \sum_{j = 1}^p|\beta_j|
$$
    - The $1_1$-regularization can be viewed as a Laplace prior on the coefficients

---&twocol

## LASSO

- $s = \frac{t}{\sum_{j=1}^p |\hat{\beta}_j|}$, where $\hat{\beta}$ is the least square estimates.
- Redlines represent the $s$ and $df(\lambda)$ with the best cross validation errors

*** left

<center>![LASSO](assets/img/LASSO.png "LASSO")</center>


*** right

<center>![df](assets/img/df.png "df")</center>





---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - Formulations
        - <b>Comparisons with ridge regression and subset selection</b>
            - <b>Orthonormal inputs</b>
            - <b>Non-orthonormal inputs</b>
        - Quadratic Programming
        - Least Angle Regression
        - Viewed as approximation for $l_0$-regularization
- Beyond LASSO


---

## LASSO

### Comparison

- <b>Orthonormal Input $\mathbf{X}$</b>
    - <b>Best subset</b>: [Hard thresholding] keep the top $M$ largest coefficeints of $\hat{\beta}^{ls}$
    - <b>Ridge</b>: [Pure shrinkage] does proportional shrinkage of $\hat{\beta}^{ls}$
    - <b>LASSO</b>: [Soft thresholding] translates each coefficient of $\hat{\beta}^{ls}$ by $\lambda$ towards 0, truncating at 0 


<center>![comp2](assets/img/comp2.png "comp2")</center>

---&twocolportion w1:60% w2:40%

## LASSO

### Comparison

- <b>Non-orthonormal Input $\mathbf{X}$</b>

*** left

<center>![comp3](assets/img/comp3.png "comp3")</center>

*** right

<b>Solid blue area</b>: the constraints
- left: $|\beta_1| + |\beta_1| \leq t$
- right: $\beta_1^2 + \beta_1^2 \leq t^2$

<b>$\hat{\beta}$</b>: least squares fit

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

$|\beta_j|^q$ can be viewed as the log-prior density for $\beta_j$, these three methods below are bayes estimates with different priors

- <b>Subset selection</b>: corresponds to $q = 0$
- <b>LASSO</b>: corresponds to $q = 1$, Laplace prior, $density = (\frac{1}{\tau})exp(\frac{-|\beta|}{\tau}), \tau = \sigma/\lambda$
- <b>Ridge regression</b>: corresponds to $q = 2$, Gaussian Prior, $\beta \sim N(0, \tau \mathbf{I})$, $\lambda = \frac{\sigma^2}{\tau^2}$

*** left

<center>![laplace](assets/img/laplace.png "laplace")</center>

*** right

<center>![gauss](assets/img/gauss.png "gauss")</center>


---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - <b>Quadratic Programming</b>
        - Least Angle Regression
        - Viewed as approximation for $l_0$-regularization
- Beyond LASSO

---

## LASSO

### Quadratic Programming

- Formulation
$$
min_{\beta}\{ \frac{1}{2}(\mathbf{X}\beta - \mathbf{y})^T (\mathbf{X}\beta - \mathbf{y}) + \lambda \|\beta\|_1\}
$$
is equivalent to 
$$
min_{w, \xi}\{ \frac{1}{2}(\mathbf{X}\beta - \mathbf{y})^T (\mathbf{X}\beta - \mathbf{y}) + \lambda \mathbf{1}^T\xi\}
$$

$$
\begin{equation}
\begin{split}
s.t. &\beta_j \leq \xi_j\\
&\beta_j \geq -\xi_j
\end{split}
\end{equation}
$$

- Note that QP can only solve LASSO for a given $\lambda$. 
    - Later in this slide, a method called least angle regression can solve LASSO for all $\lambda$

---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - Quadratic Programming
        - <b>Least Angle Regression</b>
        - Viewed as approximation for $l_0$-regularization
- Beyond LASSO


---&twocolportion w1:55% w2:45%

#### LAR Algorithm

*** left

- Initialization: 
    - Standardized all predictors s.t. $\bar{\mathbf{x}_j} = 0, \mathbf{x}_j^T\mathbf{x}_j = 1$; $\mathbf{r}_0 = \mathbf{y} - \bar{\mathbf{y}}$; $\beta = \mathbf{0}$; $\mathbf{f}_0 = \mathbf{0}$;
    - $k = argmax_{j} |\mathbf{x}_j^T \mathbf{r}_0|$, $\mathcal{A}_1 = \{k\}$    
- Main
    - for time step $t = 1,2,...min(N-1,p)$
      - $\mathbf{r}_t = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \beta_{\mathcal{A}_t}$, $\quad \mathbf{f}_t = \mathbf{X}_{\mathcal{A}_t} \beta_{\mathcal{A}_t}$
      - $\delta_t = \mathbf{(X^T_{\mathcal{A}_t} X_{\mathcal{A}_t})^{-1} X^T_{\mathcal{A}_t}r_t}$, $\quad \mathbf{u}_t = \mathbf{X}_{\mathcal{A}_t} \delta_t$
      - Search $\alpha$
          - $\beta_{\mathcal{A}_t}(\alpha) = \beta_{\mathcal{A}_t} + \alpha \cdot \delta_t$
          - Concurrently, $\mathbf{f}_t(\alpha) = \mathbf{f}_t + \alpha \cdot \mathbf{u}_t$
      - Until $|\mathbf{X}_{\mathcal{A}_t} \mathbf{r}_t(\alpha)| = max_{\mathbf{x}_j \in \bar{\mathcal{A}_t}} |\mathbf{x}_j^T \mathbf{r}_t(\alpha)|$
      - $k = argmax_{j \in \bar{\mathcal{A}_t}} |\mathbf{x}_j \mathbf{r}_t(\alpha)|$, 
      - $\mathcal{A}_{t+1} = \mathcal{A}_{t} \cup \{k\}$


*** right

- $\mathcal{A}_t$: <i>active set</i>, the set indices of features we already included in the model at time step $t$.
    - $\mathcal{A}_t \cup \bar{\mathcal{A}_t} = \{1,2,...,p\}$
    - $\mathcal{A}_t \cap \mathcal{\bar{A}}_t = \emptyset$
- $\alpha$: searching parameter within a time step
- $\beta_{\mathcal{A}_t}$: coefficients vector at the beginning of time step $t$
- $\beta_{\mathcal{A}_t}(\alpha)$: coefficients vector in time step $t$ w.r.t. $\alpha$
- $\mathbf{f}_t$: the fit vector at the beginning of time step $t$, $\mathbf{f}_0 = 0$
- $\mathbf{f}_t(\alpha)$: the fit vector in time step $t$ w.r.t. $\alpha$
- $\mathbf{r}_t$: residual vector at the beginning of time step $t$, $\mathbf{r}_0 = \mathbf{y} - \bar{\mathbf{y}}$, where $\bar{\mathbf{y}} = average(\mathbf{y})$
- $\mathbf{r}_t(\alpha)$: residual vector in time step $t$, w.r.t. $\alpha$


---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars1](assets/img/lars1.png "lars1")</center>

*** right

- Initialization: 
    - Standardized each columns of $\mathbf{X}$ 
        - s.t. $\bar{\mathbf{x}_j} = 0$, $\mathbf{x}_j^T\mathbf{x}_j = 1$
    - $\mathbf{r}_0 = \mathbf{y} - \bar{\mathbf{y}}$; 
    - $\beta = (0, 0)^T$; 
    - $\mathcal{A}_0 = \emptyset$
    - $\mathbf{f}_0$ is the current fit at time $0$ 
        - $\mathbf{f}_0 = (0, 0)^T$
    

---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars2](assets/img/lars2.png "lars2")</center>

*** right

- $k = argmax_{j} |\mathbf{x}_j^T \mathbf{r}_0| = 1$
- $\mathcal{A}_1 = \{1\}$


---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars3](assets/img/lars3.png "lars3")</center>

*** right
- $\mathbf{r}_1 = \mathbf{y} - \mathbf{X}_{\mathcal{A}_1} \beta_{\mathcal{A}_1}$
- $\delta_1 = \mathbf{(X^T_{\mathcal{A}_1} X_{\mathcal{A}_1})^{-1}X^T_{\mathcal{A}_1}r_1}$
- $\mathbf{u}_1 = \mathbf{X}_{\mathcal{A}_1} \delta_1$


---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars4](assets/img/lars4.png "lars4")</center>

*** right

### Explanations

- Search $\alpha$
    - $\beta_{\mathcal{A}_1}(\alpha) = \beta_{\mathcal{A}_1} + \alpha \cdot \delta_1$, 
    - $\mathbf{f}_1(\alpha) = \mathbf{f}_1 + \alpha \cdot \mathbf{u}_1$
    - $\mathbf{r}_1(\alpha) = \mathbf{y} - \mathbf{X}_{\mathcal{A}_1} \beta_{\mathcal{A}_1}(\alpha)$
- Until $|\mathbf{X}_{\mathcal{A}_1} \mathbf{r}_1(\alpha)| = max_{j \in \bar{\mathcal{A}_1}} |\mathbf{x}_j^T \mathbf{r}_1(\alpha)|$
- $2 = argmax_{j \in \bar{\mathcal{A}_1}} |\mathbf{x}_j^T \mathbf{r}_1(\alpha)|$
- $\mathcal{A}_2 = \{1, 2\}$

### Comments

- $\mathbf{f}_t$ is approaching $\mathbf{f}_t^{ls}$, but never reaches it, except for the final step of LAR

---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars5](assets/img/lars5.png "lars5")</center>

*** right

### Explanations

- $\mathbf{r}_2 = \mathbf{y} - \mathbf{X}_{\mathcal{A}_2} \beta_{\mathcal{A}_2}$
- $\delta_2 = \mathbf{(X^T_{\mathcal{A}_2} X_{\mathcal{A}_2})^{-1}X^T_{\mathcal{A}_2}r_2}$
- $\mathbf{u}_2 = \mathbf{X}_{\mathcal{A}_2} \delta_2$

### Comments

* the direction $\mathbf{u}_k = \mathbf{X}_{\mathcal{A}_k} \delta_k$ that our fit $\mathbf{f}_k(\alpha)$ increases actually has the same angle with any $\mathbf{x}_j \in \mathcal{A}_k$.


---&twocolportion w1:55% w2:45%

## LAR - Example

*** left

<center>![lars6](assets/img/lars6.png "lars6")</center>

*** right

- If $p = 2$
    - $\mathbf{f}_2 = \mathbf{f}_2^{LeastSquares}$
- The absolute values of correlations of $\mathbf{x}_j \in \mathcal{A}_k, \forall j$ with the residual error $\mathbf{r}_t{\alpha}$ are tied and decrease at the same rate during searching $\alpha$.




---

## LAR

### More Comments


- The procedure of searching is approaching the least-squares coefficients of fitting $\mathbf{y}$ on $\mathcal{A}_k$
- LAR solves the subset selection problem for all $t, s.t. \|\beta\| \leq t$
- Actually, $\alpha$ can be computed instead of searching
- LAR algorithm ends in $min(p, N-1)$ steps



--- 

## LAR

### Result compared with LASSO

### Observations

When the blue line coefficient cross zero, LAR and LASSO become different.


<center>![comp4](assets/img/comp4.png "comp4")</center>

--- 

## LAR

### Result compared with LASSO

### Modification for LASSO

During the searching procedure, if a non-zero coefficient hits zero, drop this variable from $\mathcal{A}_k$, and recompute the direction $\delta_k$


<center>![comp4](assets/img/comp4.png "comp4")</center>



---

## LAR

### Some heuristic analysis

- At a certain time point, we know that all $\mathbf{x}_j \in \mathcal{A}$ share the same absolute values of correlations with the residual error. That is
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta) = \gamma \cdot s_j, \quad \forall j \ \in \mathcal{A}$$
where $s_j \in \{-1,1\}$ indicates the sign of the left hand inner product and $\gamma$ is the common value. 
    - We also know that $|\mathbf{x_j}(\mathbf{y} - \mathbf{X}\beta)| \leq \gamma, \quad \forall \mathbf{x}_j \not\in \mathcal{A}$

- Consider LASSO for a fixed $\lambda$. Let $\mathcal{B}$ be the set of indices of non-zero coefficients, then we differentiate the objective function w.r.t. those coefficients in $\mathcal{B}$ and set the gradient to zero. We have
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta) = \lambda \cdot sign(\beta_j), \quad \forall j \in \mathcal{B}$$

- They are identical only if $sign(\beta_j)$ matches the sign of the lefthand side. In $\mathcal{A}$, we allow for the $\beta_j$, where $sign(\beta_j) \neq sign(\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta))$, while this is forbidden in $\mathcal{B}$. 


---

## LAR

### Some heuristic analysis

-  For LAR, we have 
$$|\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta)| \leq \gamma, \quad \forall \mathbf{x}_j \not\in \mathcal{A}$$
- According to the stationary conditions, for LASSO, we have
$$
|\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta)| \leq \lambda, \quad \forall \mathbf{x}_j \not\in \mathcal{B}
$$
- LAR and Lasso match for variables with zero coefficients too.



---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - Quadratic Programming
        - Least Angle Regression
        - <b>Viewed as approximation for $l_0$-regularization</b>
- Beyond LASSO

---

## Viewed as approximation for $l_0$-regularization

### Pure variable selection

$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2, \quad s.t. \#nonzero \beta_j \leq t
\end{equation}
$$

Actually $\#nonzero \beta_j = \|\beta\|_0$, where

$$\|\beta\|_0 = lim_{q \to 0}(\sum_{j = 1}^p|\beta_j|^q)^{\frac{1}{q}} = card(\quad \{\beta_j|\beta_j \neq 0\}\quad)$$

<center>![zeronorm](assets/img/zeronorm.png "zeronorm")</center>

---

## Viewed as approximation for $l_0$-regularization

### Problem

$l_0$-norm is not convex, which makes it very hard to optimize.

### Solutions

- <b>LASSO</b>: Approximated objective function ($l_1$-norm), with exact optimization
- <b>Subset selection</b>: Exact objective function, with approximated optimization (greedy strategy)

---

## Part 5: Beyond LASSO

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. Shrinkage Method
5. <b>Beyond LASSO</b>
   - <b>Elastic-Net</b>
   - <b>Fused LASSO</b>
   - <b>Group LASSO</b>   
   - <b>$l_1-lp$ norm</b>
   - <b>Graph-guided LASSO</b>


--- &triple w1:41% w2:55% 

## Beyond LASSO - Elastic Net

### Problems with LASSO

- LASSO tends to rather arbitrarily select one of a group of highly correlated variables (see how LAR works). Sometimes, it is better to select <b>ALL</b> the relevant varibles in a group
- LASSO selects at most $N$ variables, when $p > N$, which may be undisirable when $p >> N$
- The performance of Ridge dominates that of LASSO, when $N > p$ and variables are correlated

### Elastic Net

*** left

- <b>Penalty Term</b>
$$\lambda \sum_{j = 1}^p (\alpha \beta_j^2 + (1-\alpha)|\beta_j|)$$
which is a compromise between ridge regression and LASSO and $\alpha \in [0,1]$.

*** right

<center>![enet](assets/img/enet.png "enet")</center>


--- &triple w1:41% w2:55% 

## Beyond LASSO - Elastic Net

### Advantages of E-Net
- Solves above problems
- elects variables like LASSO, and shrinks together the coefficients of correlated predictors like ridge.
- has considerable computational advantages over the $l_q$ penalties. 
    - See 18.4 [Elements of Statistical Learning]

### Elastic Net

*** left

- <b>Penalty Term</b>
$$\lambda \sum_{j = 1}^p (\alpha \beta_j^2 + (1-\alpha)|\beta_j|)$$
which is a compromise between ridge regression and LASSO and $\alpha \in [0,1]$.

*** right

<center>![enet](assets/img/enet.png "enet")</center>


---

## Elastic Net - A simple illustration

- Two independent “hidden” factors $\mathbf{z}_1$ and $\mathbf{z}_2$
$$\mathbf{z}_1 \sim U(0, 20),\quad \mathbf{z}_2 \sim U(0, 20),$$
- Generate the response vector $\mathbf{y} = \mathbf{z}_1 + 0.1\mathbf{z}_2 + N(0,1)$
- Suppose the observed features are
$$\mathbf{x}_1 = \mathbf{z}_1 + \epsilon_1,\quad \mathbf{x}_2 = -\mathbf{z}_1 + \epsilon_2,\quad \mathbf{x}_3 = \mathbf{z}_1 + \epsilon_3$$
$$\mathbf{x}_4 = \mathbf{z}_2 + \epsilon_4,\quad \mathbf{x}_5 = -\mathbf{z}_2 + \epsilon_5,\quad \mathbf{x}_6 = \mathbf{z}_2 + \epsilon_6$$
where $\epsilon$ is $i.i.d.$ random noise.
- Fit the model on data $(\mathbf{X}, \mathbf{y})$
- A good model should identify that only $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$ are important


---

## Elastic Net - A simple illustration

<center>![enet_LASSO](assets/img/enet_LASSO.png "enet_LASSO")</center>

---

## Elastic Net - A simple illustration

<center>![enet_LASSO](assets/img/enet_re.png "enet_re")</center>


---

## Beyond LASSO - Fused LASSO

### Fused LASSO

- <b>Intuition</b>
    - Fused LASSO is designed for problems with features that can be ordered in some meaningful way, where "adjacent features" should have similar importance. 
    - The fused LASSO penalizes the $L_1$-norm of both the coefﬁcients and their successive differences.
- <b>Example</b>
    - Classification with fMRI data: each voxel has about 200 measurements over time. The coeefficients for adjacent voxels should be similar
- <b>Formulation</b>

$$\hat{\beta} = argmin_{\beta}\{\|\mathbf{X\beta - y}\|_2^2\}$$
$$s.t. \|\beta\| \leq s_1 \quad and \quad \sum_{j = 2}^p |\beta_j - \beta_{j-1}| \leq s_2$$

---

## Beyond LASSO - Fused LASSO

### Fused LASSO

<center>![nb_fs](assets/img/nb_fs.png "nb_fs")</center>



---

## Fused LASSO - Simulation results

<center>![fLASSO](assets/img/fLASSO.png "fLASSO")</center>

- $p = 100$. Black lines are the true coefficients.
- (a) Univeriate regression coefficients (red), a soft threshold version of them (green)
- (b) LASSO solution (red), $s_1 = 35.6,\quad s_2 = \infty$
- (c) Fusion estimate, $s_1 = \infty, s_2 = 26$
- (d) Fused LASSO, $s_1 = \sum |\beta_j|,\quad s_2 = \sum |\beta_j - \beta_{j-1}|$

---

## Beyond LASSO - Group LASSO

### Group LASSO

- <b>Intuition</b>
    - Features are divided into $L$ groups
    - Features within the same group should share similar coefficients
- <b>Example</b>
    - Binary dummy variables from one single discrete variable, e.g. $stage\_cancer \in \{1,2,3\}$ can be translated into three binary dummy variables $(stage1, stage2, stage3)$ 
- <b>Formulations</b>
$$obj = \left\|\mathbf{y} - \sum_{l = 1}^L \mathbf{X}_l \beta_l \right\|_2^2 + \lambda_1 \sum_{l = 1}^L\left\|\beta_l\right\|_2 + \lambda_2 \left\|\beta\right\|_1$$


---

## Group LASSO - Simulation Results

- Generate $n = 200$ observations with $p = 100$, divided into ten blocks equally
- The number of non-zero coefficients in blocks are 
    - block 1: 10 out of 10
    - block 2: 8 out of 10
    - block 3: 6 out of 10
    - block 4: 4 out of 10
    - block 5: 2 out of 10
    - block 6-10: 0 out of 10
- The coefficients are either -1 or +1, with the sign being chosen randomly.
- The predictors are standard Gaussian with correlation 0.2 within a group and zero otherwise
- A Gaussian noise with standard deviation 4.0 was added to each observation

---

## Group LASSO - Simulation Results

<center>![gl2](assets/img/gl2.png "gl2")</center>


---

## Beyond LASSO - $l_1$-$l_p$ penalization

### $l_1$-$l_p$ penalization

- <b>Applies to multi-task learning</b>, where the goal is to estimate predictive models for several related tasks. 
- <b>Examples</b>
    - <b>Example 1</b>: recognize speech of different speakers, or handwriting of different writers, 
    - <b>Example 2</b>: learn to control a robot for grasping different objects
    - <b>Example 3</b>:learn to control a robot for driving in different landscapes 
- <b>Assumptions about the tasks</b>
    - sufficiently <i>different</i> that learning a specific model for each task results in improved performance
    - <i>similar</i> enough that they share some common underlying representation that should make simul- taneous learning beneficial. 
    - different tasks share a subset of relevant features selected from a large common space of features.

---

## Beyond LASSO - $l_1$-$l_p$ penalization

### $l_1$-$l_p$ penalization

- <b>Formulation</b>
    - $\mathbf{X}_l$: $N \times p$ input matrix for task $l = 1..L$
        - $L$ is the total number of tasks
    - $\beta$: $p \times L$ coefficient matrix
    - $\mathbf{y}$: $N \times L$ output matrix
    - objective function
        $$obj = \sum_{l= 1}^L J(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \sum_{j = 1}^p \|\beta_{j:}\|_2$$
      where $J$ is some loss function and $\sum_{j = 1}^p \|\beta_{:j}\|_2$ is the $l_1$ norm of vector $(\|\beta_{:1}\|_2, \|\beta_{:2}\|_2, ..., \|\beta_{:p}\|_2)$.
    

---

## Beyond LASSO - $l_1-l_p$ penalization

### $l_1-l_p$ penalization -Coefficient matrix

<center>![l1lp](assets/img/l1lp.png "l1lp")</center>


---

## $l_1-l_p$ penalization - Experiment Result

- <b>Dataset</b>: handwritten words dataset collected by Rob Kassel
    - Contains writings from more than 180 different writers.
    - For each writer, the number of each letter we have is between 4 and 30
    - The letters are originally represented as $8 \times 16$
- <b>Task</b>: build binary classiers that discriminate between pairs of letters. Specically concentrat on the pairs of letters that are the most difficult to distinguish when written by hand.    
- <b>Experiment</b>: learned classications of 9 pairs of letters for 40 different writers

<center>![write](assets/img/write.png "write")</center>

---

## $l_1-l_p$ penalization - Experiment Result

- <b>Candidate methods</b>
    - Pooled $l_1$: a classifier is trained on all data regardless of writers
    - Independent $l_1$ regularization: For each writer, a classifier is trained
    - $l_1/l_1$-regularization:
    $$obj = \sum_{l= 1}^L J(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \sum_{l = 1}^L \|\beta_{:l}\|_1$$
    - $l_1/l_2$-regularization:
    $$obj = \sum_{l= 1}^L J(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \sum_{j = 1}^p \|\beta_{j:}\|_2$$    
   
---

---

## $l_1-l_p$ penalization - Experiment Result

<center>![l1lp_re](assets/img/l1lp_re.png "l1lp_re")</center>

- Within a cell,  the first row contains results for feature selection, the second row uses random projections to obtain a common subspace (details ommited, see paper: Multi-task feature selection)
- Bold: best of $l_1/l_2$,$l_1/l_1$, $sp.l_1$ or pooled $l_1$, Boxed : best of cell


---

## Beyond LASSO - Graph-Guided Fused LASSO

### Graph-Guided Fused LASSO (GFLASSO)

- <b>Example</b>
<center>![gfLASSO](assets/img/gfLASSO.png "gfLASSO")</center>
- <b>Formulation</b>
Graph-Guided LASSO applies to multi-task settings
$$obj = \sum_{l= 1}^L loss(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \|\beta\|_1+\gamma \sum_{e=(a,b)\in E}^p \tau(r_{ab}) \sum_{j = 1}^p |\beta_{ja} - sign(r_{a,b})\beta_{jb}|$$
where $r_{a,b} \in \mathbb{R}$ denotes the weight of the edge and $\tau(r)$ can be any positive monotonically increasing function of $|r|$, e.g. $\tau(r) = |r|$.

---&twocol

## Beyond LASSO - Graph-Guided Fused LASSO

### Graph-Guided Fused LASSO

<center>![gfLASSO_re](assets/img/gfLASSO_re.png "gfLASSO_re")</center>

*** left

- (a) The true regression coefficients
- (c) $l_1/l_2$-regularized multi-task regression

*** right

- (b) LASSO
- (d) GFLASSO


---

## Summary

### Outline

- <h3>Introduction to Dimension Reduction</h3>

- <h3>Linear Regression and Least Squares (Review)</h3>

- <h3>Subset Selection</h3>

- <h3>Shrinkage Method</h3>

- <h3>Beyond LASSO</h3>


---

## Summary

### Part 1: Introduction to Dimension Reduction

1. <b>Introduction to Dimension Reduction</b>
    - <b>General notations</b>
    - <b>Motivations</b>
    - <b>Feature selection and feature extraction</b>
    - <b>Feature Selection</b>
        - <b>Wrapper method</b>
        - <b>Filter method</b>
        - <b>Embedded method</b>
    - <b>Feature Extraction</b>
        - <b>PCA, ICA...</b>
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. Shrinkage Method
5. Beyond LASSO


---

## Summary

### Part 2: Linear Regression and Least Squares (Review)

1. Introduction to Dimension Reduction
2. <b>Linear Regression and Least Squares (Review)</b>
    - <b>Least Square Fit</b>
    - <b>Gauss Markov</b>
    - <b>Bias-Variance tradeoff</b>
    - <b>Problems</b>
3. Subset Selection
4. Shrinkage Method
- Beyond LASSO


---

## Summary

### Part 3: Subset Selection

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. <b>Subset Selection</b>
  - <b>Best-subset selection</b>
  - <b>Forward stepwise selection</b>
  - <b>Forward stagewise selection</b>
  - <b>Problems</b>
4. Shrinkage Method
5. Beyond LASSO


---

## Summary

### Part 4: Shrinkage Method - Ridge Regression

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. <b>Shrinkage Method</b>
    - <b>Ridge Regression</b>
        - <b>Formulations and closed form solution</b>
        - <b>Singular value decomposition</b>
        - <b>Degree of Freedom</b>
    - LASSO
5. Beyond LASSO

---

## Summary

### Part4 Shrinkage Method - LASSO

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>LASSO</b>
        - <b>Formulations</b>
        - <b>Comparisons with ridge regression and subset selection</b>
        - <b>Quadratic Programming</b>
        - <b>Least Angle Regression</b>
        - <b>Viewed as approximation for $l_0$-regularization</b>
5. Beyond LASSO


---

## Summary

### Part 5: Beyond LASSO

1. Introduction to Dimension Reduction
2. Linear Regression and Least Squares (Review)
3. Subset Selection
4. Shrinkage Method
5. <b>Beyond LASSO</b>
   - <b>Elastic-Net</b>
   - <b>Fused LASSO</b>
   - <b>Group LASSO</b>   
   - <b>$l_1-lp$ norm</b>
   - <b>Graph-guided LASSO</b>


---

## More on the topics skipped here

- More on feature extraction methods: 
    - http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
    - Imola K. Fodor, A survey of dimension reduction techniques
    - Christopher J. C. Burges, Dimension Reduction: A Guided Tour
    - Ali Ghodsi, Dimensionality Reduction A Short Tutorial
- Mutual-info-based feature selection: 
    - Gavin Brown, Adam Pocock, Ming-Jie Zhao, Mikel Luján; Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection
    - Howard Hua Yang, John Moody. Feature Selection Based on Joint Mutual Information
    - Hanchuan Peng, Fuhui Long, and Chris Ding. Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy
- Beyond LASSO
    - http://webdocs.cs.ualberta.ca/~mahdavif/ReadingGroup/




--- &vcenter  

## Sparse Models

### Thank You!


---

## Reference

- Trevor Hastie, Robert Tibshirani and Jerome Friedman. Elements of Statistical Learning <font color = 'green'>[p7, p15, p16, p18, p19, p21-22, p26-27, p29-30, p33, p35-37, p42-p43, p50-p54, p56, p59]</font>
- Temporal Sequence of FMRI scans (single slice): from http://www.midwest-medical.net/mri.sagittal.head.jpg <font color = 'green'>[p8]</font>
- Three Dimensional Image of Brain Activation from http://www.fmrib.ox.ac.uk/fmri_intro/brief.html <font color = 'green'>[p8]</font>
- http://en.wikipedia.org/wiki/Feature_selection <font color = 'green'>[p10-12]</font>
- http://en.wikipedia.org/wiki/Singular_value_decomposition <font color = 'green'>[p27]</font>
- http://en.wikipedia.org/wiki/Normal_distribution <font color = 'green'>[p38]</font>
- http://en.wikipedia.org/wiki/Laplacian_distribution <font color = 'green'>[p38]</font>
- http://webdocs.cs.ualberta.ca/~mahdavif/ReadingGroup/Papers/larS.pdf <font color = 'green'>[p20]</font>
- Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani. Least Angle Regression <font color = 'green'>[p20]</font>

---

## Reference
    
- Kevin P. Murphy. Machine Learning A Probabilistic Perspective<font color = 'green'>[p59]</font>
- Prof.Schuurmans' notes on LASSO <font color = 'green'>[p40]</font>
- Conditional Likelihood Maximisation: A Unifying Framework for
Information Theoretic Feature Selection <font color = 'green'>[p8]</font>
- Hui Zou and Trevor Hastie. Regularization and Variable Selection via the Elastic Net <font color = 'green'>[p59-62]</font>
- http://www.stanford.edu/~hastie/TALKS/enet_talk.pdf <font color = 'green'>[p59-62]</font>
- Robert Tibshirani and Michael Saunders, Sparsity and smoothness via the fused LASSO <font color = 'green'>[P63-p65]</font>
- Jerome Friedman Trevor Hastie and Robert Tibshirani. A note on the group LASSO and a sparse group LASSO <font color = 'green'>[p66-68]</font>
- Guillaume Obozinski, Ben Taskar, and Michael Jordan. Multi-task feature selection <font color = 'green'>[p69-70, p72-p75]</font>
- Xi Chen, Seyoung Kim, Qihang Lin, Jaime G. Carbonell, Eric P. Xing. Graph-Structured Multi-task Regression and an Efficient Optimization Method for General Fused LASSO <font color = 'green'>[p76-77]</font>

