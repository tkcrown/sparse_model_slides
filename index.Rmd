---
title       : Sparse Models
subtitle    : CMPUT 466/551
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

--- &twocolportion w1:40% w2:56%

## Difference between Feature Selection and Feature Extraction

*** left

### Feature Selection

- chooses a subset of features from the original feature set

### Feature Extraction

- transforms the original features into new ones
- e.g. projects data from high dimensions to low dimensions

### Question: 

- Why do we need two frameworks for dimension reduction?

*** right

<center>![fs](assets/img/fs.png "fs")</center>

<center>![fe](assets/img/fe.png "fe")</center>

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

### Example 2: classification with fMRI data
fMRI data are 4D images, with one dimension being the time slot.

<center>![fMRI](assets/img/fMRI.png "fmri")</center>



---

## Difference between Feature Selection and Feature Extraction

### Example 2: classification with fMRI data

- fMRI data are 4D images, with one dimension being the time slot. 

- Suppose the dimension of images is $50 \times 50 \times 50$ for single time point and we have 200 time points

- $50 \times 50 \times 50 \times 200 = 25,000,000$ dimensions in total! This will cause great computation burdun

In this task, we are not concerned about importance of particular voxels. Our purpose is to decrease the number of dimensions without losing too much information for further prediction task. 

Feature extraction applies better.



---

## Feature Selection

### Wrapper Methods

- search the space of feature subsets
- use the training/validation accuracy of a particular classifier as the measure of utility for a candidate subset

### Embedded Methods
- exploit the structure of speciﬁc classes of learning models to guide the feature selection process
- e.g. LASSO. It is embedded as part of the model construction process

### Filter Methods

- use some general rules/criterions to measure the feature selection results independent of the classifiers
- e.g. mutual information based method

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

- <b>Prediction accuracy</b>: low bias, but high variance, overfitting noise and sensitive to outlier
- <b>Interpretation</b>: Sometimes, especially when faced with numerous features, we may want a "big picture" of the model
- $(\mathbf{X}^T\mathbf{X})$ may be <b>not invertible</b> and thus no closed form solution

---

## Subset Selection

### Best-subset selection


- Best subset regression finds for each $k \in \{0, 1, 2, . . . , p\}$ the subset of size $k$ that gives smallest residual sum of squares.
- An efficient algorithm, the leaps and bounds procedure (Furnival and Wilson, 1974), makes this feasible for $p$ as large as 30 or 40.

<center>![best_sub](assets/img/best_sub.png "best_sub")</center>

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
- Subset Selection
- <b>Shrinkage Method</b>
    - <b>Ridge Regression</b>
        - <b>Formulations and closed form solution</b>
        - <b>Singular value decomposition</b>
        - <b>Degree of Freedom</b>
    - Lasso
- Beyond Lasso

---

## Ridge Regression

- <b>Linear regression with $l_2$-regularization</b>
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
    - The $l_2$-regularization can be viewed as a Gaussian prior on the coefficients, and our estimates are the posterior means
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
- <b>The SVD of</b> $\mathbf{X}$:
$$\mathbf{X} = \mathbf{UDV}^T$$
    - $\mathbf{U}$: $N \times p$ <b>orthogonal</b> matrix with columns spanning the column space of $\mathbf{X}$
    - $\mathbf{V}$: $p \times p$ <b>orthogonal</b> matrix with columns spanning the row space of $\mathbf{X}$  
    - $\mathbf{D}$: $p \times p$ <b>diagonal</b> matrix with diagonal entries $d_1 \geq d_2 \geq ... \geq d_p \geq 0$ being the singular values of $\mathbf{X}$
- <b>For least squares</b>
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
- <b>For ridge regression</b>
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

## Ridge Regression

### Pros

- $(\mathbf{X^TX + \lambda I})$ is inveritible and thus the cloased form solution always exist
- Ridge regression controls the complexity with regularization term via $\lambda$, which is less prone to overfitting compared with least squares fit, e.g. sometimes a wildly large coefficient on one variable can be cancelled by another wildly large coefficient of a correlated variable
- Possibly higher prediction accuracy, as the estimates of ridge regression trade a little bias for less variance, compared with least squares

### Cons

- Interpretability and compactness: Though coefficients are shrinked, but not to zero. For high dimensional problem, it may cause efficiency issue.

---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - <b>Formulations</b>
        - <b>Comparisons with ridge regression and subset selection</b>
        - <b>Quadratic Programming</b>
        - <b>Least Angle Regression</b>
        - <b>Viewed as approximation for $l_0$-regularization</b>
- Beyond Lasso

---


## LASSO

### Linear regression with $l_1$-regularization

- <b>Formulations</b>

    
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
    - The $1_1$-regularization can be viewed as a Laplace prior on the coefficients

---

## LASSO

<center>![lasso](assets/img/lasso.png "lasso")</center>



---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - Formulations
        - <b>Comparisons with ridge regression and subset selection</b>
            - <b>Orthonormal inputs</b>
            - <b>Non-orthonormal inputs</b>
        - Quadratic Programming
        - Least Angle Regression
        - Viewed as approximation for $l_0$-regularization
- Beyond Lasso


---

## LASSO

### Comparison

- <b>Orthonormal Input $\mathbf{X}$</b>
    - <b>Best subset</b>: [Hard thresholding] Only keep the top $M$ largest coefficeints of $\hat{\beta}^{ls}$
    - <b>Ridge</b>: [Pure shrinkage] does proportional shrinkage of $\hat{\beta}^{ls}$
    - <b>Lasso</b>: [Soft thresholding] translates each coefficient of $\hat{\beta}^{ls}$ by $\lambda$, truncating at 0 


<center>![comp2](assets/img/comp2.png "comp2")</center>

---

## LASSO

### Comparison

- <b>Non-orthonormal Input $\mathbf{X}$</b>

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


---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - <b>Quadratic Programming</b>
        - Least Angle Regression
        - Viewed as approximation for $l_0$-regularization
- Beyond Lasso

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

- Note that QP can only solve LASSO with a fixed $\lambda$

---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - Quadratic Programming
        - <b>Least Angle Regression</b>
        - Viewed as approximation for $l_0$-regularization
- Beyond Lasso


--- 

## Least Angle Regression

### Notations

- $\mathcal{A}_k$: <i>active set</i>, the set of features we already included in the model at time step $k$
- $\beta_{\mathcal{A}_k}$: coefficients vector at the beginning of time step $k$
- $\beta_{\mathcal{A}_k}(\alpha)$: coefficients vector in time step $k$ w.r.t. $\alpha$, 
- $\mathbf{f}_k$: the fit vector at the beginning of time step $k$, $\mathbf{f}_0 = 0$
- $\mathbf{f}_k(\alpha)$: the fit vector in time step $k$ w.r.t. $\alpha$
- $\mathbf{r}_k$: residual vector at the beginning of time step $k$, $\mathbf{r}_0 = \mathbf{y} - \bar{\mathbf{y}}$
- $\mathbf{r}_k(\alpha)$: residual vector in time step $k$, w.r.t. $\alpha$

---

## LAR Algorithm

- Initialization: 
    - Standardized all predictors s.t. $\bar{\mathbf{x}_j} = 0, \mathbf{x}_j^T\mathbf{x}_j = 1$; $\mathbf{r}_0 = \mathbf{y} - \bar{\mathbf{y}}$; $\beta = \mathbf{0}$; $\mathbf{f}_0 = \mathbf{0}$; $\mathcal{A}_k = \emptyset$
    - $\mathbf{x}_k = argmax_{\mathbf{x}_j} |\mathbf{x}_j^T \mathbf{r}_0|$, $\mathcal{A}_1 = \{\mathbf{x}_k\}$
- Main
    - While termination_cond != true
      - $\mathbf{r}_k = \mathbf{y} - \mathbf{X}_{\mathcal{A}_k} \beta_{\mathcal{A}_k}$, $\mathbf{f}_k = \mathbf{X}_{\mathcal{A}_k} \beta_{\mathcal{A}_k}$
      - Search $\alpha$
          - $\beta_{\mathcal{A}_k}(\alpha) = \mathcal{A}_k + \alpha \cdot \delta_k$, where $\delta_k = \mathbf{(X^T_{\mathcal{A}_k} X_{\mathcal{A}_k})^{-1} X^T_{\mathcal{A}_k}r_k}$
          - Concurrently, $\mathbf{f}_k(\alpha) = \mathbf{f}_k + \alpha \cdot \mathbf{u}_k$, where $\mathbf{u}_k = \mathbf{X}_{\mathcal{A}_k} \delta_k$
      - Until $|\mathbf{X}_{\mathcal{A}_k} \mathbf{r}_k(\alpha)| = max_{\mathbf{x}_j \in \bar{\mathcal{A}_k}} |\mathbf{x}_j^T \mathbf{r}_k(\alpha)|$
      - $\mathbf{x}_k = argmax_{\mathbf{x}_j \in \bar{\mathcal{A}_k}} |\mathbf{x}_j \mathbf{r}_k(\alpha)|$ 
      - $\mathcal{A}_{k+1} = \mathcal{A}_{k} \cup \{\mathbf{x}_k\}$

---

## LAR

### Comments

1. <b>Why called Least Angle</b>: the direction $\mathbf{u}_k = \mathbf{X}_{\mathcal{A}_k} \delta_k$ that our fit $\mathbf{f}_k(\alpha)$ increases actually has the same angle with any $\mathbf{x}_j \in \mathcal{A}_k$.
2. Note that the left-hand side of the termination condition for searching is a vector, while the right-hand side is a single value. 
$$|\mathbf{X}_{\mathcal{A}_k} \mathbf{r}_k(\alpha)| = max_{\mathbf{x}_j \in \bar{\mathcal{A}_k}} |\mathbf{x}_j^T \mathbf{r}_k(\alpha)|$$
This actually comes from the fact that the absolute values of correlations of $\mathbf{x}_j \in \mathcal{A}_k, \forall j$ with the residual error are tied and decrease at the same rate.
3. The procedure of searching is approaching the least-squares coefficients of fitting $\mathbf{y}$ on $\mathcal{A}_k$
4. LAR solves the subset selection problem for all $t, s.t. \|\beta\| \leq t$
5. Actually, $\alpha$ can be computed instead of searching


---

## LAR

### Example

<center>![Lars](assets/img/lars.png "lars")</center>

--- &twocolportion w1:28% w2:72%

## LAR

### Result compared with LASSO

*** left

#### Observations:

When the blue line coefficient cross zero, LAR and LASSO become different.

*** right

<center>![comp4](assets/img/comp4.png "comp4")</center>

---

## LAR

### Modification for LASSO

During the searching procedure, if a non-zero coefficient hits zero, drop this variable from $\mathcal{A}_k$, and recompute the direction $\delta_k$

### Some heuristic analysis

- At a certain time point, we know that all $\mathbf{x}_j \in \mathcal{A}$ share the same absolute values of correlations with the residual error. That is
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta) = \gamma \cdot s_j, \forall \mathbf{x}_j \ \in \mathcal{A}$$
where $s_j \in \{-1,1\}$ indicates the sign of the left hand inner product and $\gamma$ is the common value. We also know that $|\mathbf{x_j}(\mathbf{y} - \mathbf{X}\beta)| \leq \gamma, \forall \mathbf{x}_j \not\in \mathcal{A}$

---

## LAR

### Some heuristic analysis

- At a certain time point, we know that all $\mathbf{x}_j \in \mathcal{A}$ share the same absolute values of correlations with the residual error. That is
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta) = \gamma \cdot s_j, \forall \mathbf{x}_j \ \in \mathcal{A}$$
where $s_j \in \{-1,1\}$ indicates the sign of the left hand inner product and $\gamma$ is the common value. We also know that $|\mathbf{x_j}(\mathbf{y} - \mathbf{X}\beta)| \leq \gamma, \forall \mathbf{x}_j \not\in \mathcal{A}$

- Now consider about LASSO for a fixed given $\lambda$. Let $\mathcal{B}$ with non-zero coefficients, then we differentiate the objective function w.r.t. these non zero coefficients and set the gradient to zero
$$\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta) = \lambda \cdot sign(\beta_j), \forall j \in \mathcal{B}$$

- They are identical only if $sign(\beta_j)$ matches the sign of the lefthand side. In $\mathcal{A}$, we allow for the $\beta_j$, where $sign(\beta_j) \neq sign(\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta))$, while this is forbidden in $\mathcal{B}$. Thus, once a coefficent hits zero, we drop it.


---

## LAR

### Some heuristic analysis

-  For LAR, we have 
$$|\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta)| \leq \gamma, \forall \mathbf{x}_j \not\in \mathcal{A}$$
- According to the stationary conditions, for LASSO, we have
$$
|\mathbf{x}_j^T(\mathbf{y} - \mathbf{X}\beta)| \leq \lambda, \forall \mathbf{x}_j \not\in \mathcal{B}
$$
- They match for variables with zero coefficients too.



---

## LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- <b>Shrinkage Method</b>
    - Ridge Regression
    - <b>Lasso</b>
        - Formulations
        - Comparisons with ridge regression and subset selection
        - Quadratic Programming
        - Least Angle Regression
        - <b>Viewed as approximation for $l_0$-regularization</b>
- Beyond Lasso

---

## Viewed as approximation for $l_0$-regularization

### Pure variable selection

$$
\begin{equation}
\hat{\beta}^{ridge}= argmin_{\beta}\sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^p\mathbf{x_{ij}}\beta_j)^2, s.t. \#nonzero \beta_j \leq t
\end{equation}
$$

Actually $\#nonzero \beta_j = \|\beta\|_0$, where

$$\|\beta\|_0 = lim_{q \to 0}(\sum_{j = 1}^p|\beta_j|^q)^{\frac{1}{q}} = card(\{\beta_j|\beta_j \neq 0\})$$

<center>![zeronorm](assets/img/zeronorm.png "zeronorm")</center>

---

## Viewed as approximation for $l_0$-regularization

### Problem

$l_0$-norm is not convex, which makes it very hard to optimize.

### Solutions

- <b>LASSO</b>: Approximated objective function ($l_1$-norm), with exact optimization
- <b>Subset selection</b>: Exact objective function, with approximated optimization (greedy strategy)

---

## Beyond LASSO

- Introduction to Dimension Reduction
- Linear Regression and Least Squares (Review)
- Subset Selection
- Shrinkage Method
- <b>Beyond LASSO</b>
   - <b>Elastic-Net</b>
   - <b>Fused Lasso</b>
   - <b>Group Lasso</b>   
   - <b>$l_1-lp$ norm</b>
   - <b>Graph-guided Lasso</b>


--- &triple w1:41% w2:55% 

## Beyond LASSO

### Elastic Net

*** left

- <b>Formualtion</b>
$$\lambda \sum_{j = 1}^p (\alpha \beta_j^2 + (1-\alpha)|\beta_j|)$$
which is a compromise between ridge regression and LASSO.

*** right

<center>![enet](assets/img/enet.png "enet")</center>

*** down

- <b>Advantages</b>
    - The elastic-net selects variables like the lasso, and shrinks together the coefficients of correlated predictors like ridge.
    - It also has considerable computational advantages over the $l_q$ penalties. (detailed in Section 18.4 in Elements of Statistical Learning)



---

## Beyond LASSO

### Fused Lasso

- <b>Intuition</b>
    - Fused lasso is a generalization that is designed for problems with features that can be ordered in some meaningful way. 
    - The fused lasso penalizes the $L_1$-norm of both the coefﬁcients and their successive differences.
- <b>Formulation</b>

$$\hat{\beta} = argmin_{\beta}\{\|\mathbf{X\beta - y}\|_2^2\}$$
$$s.t. \|\beta\| \leq s_1 and \sum_{j = 2}^p |\beta_j - \beta_{j-1}| \leq s_2$$

---

## Beyond LASSO

### Fused Lasso

<center>![nb_fs](assets/img/nb_fs.png "nb_fs")</center>


---

## Beyond LASSO

### Group Lasso

- <b>Intuition</b>
    - Features are divided into $L$ groups
    - Features within the same group should share similar coefficients
- <b>Example</b>
    - Binary dummy variables from one single discrete variable, e.g. $stage\_cancer \in \{1,2,3\}$ can be translated into three binary dummy variables $(stage1, stage2, stage3)$ 
- <b>Formulations</b>
$$obj = \|\mathbf{y} - \sum_{l = 1}^L \mathbf{X}_l \beta_l \|_2^2 + \lambda_1 \sum_{l = 1}^L\|\beta_l\|_2 + \lambda_2 \|\beta\|_1$$



---

## Beyond LASSO

### $l_1$-$l_p$ penalization

- <b>Applies to multi-task learning</b>, where the goal is to estimate predictive models for several related tasks. 
- <b>Examples</b>
    - <b>Example 1</b>: recognize speech of different speakers, or handwriting of different writers, 
    - <b>Example 2</b>: learn to control a robot for grasping different objects or drive in different landscapes, etc. 
- <b>Assumptions about the tasks</b>
    - sufficiently <i>different</i> that learning a specific model for each task results in improved performance
    - <i>similar</i> enough that they share some common underlying representation that should make simul- taneous learning beneficial. 
    - In particular, we focus on the scenario where the different tasks share a subset of relevant features to be selected from a large common space of features.

---

## Beyond LASSO

### $l_1$-$l_p$ penalization

- Formulation
    - $\mathbf{X}_l$: $N \times p$ input matrix for task $l$ and $L$ is the total number of tasks
    - $\beta$: $p \times L$ coefficient matrix
    - $\mathbf{y}$: $N \times L$ output matrix
    - objective function
        $$obj = \sum_{l= 1}^L loss(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \sum_{j = 1}^p \|\beta_{:j}\|_2$$
      where $loss()$ is some loss function and $\sum_{j = 1}^p \|\beta_{:j}\|_2$ is equavalent to get the $l_1$ norm of vector $(\|\beta_{:1}\|_2, \|\beta_{:2}\|_2, ..., \|\beta_{:p}\|_2)$.
    

---

## Beyond LASSO

### $l_1-l_p$ penalization -Coefficient matrix

<center>![l1lp](assets/img/l1lp.png "l1lp")</center>

---

## Beyond LASSO

### $l_1-l_p$ penalization -Norm ball

<center>![normball](assets/img/normball.png "normball")</center>

---

## Beyond LASSO

### Graph-Guided LASSO

- <b>Example</b>
<center>![gflasso](assets/img/gflasso.png "gflasso")</center>
- <b>Formulation</b>
Graph-Guided Lasso applies to multi-task settings
$$obj = \sum_{l= 1}^L loss(\beta_{:l}, \mathbf{X}_l, \mathbf{y}_{:l}) + \lambda \|\beta\|_1+\gamma \sum_{e=(a,b)\in E}^p \tau(r_{ab}) \sum_{j = 1}^p |\beta_{ja} - sign(r_{a,b})\beta_{jb}|$$
where $r_{a,b} \in \mathbb{R}$ denotes the weight of the edge and $\tau(r)$ can be any positive monotonically increasing function of $|r|$, e.g. $\tau(r) = |r|$.

---

## Beyond LASSO

### Graph-Guided LASSO

<center>![gflasso_re](assets/img/gflasso_re.png "gflasso_re")</center>


--- &vcenter  

## Sparse Models

### Thank You!

