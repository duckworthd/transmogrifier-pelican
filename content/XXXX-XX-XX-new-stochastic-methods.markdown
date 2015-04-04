Title: Recent Advances in Stochastic Methods
Date: 3000-01-01 00:00
Category: optimization
Tags: optimization, stochastic
Slug: recent-advances-in-stochastic-methods

In the beginning, there was Gradient Descent (GD), and lo, the optimization gods
looked upon it, and saw that it was good. But soon came the days of supervised
learning and [empirical risk minimization][empirical-risk-minimization], where
even evaluating the objective function required computation _linear_ in the
number of training examples. Thus Stochastic Gradient Descent (SGD) was born, a
computationally advantageous alternative to Gradient Descent with computation
_constant_ instead.

And lo, this method too was good. In the data-rich setting where generalization
error (not optimization error!) is the goal it can even be
[vastly superior][bottou-2011] to its older sibling, Gradient Descent.

Yet something was amiss. In practice, SGD obtained "good" solutions very quickly
but struggled to reach high accuracy ones. One of the most scalable open source
systems for empirical risk minimization, [Vowpal Wabbit][vowpwal-wabbit], limits
the use of SGD to its initial phase precisely for this reason. Theory, too,
verified what was observed. For strongly convex objectives, SGD requires
$O(1/\epsilon)$ iterations to achieve an $\epsilon$-accurate solution; Gradient
Descent, on the other hand, only requires $O(\log 1 / \epsilon)$.

For many years, this rift remained. Gradient Descent, the heavy weight,
theoretically quick ideal; Stochastic Gradient Descent, its nimble but
theoretically crippled counterpart. Was there not some way to marry the two? To
obtain the benefits of both?

In 2012, all that changed. Two methods,
[Stochastic Dual Coordinate Ascent (SDCA)][shalev-schwartz-2012] and
[Stochastic Average Gradient (SAG)][schmidt-2013], showed that we can have our
cake and eat it too. Since then, "Incremental Optimization" has matured into a
full-fledged research topic of its own, resulting in tens of methods with
different assumptions, guarantees, and variations.

In the following, I'll  summarize these methods and their successors. What do
they assume? What do they guarantee? Why do they work? *TODO: Add conclusion
sentence*

# A New Perspective, A New Convergence Rate

[Empirical risk minimization][empirical-risk-minimization] is the theoretical
framework for supervised machine learning. Given $m$ input-output pairs
$\{ (x_i, y_i) \}_{i=1}^{m}$ sampled from a distributed $P(x,y)$, the goal is
to find a set of weights $w$ such that we have low average "risk",

$$
\begin{align*}
w^{*}
  &= \underset{w}{\text{argmin}} \; L(w) \\
  &= \underset{w}{\text{argmin}} \;\frac{1}{m} \sum_{i=1}^{m} \ell(w^T x_i, y_i)
\end{align*}
$$

Traditional Gradient Descent minimizes $L(w)$ by assuming access to a "first
order oracle" $\nabla L(w)$, a function that, given an input $w$, produces the
gradient of $L$ at $w$. The ensuing algorithm is then,

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Compute the gradient of $L$ at $w^{(t)}$,
      $g^{(t)} \triangleq \nabla L(w^{(t)})$
    3. $w^{(t+1)} = w^{(t)} - \eta^{(t)} g^{(t)}$
</div>

<!--
To prove the convergence of $w^{(t)}$ to $w^{*}$, There are a handful of common
assumptions made which can be grouped into two regimes -- each resulting in its
own upper bound on the rate of convergence.

The first regime assumes that $\ell(z,y)$ be differentiable in $z$ and that
$\ell'(z,y)$ is $L$-Lipschitz. For twice differentiable $\ell(z,y)$, this is
equivalent to assuming the second derivative is upper bounded by a constant.
As a result, it can be shown that an $L(w^{(t)})$ is within $\epsilon$ of
$L(w^{*})$ after $O(1 / \epsilon)$ iterations.

The second regime further assumes that $\ell(z,y)$ is also strongly convex.
Equivalently, this assumption asks that the second derivative of $\ell(z,y)$ be
_lower bounded_ by a constant. This is most commonly the case for
$L_2$-regularized empirical risk minimization problems. Under this regime,
Gradient Descent guarantees the same $\epsilon$-accurate solution in $O( \log 1
/ \epsilon )$ iterations.
-->

Stochastic Gradient Descent instead assumes access to a _random_ first order
oracle $\hat{\nabla} L(w)$ that produces a _random approximation_ to the
gradient at $w$. So long as the expectation of this approximation matches the
true gradient -- $\mathbb{E}[\hat{\nabla} L(w)] = \nabla L(w)$ -- using
$\hat{\nabla} L(w)$ in place of $\nabla L(w)$ -- results in a similarly
intuitive algorithm,

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Compute an approximation to the gradient of $L$ at $w^{(t)}$,
      $\hat{g}^{(t)} \triangleq \hat{\nabla} L(w^{(t)})$
    3. $w^{(t+1)} = w^{(t)} - \eta^{(t)} \hat{g}^{(t)}$
</div>

For empirical risk minimization, an approximation that satisfies this property
is $\nabla \ell_i(w^T x_i, y_i)$ for an example $i$ selected uniformly at
random.

In practice and theory, Gradient Descent and Stochastic Gradient Descent are
almost identical. Where they diverge is in their choice of learning rate
$\eta^{(t)}$. For GD, a small, _constant_ step size suffices, but in SGD,
a learning rate that _tends to zero_ as $t \rightarrow \infty$ is crucial. The
variance introduced by randomization needs to be controlled, and, unlike GD's
first order oracle, SGD's does not approach zero as $w^{(t)}$ approaches
$w^{*}$.

The consequences of variance are harsh. For convex $L(w)$, Gradient Descent can
guarantee $L(w^{(t)}) - L(w^{*}) \leq \epsilon$ in $O(1/\epsilon)$ iterations,
Stochastic Gradient Descent needs $O(1 / \epsilon^2)$. For strongly convex
$L(w)$, Gradient Descent needs only $O(\log 1 / \epsilon)$ iterations; SGD, $O(1
/ \epsilon)$. [Agarwal et al][agarwal-2012] have further shown that _no method
based on random first order oracles can hope to do better_.

How then do methods like SDCA and SAG circumvent SGD's weakness? By recognizing
that in empirical risk minimization, $L(w)$ isn't just any function, it's
a _finite sum_. When the random first order oracle selects an index $i$, these
methods take knowledge of $i$ into account. By maintaining per-index
information, _Incremental Methods_ are able to combine a "global" view of $L$
while operating "locally". Precisely what that information is maintained and
how it's used differentiates these new methods.

# Stabilizing Random Gradient Approximations

*TODO: Summarize SVRG too*

[Stochastic Average Gradient (SAG)][schmidt-2013] was the first incremental
optimization algorithm to guarantee Gradient Descent-like convergence rates.
The algorithm itself is identical to another method, [Incremental Average
Gradient (IAG)][blatt-2007], except with coordinates chosen randomly instead of
cyclically. The novelty of SAG was its theory rather than implementation.

The method can be applied to all empirical risk minimization problems of the
following form,

$$
\min_{w} P(w) =
  \frac{1}{m} \sum_{i=1}^{m} \ell_i(w^T x_i)
$$

It is assumed that $\ell_i$ is differentiable and its derivative is
$L$-Lipschitz continuous, $||\ell_{i}^{\prime}(z_1) - \ell_{i}^{\prime}(z_2)||
\leq L ||z_1 - z_2||$. This disqualifies non-smooth losses such as SVM's hinge
loss and non-smooth regularizers such as $L_1$ penalization but admits other
common regimes such as $L_2$-penalized logistic and linear regression.

In practice, the SAG algorithm is similar to Gradient Descent. Whereas we may
interpret GD as recalculating the gradient contribution for every example with
each iteration, SAG updates only one example's gradient contribution per
iteration and reuses the "stale" contributions of all others,

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Select an example $i$ uniformly at random from $\{1 \ldots m \}$
    3. Update $g_i^{(t)} = \nabla \ell_i(x_i^T w^{(t)})$ for selected example
       $i$. For all other examples $i'$, re-use $g_{i'}^{(t)} = g_{i'}^{(t-1)}$.
    4. Set $w^{(t+1)} = w^{(t)} - \frac{\eta^{(t)}}{m} \sum_{i=1}^{m} g_i^{(t)}$
</div>

This small change permits SAG to use a constant, rather than decreasing, step
size, resulting in $O(1/\epsilon)$ convergence for $L$-Lipschitz continuous
losses and $O(\log 1 / \epsilon)$ convergence for strongly convex. If the
algorithm maintains a running sum for $\sum_{i=1}^{m} g_i^{(t)}$, updating one
entry can be efficiently implemented as performing one vector addition and
subtraction.

**Benefits**

- Gradient Descent-like convergence rates
- Per-iteration computational cost is constant with respect to the number of
  examples
- Able to handle handle objectives without $L_2$ regularization

**Drawbacks**

- Unable to handle non-differentiable loss functions
- Unable to handle $L_1$ regularization
- Storage scales linearly with the number of examples


# Solve the Dual, Solve the Primal

One of the two initial algorithms to kick off Incremental Optimization,
[Stochastic Dual Coordinate Ascent (SDCA)][shalev-schwartz-2012], was
not a novel algorithm in and of itself at time of publication. Hsieh et al. had
[already established][hsieh-2008] that Dual Coordinate Ascent obtains linear
convergence when applied to $L_2$-regularized Support Vector Machines. What was
novel, however, was showing that the same result holds for all $L_2$
regularized empirical risk minimization problems.

We begin by assuming the (primal) objective has the following form,

$$
\min_{w} P(w) =
  \frac{1}{m} \sum_{i=1}^{m} \ell_i(w^T x_i)
  + \frac{\lambda}{2} \left| \left| w \right| \right|_2^2
$$

Rather than minimizing the primal directly, SDCA maximizes the dual,

$$
\max_{\alpha} D(\alpha) =
  - \frac{1}{m} \sum_{i=1}^{m} \ell_i^{*}(\alpha_i)
  - \frac{\lambda}{2} \left| \left| \frac{1}{\lambda m} \sum_{i=1}^{m} \alpha_i x_i \right| \right|_2^2
$$

Due to problem structure, we can easily recover the primal variables from the
dual via the following equality,

$$
  w(\alpha) = \frac{1}{\lambda m} \sum_{i=1}^{m} \alpha_i x_i
$$

Notice that each example corresponds to a single dual variable. Coordinate
Ascent in the dual can thus be understood as picking a single example, then
updating the dual variable corresponding to it. Since dual coordinates interact
only via the quadratic on the right, this single-coordinate optimization can be
performed without visiting every other example,

$$
\max_{\alpha_i} D(\alpha) =
  - \frac{1}{m} \ell_i^{*}(\alpha_i)
  - \frac{\lambda}{2} \left| \left| \frac{1}{\lambda m} \alpha_i x_i \right| \right|_2^2
  - \lambda \left< \frac{1}{\lambda m} \alpha_i x_i, \frac{1}{\lambda m} \sum_{i' \neq i} \alpha_{i'} x_{i'} \right>
$$

When examples are selected uniformly at random, we obtain Stochastic Dual
Coordinate Ascent,

<div class="pseudocode" markdown>
  **Input**: initial iterate $\alpha^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Select a coordinate $i$ uniformly at random from $\{1 \ldots m \}$
    3. Choose $\alpha_i^{(t+1)}$ such that it maximizes $D(\alpha)$ while
       holding all other $\alpha_j = \alpha_j^{(t)}$ fixed for all $j \neq i$.
    4. Set $\alpha_j^{(t+1)} = \alpha_j^{(t)}$ for all $j \neq i$
</div>

If $\ell_i$ is $L$-Lipschitz for all $i$, then the number of iterations to
reach a duality gap of $\epsilon$ is $O(1/\epsilon)$; if $\ell_i$ is also
differentiable and its derivative is $1 / \gamma$-Lipschitz, then $O(\log
1 / \epsilon)$. Note that this matches the guarantees of batch Gradient
Descent.

**Benefits**

- Gradient Descent-like convergence rates
- Per-iteration computational cost is constant with respect to the number of
  examples

**Drawbacks**

- Unable to handle  $L_1$ regularization
- Unable to handle objectives without $L_2$ regularization
- Requires analytic derivation of dual objective and efficient dual coordinate
  update
- Storage scales linearly with the number of examples

# Extensions

*TODO: Non-uniform sampling, Reducing memory usage, weaker assumptions, Mini-batch*

# The Family Tree

[empirical-risk-minimization]: http://en.m.wikipedia.org/wiki/Empirical_risk_minimization#Empirical_risk_minimization
[bottou-2011]: http://leon.bottou.org/publications/pdf/mloptbook-2011.pdf
[vowpwal-wabbit]: https://github.com/JohnLangford/vowpal_wabbit
[shalev-schwartz-2012]: http://arxiv.org/abs/1209.1873
[le-roux-2012]: http://arxiv.org/abs/1202.6258
[schmidt-2013]: http://arxiv.org/abs/1309.2388
[tsitsikilis-bertsekas-1986]: http://www.mit.edu/~jnt/Papers/J014-86-asyn-grad.pdf
[hsieh-2008]: http://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf
[Agarwal-2012]: http://www.eecs.berkeley.edu/~wainwrig/Papers/AgaEtAl12.pdf
[blatt-2007]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.4020&rep=rep1&type=pdf
[cvxbook]: http://stanford.edu/~boyd/cvxbook/
