<!--
Title: Recent Advances in Stochastic Methods
Date: 3000-01-01 00:00
Category: optimization
Tags: optimization, stochastic
Slug: recent-advances-in-stochastic-methods
-->


Outline
=======
- Introduction: Explain the GD/SGD tradeoff. State that Incremental Methods marry the best of both.
- Background: Precisely define (regularized) empirical risk minimization, GD/SGD, and convergence rates.
- Incremental Methods: Give framework and intuition for incremental methods.
  - Compare updates for as many as you can (SAG, SVRG/S2GD, SAGA, SDCA, Finito/MISO)
  - Give assumptions necessary for each. Some need L2 reg, some don't.
- Extensions: Talk about Proximal, Accelerated variants.
- Future Work: What questions are still open?

-->

# Introduction

Stochastic Gradient Descent (SGD) is flawed. Originally developed [in 1951][robbins-1951] for finding roots in polynomials, SGD stands as the go-to method for large scale machine learning. Unlike its batch cousin, Gradient Descent (GD), Stochastic Gradient Descent is capable of making quick progress in only one or two passes through the data set.

But it is after this point that SGD's weakness makes itself known. After several passes, the algorithm's progress slows to a crawl. There's a reason large scale systems such as [Vowpal Wabbit][vowpwal-wabbit] only use SGD in their initial phase.

But alas, all is not lost. In 2012, a new class of _Incremental Methods_ were introduced, marrying Stochastic and batch Gradient Descent. Like Stochastic Gradient Descent, Incremental Methods make quick progress with a handful of passes through the data set. Like Gradient Descent, their progress continues long afterwards.

What changed? How, after [thirty years][nemirovski-1983], have researchers finally been able to overcome the limitations of SGD? In short, by taking a new perspective on an old problem. Now let's make this a little more precise...

# Background

Gradient Descent and Stochastic Gradient Descent are two of the most common optimization algorithms applied in machine learning. Quite often, these problems take on the form of regularized [empirical risk minimization][empirical-risk-minimization],

$$
\begin{align*}
w^{*}
&= \underset{w}{\text{argmin}} \; L(w) \\
&= \underset{w}{\text{argmin}} \; \left[
    \frac{1}{m} \sum_{i=1}^{m} \ell_{i}(w^T x_i)
    + \frac{\lambda}{2} || w ||_2^2
  \right]
\end{align*}
$$

Gradient Descent minimizes the objective $L(w)$ by repeatedly making use of its _gradient_

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Compute the gradient of $L$ at $w^{(t)}$, $g^{(t)} \triangleq \nabla L(w^{(t)})$
    4. $w^{(t+1)} = w^{(t)} - \eta^{(t)} g^{(t)}$
</div>

Similarly, Stochastic Gradient Descent repeatedly makes of a _random approximation_ to $L$'s gradient,

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Compute an approximation to the gradient of $L$ at $w^{(t)}$, $\hat{g}^{(t)} \triangleq \hat{\nabla} L(w^{(t)})$
    4. $w^{(t+1)} = w^{(t)} - \eta^{(t)} g^{(t)}$
</div>

If the approximation is unbiased -- that is, its expectation is the true gradient -- then Stochastic Gradient Descent converges to $w^{*}$ much the same way Gradient Descent does. A common choice for $\hat{\nabla} L(w)$ is $\nabla \ell_j(w^T x_j)$ for an example $j$ picked uniformly at random.

The gulf between these two methods arises not in what they converge to but how quickly they get there. Whereas Gradient Descent will reach $\epsilon$ accuracy within $O(\log 1/\epsilon)$ iterations, Stochastic Gradient Descent requires $O(1 / \epsilon)$. The cause? Noise.

Unlike Gradient Descent, Stochastic Gradient Descent's random approximation doesn't approach zero as $w^{(t)}$ approaches $w^{*}$. To keep the iterates from "bouncing around", a decreasing step size is required. Contrast this with Gradient Descent, for which a small, constant step size suffices. It has even been shown that under this regime, [no method can hope to converge faster][agarwal-2012].

# Incremental Methods

Incremental Methods combine the fast convergence of Gradient Descent with the low computational overhead of Stochastic Gradient Descent. Like Gradient Descent, they admit a constant step size; like Stochastic Gradient Descent, the gradient for a single randomly chosen example suffices for each iteration.

How do Incremental Methods achieve this? By recognizing that $L(w)$ isn't just any function -- it's a _finite sum_. By retaining example-specific information and updating only as necessary, Incremental Methods recover Gradient Descent's constant step size while maintaining Stochastic Gradient Descent's low computational cost.

Most incremental methods fall into the following structure,

<div class="pseudocode" markdown>
  **Input**: initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Select an example $i$ uniformly from $\{ 1 \ldots m \}$
    3. Set $\phi_i^{(t)} = w^{(t)}$ for example $i$. Retain $\phi_j^{(t)} = \phi_j^{(t-1)}$ for all $j \neq i$.
    4. Update $w^{(t+1)}$ based on $w^{(t)}$ and $\{ \phi_i^{(t)} \}_{i=1}^{t}$
</div>


Below, we can see the updates for a variety of incremental methods as presented by [Aaron Defanzio][defanzio-2014-2].

The first three, [Stochastic Average Gradient (SAG)][schmidt-2013], [SAGA][defanzio-2014-2], and [Stochastic Variance Reduced Gradient (SVRG)][johnson-2013] / [Semi-Stochastic Gradient Descent (S2GD)][konecny-2013], are similar to Stochastic Gradient Descent. SAG and SAGA differ only in the weight given to the new gradient contribution; SVRG / S2GD is similar to SAGA, except that a global $\phi^{(t)}$, rather than local $\{ \phi_i^{(t)} \}_{i=1}^{m}$, that is updated once every few iterations.

Rather than operating directly on the latest weight vector, [Finito][defanzio-2014] / [MISO][mairal-2013] and [Stochastic Dual Coordinate Ascent][shalev-schwartz-2012] operate on the _average_ of stale weights. The former is a descent algorithm similar to Gradient Descent; were all examples updated simultaneously, we would recover Gradient Descent. The latter, you may notice, defines $w^{(t+1)}$ in terms of $\phi_i^{(t+1)}$, which is in turn defined as a function of $w^{(t+1)}$ -- a circular dependency. In implementation, we may obtain $\phi_i^{(t+1)}$ via a certain proximal update ([Appendix A][defanzio-2014-2]). The operation is reminiscent of the [Forward-Backward algorithm][duchi-2009].

$$
\begin{align*}
\text{(SAG)}
  && w^{(t+1)} &= w^{(t)} - \eta \left[
    \frac{\nabla \ell_i (w^{(t)}) - \nabla \ell_i (\phi_i^{(t)})}{m} + \frac{1}{m} \sum_{j=1}^{m} \nabla \ell_i (\phi_i^{(t)})
  \right] \\
\text{(SAGA)}
  && w^{(t+1)} &= w^{(t)} - \eta \left[
    \nabla \ell_i (w^{(t)}) - \nabla \ell_i (\phi_i^{(t)}) + \frac{1}{m} \sum_{j=1}^{m} \nabla \ell_i (\phi_i^{(t)})
  \right] \\
\text{(SVRG/S2GD)}
  && w^{(t+1)} &= w^{(t)} - \eta \left[
    \nabla \ell_i (w^{(t)}) - \nabla \ell_i (\phi^{(t)}) + \frac{1}{m} \sum_{j=1}^{m} \nabla \ell_i (\phi^{(t)})
  \right] \\
\text{(Finito/MISO)}
  && w^{(t+1)} &= \frac{1}{m} \sum_{i=1}^{m} \phi_i^{(t)} - \left[
    \frac{1}{\lambda m} \sum_{j=1}^{m} \nabla \ell_i (\phi^{(t)})
  \right] \\
\text{(SDCA)}
  && w^{(t+1)} &= \frac{1}{m} \sum_{j=1}^{m} \phi_j^{(t+1)} - \left[
    \frac{1}{\lambda m} \sum_{j=1}^{m} \nabla \ell_j (\phi^{(t+1)})
  \right] \\
\end{align*}
$$

The tools used to prove convergence for these algorithms are more difficult to compare. SAG and SAGA depend on Lyapunov functions, SDCA via the dual objective, and SVRG via a variance-reduction argument.


# Extensions

# Future Work

<!-- References -->
[agarwal-2012]: http://www.eecs.berkeley.edu/~wainwrig/Papers/AgaEtAl12.pdf
[blatt-2007]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.4020&rep=rep1&type=pdf
[bottou-2011]: http://leon.bottou.org/publications/pdf/mloptbook-2011.pdf
[cvxbook]: http://stanford.edu/~boyd/cvxbook/
[defanzio-2014-2]: http://arxiv.org/abs/1407.0202
[defanzio-2014]: http://arxiv.org/abs/1407.2710
[empirical-risk-minimization]: http://en.m.wikipedia.org/wiki/Empirical_risk_minimization#Empirical_risk_minimization
[hsieh-2008]: http://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf
[le-roux-2012]: http://arxiv.org/abs/1202.6258
[nemirovski-1983]: http://www.palgrave-journals.com/jors/journal/v35/n5/abs/jors198492a.html
[robbins-1951]: http://projecteuclid.org/euclid.aoms/1177729586
[schmidt-2013]: http://arxiv.org/abs/1309.2388
[shalev-schwartz-2012]: http://arxiv.org/abs/1209.1873
[tsitsikilis-bertsekas-1986]: http://www.mit.edu/~jnt/Papers/J014-86-asyn-grad.pdf
[vowpwal-wabbit]: https://github.com/JohnLangford/vowpal_wabbit
[johnson-2013]: http://stat.rutgers.edu/home/tzhang/papers/nips13-svrg.pdf
[konecny-2013]: http://arxiv.org/abs/1312.1666
[mairal-2013]: http://arxiv.org/pdf/1305.3120.pdf
[duchi-2009]: http://jmlr.csail.mit.edu/papers/volume10/duchi09a/duchi09a.pdf
