Title: The Big Table of Convergence Rates
Date: 2014-08-01 00:00
Category: optimization
Tags: optimization, convergence, reference
Slug: big-table-of-convergence-rates

  In the past 50+ years of convex optimization research, a great many
algorithms have been developed, each with slight nuances to their assumptions,
implementations, and guarantees. In this article, I'll give a shorthand
comparison of these methods in terms of the number of iterations required
to reach a desired accuracy $\epsilon$ for convex and strongly convex objective
functions.

  Below, methods are grouped according to what "order" of information they
require about the objective function. In general, the more information you
have, the faster you can converge; but beware, you will also need more memory
and computation. Zeroth and first order methods are typically appropriate for
large scale problems, whereas second order methods are limited to
small-to-medium scale problems that require a high degree of precision.

  At the bottom, you will find algorithms aimed specifically at minimizing
supervised learning problems and other meta-algorithms useful for distributing
computation across multiple nodes.

  Unless otherwise stated, all objectives are assumed to be Lipschitz
continuous (though not necssarily differentiable) and the domain convex. The
variable being optimized is $x \in \mathbb{R}^n$.

Zeroth Order Methods
====================

  Zeroth order methods are characterized by not requiring any gradients or
subgradients for their objective functions. In exchange, however, it is
assumed that the objective is "simple" in the sense that a subset of variables
(a "block") can be minimized exactly while holding all other variables fixed.

<table markdown class="table table-bordered table-centered">
  <colgroup>
    <col style="width:20%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:40%">
  </colgroup>
  <thead>
    <tr>
      <th>Algorithm          </th>
      <th>Problem Formulation</th>
      <th>Convex             </th>
      <th>Strongly Convex    </th>
      <th>Per-Iteration Cost </th>
      <th>Notes              </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <!-- Algorithm          -->
      <td>Randomized Block Coordinate Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^{n}} f(x) + g(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon)$[^richtarik-2011]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1 / \epsilon))$[^richtarik-2011]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(1)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable and $g(x)$ is separable in
        each block. $g(x)$ may be a barrier function.
      </td>
    </tr>
  </tbody>
</table>

First Order Methods
===================

  First order methods typically require access to an objective function's
gradient or subgradient. The algorithms typically take the form $x^{(t+1)}
= x^{(t)} - \alpha^{(t)} g^{(t)}$ for some step size $\alpha^{(t)}$ and descent
direction $g^{(t)}$. As such, each iteration takes approximately $O(n)$ time.

<table markdown class="table table-bordered table-centered">
  <colgroup>
    <col style="width:20%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:40%">
  </colgroup>

  <thead>
    <tr>
      <th>Algorithm          </th>
      <th>Problem Formulation</th>
      <th>Convex             </th>
      <th>Strongly Convex    </th>
      <th>Per-Iteration Cost </th>
      <th>Notes              </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <!-- Algorithm          -->
      <td>Subgradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle  \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon^{2})$[^blog-sd]</td>
      <!-- Strongly Convex    -->
      <td>...</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Cannot be improved upon without further assumptions.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Mirror Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathcal{C}} f(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon^{2} )$[^ee381-md]</td>
      <!-- Strongly Convex    -->
      <td>...</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Different parameterizations result in gradient descent and
        exponentiated gradient descent.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Gradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon)$[^blog-gd]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1 / \epsilon))$[^ee381-gd]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Accelerated Gradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \sqrt{\epsilon})$[^blog-agd]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1 / \epsilon))$[^bubeck-agd]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable.
        Cannot be improved upon without further assumptions.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Proximal Gradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathcal{C}} f(x) + g(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon)$[^blog-pgd]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1 / \epsilon))$[^mairal-2013]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable and
        $\text{prox}_{\tau_t g}(x)$ is easily computable.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Proximal Accelerated Gradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathcal{C}} f(x) + g(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \sqrt{\epsilon})$[^blog-apgd]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1 / \epsilon))$[^mairal-2013]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable and
        $\text{prox}_{\tau_t g}(x)$ is easily computable.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Frank-Wolfe Algorithm / Conditional Gradient Algorithm</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathcal{C}} f(x)$</td>
      <!-- Convex             -->
      <td>$O(1/\epsilon)$[^blog-fw]</td>
      <!-- Strongly Convex    -->
      <td>$O(1/\sqrt{\epsilon})$[^garber-2014]</td>
      <!-- Per-Iteration Cost -->
      <td>...</td>
      <!-- Notes              -->
      <td>
        Applicable when $\mathcal{C}$ is bounded. Most useful when
        $\mathcal{C}$ is a polytope in a very high dimensional space with
        sparse extrema.
      </td>
    </tr>
  </tbody>
</table>

Second Order Methods
====================

  Second order methods either use or approximate the hessian ($\nabla^2 f(x)$)
of the objective function to result in better-than-linear rates of convergence.
As such, each iteration typically requires $O(n^2)$ memory and between $O(n^2)$
and $O(n^3)$ computation per iteration.

<table markdown class="table table-bordered table-centered">
  <colgroup>
    <col style="width:20%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:40%">
  </colgroup>

  <thead>
    <tr>
      <th>Algorithm          </th>
      <th>Problem Formulation</th>
      <th>Convex             </th>
      <th>Strongly Convex    </th>
      <th>Per-Iteration Cost </th>
      <th>Notes              </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <!-- Algorithm          -->
      <td>Newton's Method</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>...</td>
      <!-- Strongly Convex    -->
      <td>$O(\log \log (1/\epsilon))$[^ee364a-unconstrained]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n^3)$</td>
      <!-- Notes              -->
      <td>
        Only applicable when $f(x)$ is twice differentiable. Constraints can be
        incorporated via interior point methods.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Conjugate Gradient Descent</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>...</td>
      <!-- Strongly Convex    -->
      <td>$O(n)$</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n^2)$</td>
      <!-- Notes              -->
      <td>
        Converges in exactly $n$ steps for quadratic $f(x)$. May fail to
        converge for non-quadratic objectives.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>L-BFGS</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} f(x)$</td>
      <!-- Convex             -->
      <td>...</td>
      <!-- Strongly Convex    -->
      <td>Between $O(\log (1/\epsilon))$ and $O(\log \log (1/\epsilon))$[^ee236c-qnewton]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n^2)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f(x)$ is differentiable, but works best when twice
        differentiable. Convergence rate is not guaranteed.
      </td>
    </tr>
  </tbody>
</table>

Stochastic Methods
==================

  The following algorithms are specifically designed for supervised machine
learning where the objective can be decomposed into independent "loss"
functions and a regularizer,

$$
\begin{align*}
  \min_{x} \frac{1}{N} \sum_{i=1}^{N} f_{i}(x) + \lambda g(x)
\end{align*}
$$

  The intuition is that finding the optimal solution to this problem is
unnecessary as the goal is to minimize the "risk" (read: error) with respect to
a set of _samples_ from the true distribution of potential loss functions.
Thus, the following algorithms' convergence rates are for the _expected_ rate
of convergence (as opposed to the above algorithms which upper bound the _true_
rate of convergence).

<table markdown class="table table-bordered table-centered">
  <colgroup>
    <col style="width:20%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:40%">
  </colgroup>

  <thead>
    <tr>
      <th>Algorithm          </th>
      <th>Problem Formulation</th>
      <th>Convex             </th>
      <th>Strongly Convex    </th>
      <th>Per-Iteration Cost </th>
      <th>Notes              </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <!-- Algorithm          -->
      <td>Stochastic Gradient Descent (SGD)</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} \sum_{i} f_{i}(x) + \lambda g(x)$</td>
      <!-- Convex             -->
      <td>$O(n/\epsilon^2)$[^bach-2012]</td>
      <!-- Strongly Convex    -->
      <td>$O(n/\epsilon)$[^bach-2012]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Assumes objective is differentiable.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Stochastic Dual Coordinate Ascent (SDCA)</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} \sum_{i} f_{i}(x) + \frac{\lambda}{2} \norm{x}_2^2$</td>
      <!-- Convex             -->
      <td>$O(\frac{1}{\lambda \epsilon})$[^shalevshwartz-2012]</td>
      <!-- Strongly Convex    -->
      <td>$O(( \frac{1}{\lambda} ) \log ( \frac{1}{\lambda \epsilon} ))$[^shalevshwartz-2012]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Accelerated Proximal Stochastic Dual Coordinate Ascent (APSDCA)</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{C}} \sum_{i} f_{i}(x) + \lambda g(x)$</td>
      <!-- Convex             -->
      <td>$O(\min (\frac{1}{\lambda \epsilon}, \sqrt{\frac{N}{\lambda \epsilon}} ))$[^shalevshwartz-2013]</td>
      <!-- Strongly Convex    -->
      <td>$O(\min (\frac{1}{\lambda}, \sqrt{\frac{N}{\lambda}}) \log ( \frac{1}{\epsilon} ))$[^shalevshwartz-2013]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>Stochastic Average Gradient (SAG)</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} \sum_{i} f_{i}(x) + \lambda g(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon)$[^schmidt-2013]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1/\epsilon))$[^schmidt-2013]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f_{i}(x)$ is differentiable.
      </td>
    </tr>
    <tr>
      <!-- Algorithm          -->
      <td>MISO</td>
      <!-- Problem            -->
      <td>$\displaystyle \min_{x \in \mathbb{R}^n} \sum_{i} f_{i}(x) + \lambda g(x)$</td>
      <!-- Convex             -->
      <td>$O(1 / \epsilon)$[^mairal-2013]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1/\epsilon))$[^mairal-2013]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        Applicable when $f_{i}(x)$ is differentiable. $g(x)$ may be used as
        a barrier function.
      </td>
    </tr>
  </tbody>
</table>

Other Methods
=============

  The following methods are meta-algorithms, typically used in distributed
settings. Unlike preceding methods, they require solutions to optimization as
steps within each iteration.

<table markdown class="table table-bordered table-centered">
  <colgroup>
    <col style="width:20%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:10%">
    <col style="width:40%">
  </colgroup>

  <thead>
    <tr>
      <th>Algorithm          </th>
      <th>Problem Formulation</th>
      <th>Convex             </th>
      <th>Strongly Convex    </th>
      <th>Per-Iteration Cost </th>
      <th>Notes              </th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <!-- Algorithm          -->
      <td>Alternating Direction Method of Multipliers (ADMM)</td>
      <!-- Problem            -->
      <td>
        $$
          \begin{align*}
            \min_{x,z} \quad
              & f(x) + g(z) \\
            \text{s.t.} \quad
              & Ax + Bz = c
          \end{align*}
        $$
      </td>
      <!-- Convex             -->
      <td>$O(1/\epsilon)$[^blog-admm]</td>
      <!-- Strongly Convex    -->
      <td>$O(\log (1/\epsilon))$[^hong-2012]</td>
      <!-- Per-Iteration Cost -->
      <td>$O(n)$</td>
      <!-- Notes              -->
      <td>
        The stated convergence rate for "Strongly Convex" only requires $f(x)$ to
        be strongly convex, not $g(x)$. This same rate can also be applied to
        the "Convex" case under several non-standard assumptions[^hong-2012].
        Matrices $A$ and $B$ may also need to be full column rank[^deng-2012] .
      </td>
    </tr>
  </tbody>
</table>

<!-- Footnotes -->
[^blog-gd]:
  [Gradient Descent][blog-gd]

[^blog-sd]:
  [Subgradient Descent][blog-sd]

[^blog-agd]:
  [Accelerated Gradient Descent][blog-agd]

[^blog-pgd]:
  [Proximal Gradient Descent][blog-pgd]

[^blog-apgd]:
  [Accelerated Proximal Gradient Descent][blog-apgd]

[^blog-fw]:
  [Franke-Wolfe Algorithm][blog-fw]

[^blog-admm]:
  [Alternating Direction Method of Multipliers][blog-admm]

[^richtarik-2011]:
  [Richtarik and Takac, 2011][richtarik-2011]

[^ee381-md]:
  [EE381 Slides on Mirror Descent][ee381-md]

[^ee381-gd]:
  [EE381 Slides on Gradient Descent][ee381-gd]

[^bubeck-agd]:
  [Sebastien Bubeck's article on Accelerated Gradient Descent for Smooth and Strongly Convex objectives][bubeck-agd]

[^garber-2014]:
  [Garber and Hazan, 2014][garber-2014]

[^mairal-2013]:
  [Mairal, 2013][mairal-2013]

[^ee364a-unconstrained]:
  [EE364a Slides on Unconstrained Optimization Algorithms][ee364a-unconstrained]

[^ee236c-qnewton]:
  [EE236c Slides on Quasi-Newton Methods][ee236c-qnewton]

[^bach-2012]:
  [Bach's slides on Stochastic Methods][bach-2012]

[^shalevshwartz-2012]:
  [Shalev-Shwartz and Zhang, 2012][shalevshwartz-2012]

[^shalevshwartz-2013]:
  [Shalev-Shwartz and Zhang, 2013][shalevshwartz-2013]

[^schmidt-2013]:
  [Schmidt et al, 2013][schmidt-2013]

[^garber-2014]:
  [Garber and Hazan, 2014][garber-2014]

[^deng-2012]:
  [Deng and Yin, 2012][deng-2012], Table 1.1

[^hong-2012]:
  [Hong and Luo, 2012][hong-2012], Section 2

<!-- References -->
[blog-gd]: {filename}/2013-04-10-gradient-descent.markdown
[blog-sd]: {filename}/2013-04-11-subgradient-descent.markdown
[blog-agd]: {filename}/2013-04-12-accelerated-gradient-descent.markdown
[blog-pgd]: {filename}/2013-04-19-proximal-gradient.markdown
[blog-apgd]: {filename}/2013-04-25-accelerated-proximal-gradient.markdown
[blog-fw]: {filename}/2013-05-04-frank-wolfe.markdown
[blog-admm]: {filename}/2014-07-20-admm-revisited.markdown

[richtarik-2011]: http://arxiv.org/abs/1107.2848
[ee381-md]: http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_24_Scribe_Notes.final.pdf
[ee381-gd]: http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf
[bubeck-agd]: https://blogs.princeton.edu/imabandit/2014/03/06/nesterovs-accelerated-gradient-descent-for-smooth-and-strongly-convex-optimization/
[garber-2014]: http://arxiv.org/abs/1406.1305
[mairal-2013]: http://arxiv.org/abs/1305.3120
[ee364a-unconstrained]: http://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf
[ee236c-qnewton]: http://www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf
[bach-2012]: http://www.ann.jussieu.fr/~plc/bach2012.pdf
[shalevshwartz-2012]: http://arxiv.org/abs/1209.1873
[shalevshwartz-2013]: http://arxiv.org/abs/1309.2375
[schmidt-2013]: http://arxiv.org/abs/1309.2388
[hong-2012]: http://arxiv.org/abs/1208.3922
[deng-2012]: ftp://ftp.math.ucla.edu/pub/camreport/cam12-52.pdf
