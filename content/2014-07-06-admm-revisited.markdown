Title: ADMM, revisited
Date: 2013-07-06 00:00
Category: optimization
Tags: optimization, distributed, admm
Slug: admm-revisited

  When I originally wrote about the [Alternating Direction Method of
Multipliers][admm] algorithm, the community's understanding of its
convergence properties was light to say the least. While it has long been
known (See [Boyd's excellent article][boyd], Appendix A) that ADMM _will_
converge, it is only recently that the community has begun to establish _how
fast_ it converges (e.g. [Hong][hong], [Deng][deng], [Feng][feng], [He][he]).

  In this article, we'll explore one way to establish an $O(1 / \epsilon)$ rate
of convergence. Unlike previous convergence proofs presented in this blog, we
won't directly show that the primal objective value alone converges to its
optimal value; instead, we'll show that a particular function involving the
primal objective and a [Variational Inequality][variational-inequality]
converges at the desired rate.

  Finally, we'll end by characterizing the relationship between the Alternating
Minimization Algorithm algorithm (AMA) and ADMM. We'll see that a vary slight
variation to ADMM's setup results in AMA, an algorithm identical to [Proximal
Gradient Descent][prox-grad].


<a name="implementation" href="#implementation">How does it work?</a>
=====================================================================

  Let's begin by introducing the optimization problem ADMM solves,

$$
\begin{align}
\begin{split}
  \underset{x,z}{\text{minimize}} \qquad
    & f(x) + g(z) \\
  \text{s.t.}                     \qquad
    & Ax + Bz = c \\
\end{split} \label{eqn:objective}
\end{align}
$$

  This problem is characterized by 2 primal variables, $x$ and $z$, which are
related by a linear equation. In machine learning, a common scenario is to
choose $A$, $B$, and $c$ such that $x = z$, making the setup particularly
simple. For the rest of this article, we'll assume that $Ax + Bz = c$ is the
only constraint we consider -- otherwise, constraints can be incorporated into
$f$ and $g$ by letting them be infinite when constraints are broken.

  The ADMM algorithm then finds the "saddle point" of the Augmented
Lagrangian for the corresponding problem,

$$
\begin{align} \label{eqn:lagrangian}
  L_{\rho}(x, z, y) = f(x) + g(z) + \langle y, Ax + Bz - c \rangle
                      + \frac{\rho}{2} || Ax + Bz - c ||_2^2
\end{align}
$$

  Note that we say _Augmented_ Lagrangian, as the typical Lagrangian does not
include the final quadratic term. It's easy to see, however, that the quadratic
does not affect the problem's optimal solution, as the constraint $Ax + Bz = c$
holds for all valid solutions.

  The ADMM algorithm iteratively minimizes $L_{\rho}$ with respect to $x$ for
fixed $z$ and $y$, then minimizes $z$ for fixed $x$ and $y$, and finally
takes a gradient step with respect to $y$ for fixed $x$ and $z$.

<div class="pseudocode" markdown>
  **Input** Step size $\rho$, initial primal iterates $x^{(0)}$ and $z^{(0)}$,
            initial dual iterate $y^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Let $x^{(t+1)} = \underset{x}{\text{argmin}} \quad L_{\rho}( x        , z^{(t)}, y^{(t)} )$
    3. Let $z^{(t+1)} = \underset{z}{\text{argmin}} \quad L_{\rho}( x^{(t+1)}, z      , y^{(t)} )$
    4. Let $y^{(t+1)} = y^{(t)} + \rho ( Ax^{(t+1)} + Bz^{(t+1)} - c )$
</div>

  Intuitively, the extra quadratic term prevents each iteration of the
algorithm from stepping "too far" from the last iteration, an idea that's also
at the core of [Proximal Gradient Descent][prox-grad]. In fact, [we'll see
below](#ama) that a slight variation to ADMM is precisely Proximal Gradient
Descent.

<div class="img-center">
  <img src="http://placehold.it/400x300"></img>
  <span class="caption">
    TODO Animation of ADMM on an example problem
  </span>
</div>

  In the remainder of the article, we'll often use the following notation for
conciseness,

$$
\begin{align*}
  w    &= \begin{bmatrix}
            x \\
            z \\
            y
          \end{bmatrix} \\
  h(w) &= f(x) + g(z) \\
  F(w) &= \begin{bmatrix}
            A^T y \\
            B^T y \\
            - (Ax + Bz - c)
          \end{bmatrix}
\end{align*}
$$


<a name="proof" href="#proof">Why does it work?</a>
===================================================

  Unlike other convergence proofs presented on this website, we won't directly
show that the objective converges to its minimum as $t \rightarrow \infty$.
Indeed, limiting ourselves to analysis of the objective completely ignores the
constraint $Ax + Bz = c$. Instead, we'll use the following variational
inequality condition to describe an optimal solution. In particular, a solution
$w^{*}$ is optimal if,

$$
\begin{align} \label{vi}
  \forall w \in \mathbb{R}^{n} \qquad
    h(w) - h(w^{*}) + \langle w - w^{*}, F(w^{*}) \rangle &\ge 0
\end{align}
$$

<div class="img-center">
  <img src="http://placehold.it/400x300"></img>
  <span class="caption">
    TODO geometric interpretation of variational inequality.
  </span>
</div>

  For the following proof, we'll replace $w^{*}$ with
$\bar{w}_t = (1/t) \sum_{\tau=1}^{t} w_{\tau}$ and $0$ on the right hand side
with $-\epsilon_t$ where $\epsilon_t = O(1/t)$. By showing that we can
approximately satisfy this inequality at a rate $O(1/t)$, we establish the
desired convergence rate.

**Assumptions**

  The assumptions on ADMM are almost as light as we can imagine. This is
largely due to the fact that we needn't use gradients or subgradients for
$h(z)$.

1. $f(x) + g(z)$ is convex.
2. There exists a solution $[ x^{*}; z^{*} ]$ that minimizes $f(x) + g(z)$
  while respecting the constraint $Ax + Bz = c$.

**Proof Outline**

  The proof presented hereafter is a particularly simple if unintuitive one.
Theoretically, the only tools necessary are the linear lower bound definition
of a convex function, the subgradient condition for optimality in an
unconstrained optimization problem, and Jensen's Inequality. Steps 1 and
2 below rely purely on the first 2 of these tools. Step 3 merely massages
a preceding equation into a simpler form via completing squares. Step 4 closes
by exploiting a telescoping sum and Jensen's Inequality to obtain the desired
result,

$$
\forall w \qquad
h(\bar{w}_t) - h(w) + \langle
  F(\bar{w}^{(t)}),
  \bar{w}^{(t)} - w
\rangle
\le \frac{1}{t} \left(
  \frac{\rho}{2} \norm{Ax-c}_2^2 + \frac{1}{2\rho} \norm{y}_2^2
\right)
$$

  As $t \rightarrow \infty$, the right hand side of this equation goes to 0,
rendering the same statement as the variational inequality optimality condition
in Equation $\ref{vi}$.

**Step 1** Optimality conditions for Step A. In this portion of the proof,
we'll use the fact that $x^{(t+1)}$ is defined as the solution of an
optimization problem to derive a subgradient for $f$ at $x^{(t+1)}$. We'll then
substitute this into $f$'s definition of convexity. Finally, terms are
rearranged and the contents of Step C of the algorithm are used to derive
a final expression.

  We begin by recognizing that $x^{(t+1)}$ minimizes
$L_{\rho}(x, z^{(t)}, y^{(t)})$ as a function of $x$. As $x$ is unconstrained,
zero must be a valid subgradient for $L_{\rho}$ evaluated at $x^{(t+1)},
z^{(t)}, y^{(t)}$,

$$
\begin{align*}
  0
  &\in \partial_x L_{\rho}(x^{(t+1)}, z^{(t)}, y^{(t)})                               \\
  &= \partial_{x} f(x^{(t+1)}) + A^T y^{(t)} + \rho A^T (Ax^{(t+1)} + Bz^{(t)} - c)   \\
  - A^T \left( y^{(t)} + \rho (Ax^{(t+1)} + Bz^{(t)} - c) \right)
  &\in \partial_x f(x^{(t+1)})
\end{align*}
$$

  As $f$ is convex, we further know that it is lower bounded by its linear
approximation everywhere,

$$
\begin{align*}
  \forall x \qquad
    f(x) &\ge f(x^{(t+1)}) + \langle
      \partial_x f(x^{(t+1)}),
      x - x^{(t+1)}
    \rangle
\end{align*}
$$

  Substituting in our subgradient for $\partial_x f(x^{(t+1)})$ and subtracting
the contents of the right hand side from both sides, we obtain,

$$
\begin{align*}
  \forall x \qquad
    0 &\le f(x) - f(x^{(t+1)}) + \langle
      A^T (y^{(t)} + \rho (A x^{(t+1)} + B z^{(t)} - c ),
      x - x^{(t+1)}
    \rangle
\end{align*}
$$

  Now recall Step C of the algorithm:
$y^{(t+1)} = y^{(t)} + \rho (A x^{(t+1)} + Bz^{(t+1)} - c)$. The left side of
the inner product looks very similar to this, so we'll substitute it in as best
we can,

$$
\begin{align*}
  \forall x \qquad
    0 &\le f(x) - f(x^{(t+1)}) + \langle
      A^T (y^{(t+1)} + \rho Bz^{(t)} - \rho Bz^{(t+1)}),
      x - x^{(t+1)}
    \rangle
\end{align*}
$$

  We finish by moving everything _not_ multiplied by $\rho$ to the opposite
side of the inequality,

$$
\begin{align} \label{eqn:36}
  \forall x \qquad
    f(x^{(t+1)}) - f(x) + \langle
      x^{(t+1)} - x,
      A^T y^{(t+1)}
    \rangle
    &\le \rho \langle
      Bz^{(t)} - Bz^{(t+1)},
      A x - A x^{(t+1)}
    \rangle
\end{align}
$$

**Step 2** Optimality conditions for Step B. Similar to Step 1, we'll use the
fact that $z^{(t+1)}$ is the solution to an unconstrained optimization problem
and will substitute in Step C's definition for $y^{(t+1)}$.

$$
\begin{align*}
  0
  &\in \partial_z L(x^{(t+1)}, z, y^{(t)})                                          \\
  &= \partial_z g(z^{(t+1)}) + B^T y^{(t)} + \rho B^T (Ax^{(t+1)} + Bz^{(t+1)} - c) \\
  - B^T \left( y^{(t)} + \rho (Ax^{(t+1)} + Bz^{(t+1)} - c) \right)
  &\in \partial_z g(z^{(t+1)})
\end{align*}
$$

  As $g$ is convex, it is lower bounded by its linear approximation,

$$
\begin{align*}
  \forall z \qquad
    g(z) &\ge g(z^{(t+1)}) + \langle
      \partial_z g(z^{(t+1)}),
      z - z^{(t+1)}
    \rangle
\end{align*}
$$

  Substituting in the previously derived subgradient and moving all terms to
the left side, we obtain,

$$
\begin{align*}
  \forall z \qquad
    0 &\le g(z) - g(z^{(t+1)}) + \langle
      B^T (y^{(t)} + \rho (A x^{(t+1)} + B z^{(t+1)} - c )),
      z - z^{(t+1)}
    \rangle
\end{align*}
$$

  Substituting in Step C's definition for $y^{(t+1)}$ again and moving
everything to the opposite side of the inequality, we conclude that,

$$
\begin{align} \label{eqn:37}
  \forall z \qquad
    g(z^{(t+1)}) - g(z) + \langle
      B^T y^{(t+1)},
      z^{(t+1)} - z
    \rangle
    &\le 0
\end{align}
$$

**Step 3** We now sum Equation $\ref{eqn:36}$ with Equation $\ref{eqn:37}$.
We'll end up with an expression that is not easy to understand initially, but
by factoring several of its terms into quadratic forms and substituting them
back in, we obtain a simpler expression that can be described as a sum of
squared 2-norms.

  We begin by summing equations $\ref{eqn:36}$ and $\ref{eqn:37}$.

$$
\begin{align*}
  & f(x^{(t+1)}) + g(z^{(t+1)}) - f(x) - g(z) + \langle
    B^T y^{(t+1)},
    z^{(t+1)} - z
  \rangle + \langle
    A^T y^{(t+1)},
    x^{(t+1)} - x
  \rangle \\
  & \qquad \le \rho \langle
    Ax - Ax^{(t+1)},
    Bz^{(t)} - Bz^{(t+1)}
  \rangle
\end{align*}
$$

  Next, we use the definitions of $h(w)$ and $F(w)$ on the left hand side,

$$
\begin{align*}
  & h(w^{(t+1)}) - h(w) + \langle
    F(w^{(t+1)}),
    w^{(t+1)} - w
  \rangle + \langle
    Ax^{(t+1)} + Bz^{(t+1)} - c,
    y^{(t+1)} - y
  \rangle \\
  & \qquad \le \rho \langle
    Ax - Ax^{(t+1)},
    Bz^{(t)} - Bz^{(t+1)}
  \rangle
\end{align*}
$$

  Then, moving the last term on the left side of the inequality over and
observing that Step C implies
$(1/\rho) (y^{(t+1)} - y^{(t)}) = Ax^{(t+1)} + Bz^{(t+1)} - c$,

$$
\begin{align}
\begin{split}
  & h(w^{(t+1)}) - h(w) + \langle
    F(w^{(t+1)}),
    w^{(t+1)} - w
  \rangle  \\
  & \qquad \le \rho \langle
    Ax - Ax^{(t+1)},
    Bz^{(t)} - Bz^{(t+1)}
  \rangle + \frac{1}{\rho} \langle
    y^{(t+1)} - y^{(t)},
    y - y^{(t+1)}
  \rangle
\end{split} \label{eqn:38}
\end{align}
$$

  We will now tackle the two components on the right hand side of the
inequality in isolation. Our goal is to rewrite these inner products in terms
of sums of $\norm{\cdot}_2^2$ terms.

  We'll start with $\langle Ax - Ax^{(t+1)}, Bz^{(t)} - Bz^{(t+1)} \rangle$. In
the next equations, we'll add many terms that will cancel themselves out, then
we'll group them together into a sum of 4 terms,

$$
\begin{align}
  &
  \begin{split}
    2 \langle Ax - Ax^{(t+1)}, Bz^{(t)} - Bz^{(t+1)} \rangle
  \end{split} \notag \\
  &
  \begin{split}
  = & + \norm{Ax        -c}_2^2 & + 2 \langle Ax         - c, B z^{(t  )} \rangle & + \norm{Bz^{(t  )}}_2^2 \\
    & - \norm{Ax        -c}_2^2 & - 2 \langle Ax         - c, B z^{(t+1)} \rangle & - \norm{Bz^{(t+1)}}_2^2 \\
    & + \norm{Ax^{(t+1)}-c}_2^2 & + 2 \langle Ax^{(t+1)} - c, B z^{(t+1)} \rangle & + \norm{Bz^{(t+1)}}_2^2 \\
    & - \norm{Ax^{(t+1)}-c}_2^2 & - 2 \langle Ax^{(t+1)} - c, B z^{(t  )} \rangle & - \norm{Bz^{(t  )}}_2^2
  \end{split} \notag \\
  &
  \begin{split}
  = & + \norm{Ax         + Bz^{(t)}   - c}_2^2 & - \norm{Ax         + Bz^{(t+1)} - c}_2^2 \\
    & + \norm{Ax^{(t+1)} + Bz^{(t+1)} - c}_2^2 & - \norm{Ax^{(t+1)} + Bz^{(t  )} - c}_2^2
  \end{split} \label{eqn:39}
\end{align}
$$

  We'll do the same for $\langle y^{(t+1)} - y^{(t)}, y - y^{(t+1)} \rangle$,

$$
\begin{align}
  &
  \begin{split}
    2 \langle y^{(t+1)} - y^{(t)}, y - y^{(t+1)} \rangle
  \end{split} \notag \\
  &
  \begin{split}
  = & + \norm{y      }_2^2 & + 2 \langle y      , - y^{(t  )} \rangle & + \norm{y^{(t  )}}_2^2 \\
    & - \norm{y      }_2^2 & - 2 \langle y      , - y^{(t+1)} \rangle & - \norm{y^{(t+1)}}_2^2 \\
    & - \norm{y^{(t)}}_2^2 & - 2 \langle y^{(t)}, - y^{(t+1)} \rangle & - \norm{y^{(t+1)}}_2^2 \\
  \end{split} \notag \\
  &
  \begin{split}
  = & + \norm{y       - y^{(t  )}}_2^2
      - \norm{y       - y^{(t+1)}}_2^2
      - \norm{y^{(t)} - y^{(t+1)}}_2^2
  \end{split} \label{eqn:40}
\end{align}
$$

  Finally, let's plug equations $\ref{eqn:39}$ and $\ref{eqn:40}$ into
$\ref{eqn:38}$.

$$
\begin{align}
\begin{split}
  & h(w^{(t+1)}) - h(w) + \langle
    F(w^{(t+1)}),
    w^{(t+1)} - w
  \rangle  \\
  & \qquad \le \frac{\rho}{2} \left( \begin{split}
    & + \norm{Ax         + Bz^{(t  )} - c}_2^2 & - \norm{Ax         + Bz^{(t+1)} - c}_2^2 \\
    & + \norm{Ax^{(t+1)} + Bz^{(t+1)} - c}_2^2 & - \norm{Ax^{(t+1)} + Bz^{(t  )} - c}_2^2
  \end{split} \right) \\
  & \qquad + \frac{1}{2\rho} \left( \begin{split}
    & + \norm{y       - y^{(t  )}}_2^2 \\
    & - \norm{y       - y^{(t+1)}}_2^2 \\
    & - \norm{y^{(t)} - y^{(t+1)}}_2^2
  \end{split} \right)
\end{split}
\end{align}
$$

  Recall that $(1/\rho)(y^{(t+1)} - y^{(t)}) = Ax^{(t+1)} + Bz^{(t+1)} - c$. Then,

$$
\begin{align*}
  \frac{\rho}{2} \norm{ Ax^{(t+1)} + Bz^{(t+1)} - c }
  &= \frac{\rho}{2}  \norm{ \frac{1}{\rho} (y^{(t+1)} - y^{(t  )}) } \\
  &= \frac{1}{2\rho} \norm{                 y^{(t  )} - y^{(t+1)}  }
\end{align*}
$$

  We can substitute that into the right hand side of the preceding equation to
cancel out a couple terms,

$$
\begin{align*}
  = &
  \frac{\rho}{2} \left( \begin{split}
    & + \norm{Ax + Bz^{(t)} - c}_2^2 &             - \norm{Ax         + Bz^{(t+1)} - c}_2^2 \\
    &                                & \underbrace{- \norm{Ax^{(t+1)} + Bz^{(t  )} - c}_2^2}_{ \text{ always $\le 0$ } }
  \end{split} \right) \\
  & + \frac{1}{2\rho} \left( \begin{split}
    & + \norm{y - y^{(t  )}}_2^2 \\
    & - \norm{y - y^{(t+1)}}_2^2
  \end{split} \right)
\end{align*}
$$

  Finally dropping the portion of the equation that's always non-positive
(doing so doesn't affect the validity of the inequality), we obtain a concise
inequality in terms of sums of $\norm{\cdot}_2^2$.

$$
\begin{align*}
  & h(w^{(t+1)}) - h(w) + \langle
    F(w^{(t+1)}),
    w^{(t+1)} - w
  \rangle  \\
  & \qquad \le \frac{\rho}{2} \left(
      \norm{Ax + Bz^{(t  )} - c}_2^2
    - \norm{Ax + Bz^{(t+1)} - c}_2^2
  \right) \\
  & \qquad + \frac{1}{2\rho} \left(
      \norm{y - y^{(t  )}}_2^2
    - \norm{y - y^{(t+1)}}_2^2
  \right)
\end{align*}
$$

**Step 4** Averaging across iterations. We're now in the home stretch. In this
step, we'll sum the previous equation across $t$. The sum will "telescope",
crossing out terms until we're left only with the initial and final conditions.
A quick application of Jensen's inequality will get us the desired result.

  We begin by summing the previous equation across iterations,

$$
\begin{align*}
  & \sum_{\tau=0}^{t-1} h(w^{(\tau+1)}) - h(w) + \langle
    F(w^{(\tau+1)}),
    w^{(\tau+1)} - w
  \rangle  \\
  & \qquad \le \frac{\rho}{2} \left(
                  \norm{Ax + Bz^{(0)} - c}_2^2
    \underbrace{- \norm{Ax + Bz^{(t)} - c}_2^2}_{\le 0}
  \right) + \frac{1}{2\rho} \left(
                  \norm{y - y^{(0)}}_2^2
    \underbrace{- \norm{y - y^{(t)}}_2^2}_{\le 0}
  \right)
\end{align*}
$$

  For convenience, we'll choose $z^{(0)}$ and $y^{(0)}$ equal to zero. We'll
also drop the terms $-\norm{Ax + Bz^{(t)} - c}_2^2$ and
$-\norm{y - y^{(t)}}_2^2$ from the expression, as both terms are always
non-positive. This gives us,

$$
\begin{align*}
  \sum_{\tau=0}^{t-1} h(w^{(\tau+1)}) - h(w) + \langle
    F(w^{(\tau+1)}),
    w^{(\tau+1)} - w
  \rangle
  \le \frac{\rho}{2}  \norm{Ax - c}_2^2
             + \frac{1}{2\rho} \norm{ y    }_2^2
\end{align*}
$$

  Finally, recall that for a convex function $h(w)$, Jensen's Inequality states that

$$
  h(\bar{w}_t)
  = h \left( \frac{1}{t} \sum_{\tau=1}^{t} w_{\tau} \right)
  \le \frac{1}{t} \sum_{\tau=1}^{t} h(w_{\tau})
$$

  The same is true for each of $F(w)$'s components (they're linear in $w$).
Thus, we can apply this statement to the left hand side of the preceding
equation after multiplying by $1/t$ to obtain,

$$
\begin{align*}
  h(\bar{w}^{(t)}) - h(w) + \langle
    F(\bar{w}^{(t)}),
    \bar{w}^{(t)} - w
  \rangle
  \le \frac{1}{t} \left(
      \frac{\rho}{2}  \norm{Ax - c}_2^2
    + \frac{1}{2\rho} \norm{ y    }_2^2
  \right)
\end{align*}
$$

  The right hand side decreases as $O(1/t)$, thus ADMM converges at a rate of
at least $O(1/\epsilon)$ as desired.

<a name="usage" href="#usage">When should I use it?</a>
=======================================================

  Similar to the proximal methods presented on this website, ADMM is only
efficient if we can perform each of its steps efficiently. Solving
2 optimization problems at each iteration may be very fast or very slow,
depending on if a closed form solution exists for $x^{(t+1)}$ and $z^{(t+1)}$.

  ADMM has been particularly useful in supervised machine learning, where $A$,
$B$, and $c$ are chosen such that $x = z$. In this scenario, $f$ is taken to be
the prediction loss on the training set, and $g$ an appropriate regularizer,
typically a norm such as $\ell_1$ or a [group sparsity norm][group-sparsity].
[ADMM also lends][map-admm] itself to inferring the most likely setting for
settings for latent variables in a factor graph. The primary benefit of ADMM in
both of these cases is not its rate of convergence but rather how [easily it
lends itself to distributed computation][distributed-svm]. [Applications in
Compressed Sensing][basis-pursuit] see similar benefits.

  All in all, ADMM is _not_ a quick method, but it is a scalable one. ADMM is
best suited when data is too large to fit on a single machine or when
$x^{(t+1)}$ and $z^{(t+1)}$ can be solved in closed form. While very
interesting in its own right, ADMM is rarely an algorithm of choice.

<a name="ama" href="#ama">ADMM and Proximal Gradient Descent</a>
================================================================

  While initially difficult to see, ADMM and Proximal Gradient Descent are in
fact very similar algorithms. To be precise, a slight variant of ADMM called
the [Alternating Minimization Algorithm][ama] (AMA) is exactly the same as
Proximal Gradient Descent as applied to the _dual_ problem. Let's make this
concrete.

  First, recall that the ADMM algorithm minimizes the Augmented Lagrangian,
first with respect to $x$, then $z$, then finally takes a gradient ascent
step with respect to the dual variable $y$. AMA is identical to ADMM, except
that instead of minimizing the Augmented Lagrangian with respect to $x$, it
minimizes the normal Lagrangian. Note that this only works if $f(x)$ is bounded
below (we'll assume its strongly convex and differentiable henceforth).

<div class="pseudocode" markdown>
  **Input** Step size $\rho$, initial primal iterates $x^{(0)}$ and $z^{(0)}$,
            initial dual iterate $y^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Let $x^{(t+1)} = \underset{x}{\text{argmin}} \quad L_{0   }( x        , z^{(t)}, y^{(t)} )$
    3. Let $z^{(t+1)} = \underset{z}{\text{argmin}} \quad L_{\rho}( x^{(t+1)}, z      , y^{(t)} )$
    4. Let $y^{(t+1)} = y^{(t)} + \rho ( Ax^{(t+1)} + Bz^{(t+1)} - c )$
</div>

  We'll show that AMA is the same as Proximal Gradient Ascent on the dual
problem by way of [Forward-Backward Splitting][fobos] (FoBoS), another
optimization algorithm originating in the '70s. We'll first show that AMA is
equivalent to FoBoS, then that Proximal Gradient Descent is a particular
instance of FoBoS.

[**FoBoS**][fobos] Unlike optimization algorithms presented here,
Forward-Backward Splitting does not aim to minimize an objective. Rather, given
a multi-valuated mapping $w \rightarrow F(w)$, its goal is to find find a point
$w^{*}$ such that $0 \in F(w^{*})$. Keep in mind that $F$ is not necessarily
a function with a single output -- $F(w)$ denotes a set of values. For this
discussion, we'll assume that $F$ is a "monotone" operator, meaning that for
all $u \in F(w)$ and $u' \in F(w')$, $\langle u-u', w-w' \rangle \ge 0$.

  Forward-Backward Splitting is an algorithm for finding such a $w^{*}$, given
a few assumptions. Namely,

1. $F(w) = \Psi(w) + \Theta(w)$ for monotone operators $\Psi$ and $\Theta$
2. $\Psi(w)$ has exactly one value for each $w$ in the domain

  Given this, FoBoS will converge to a $w^{*}$ such that $0 \in F(w^{*})$. The
algorithm itself is,

<div class="pseudocode" markdown>
  **Input** Step sizes $\{ \rho_t \}_{t=1}^{\infty}$, initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Let $w^{(t+1/2)} = w^{(t)} - \rho_t \Psi(w^{(t)})$
    3. Let $w^{(t+1)}$ be such that $w^{(t+1)} + \rho_t \Theta(w^{(t+1)}) = w^{(t+1/2)}$
</div>

  An equivalent, more concise way to describe FoBoS is with
$w^{(t+1)} = (I + \rho_t \Theta)^{-1} (I - \rho_t \Psi) (w^{(t)})$. We'll now
show that for appropriate choice of $F$, $\Psi$, and $\Theta$, Proximal
Gradient Descent is merely FoBoS in disguise.

[**Prox. Grad. Desc. to FoBoS**][fobos-slides] Suppose we want to minimize
$f(x) + g(x)$. If the problem is unconstrained, this is equivalent to finding
$0 \in F(x) = \nabla f(x) + \partial_x g(x)$. Let's now define,

$$
\begin{align}
  \Psi(x) &= \nabla f(x) & \Theta(x) &= \partial_x g(x) \label{eqn:fobos-def}
\end{align}
$$

  Clearly, $(I - \rho_{t} \Psi)(x) = x - \rho_{t} \nabla f(x)$ matches the first
part of FoBoS, but we also need to show that
$\text{prox}_{\rho_t g}(x) = (I + \rho_{t} \Theta)^{-1}(x)$,

$$
\begin{align*}
  y
  & = \text{prox}_{\rho g}(x)                           \\
  & = \argmin_{y} g(y) + \frac{1}{2\rho}\norm{y-x}_2^2  \\
  0
  & \in \partial_x g(y) + \frac{1}{\rho} (y-x)          \\
  x
  & \in (I + \rho \Theta)(y)
\end{align*}
$$

  We now have that for the above choices of $\Psi$ and $\Theta$, the proximal
gradient descent algorithm can be reframed as identical to FoBoS:
$x^{(t+1)} = (I + \rho_t \Theta)^{-1} (I - \rho_t \Psi) (x^{(t)})$.

**AMA to FoBoS** We'll now show that AMA as applied to the ADMM objective is
simply an instance of FoBoS. We'll make use of the following operators,

$$
\begin{align*}
  \Psi(y)   &= A \nabla   f^{*}(A^T y)      &
  \Theta(y) &= B \partial g^{*}(B^T y) - c
\end{align*}
$$

  First, recall the subgradient optimality condition as applied to Step B of
ADMM (same as AMA). In particular, for $z^{(t+1)}$ to be the argmin of
$L(x^{(t+1)}, z, y^{(t)})$, it must be the case that,

$$
\begin{align*}
  0
  &\in \partial g(z^{(t+1)}) - B^T y^{(t)} - \rho B^T (c - Ax^{(t+1)} - Bz^{(t+1)}) \\
  B^T ( y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)}) )
  &\in \partial g(z^{(t+1)})
\end{align*}
$$

  Using $y \in \partial f(x) \Rightarrow x \in \partial f^{*}(y)$, we obtain,

$$
\begin{align*}
  z^{(t+1)}
  & \in \partial g^{*}(B^T ( y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)}) ))
\end{align*}
$$

  We now left-multiply by $B$, subtract $c$ from both sides to obtain, and use
the definition of $\Theta$ to obtain,

$$
\begin{align*}
  B z^{(t+1)} - c
  & \in \Theta( y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)}) )
\end{align*}
$$

  Now we multiply both sides by $\rho$ and add,
$y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)})$,

$$
\begin{align*}
  y^{(t)} - \rho Ax^{(t+1)}
  & \in (I + \rho \Theta)( y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)}) )
\end{align*}
$$

  We can invert $I + \rho \Theta$ and notice that the other side is
single-valued to obtain,

$$
\begin{align}
  (I + \rho \Theta)^{-1} (y^{(t)} - \rho Ax^{(t+1)})
  & = y^{(t)} + \rho (c - Ax^{(t+1)} - Bz^{(t+1)})   \notag \\
  (I + \rho \Theta)^{-1} (y^{(t)} - \rho Ax^{(t+1)})
  & = y^{(t+1)}                                                 \label{eqn:ama1} \\
\end{align}
$$

  Now, let's apply the same subgradient optimality to Step A of AMA.

$$
\begin{align*}
  0
  &\in \partial f(x^{(t+1)}) - A^T y^{(t)} \\
  A^T y^{(t)}
  &= \nabla f(x^{(t+1)})
\end{align*}
$$

  Using $y = \nabla f(x) \Rightarrow x = \nabla f^{*}(y)$ for strongly convex
$f$ and multiplying both sides by $A$,

$$
\begin{align}
  A f^{*} (A^T y^{(t)}) &= A f(x^{(t+1)}) \notag            \\
  \Psi(y^{(t)})         &= A x^{(t+1)}    \label{eqn:ama2}
\end{align}
$$

  Substituting in equation $\ref{eqn:ama2}$ into $\ref{eqn:ama1}$, we obtain,

$$
\begin{align*}
  y^{(t+1)} = (I + \rho \Theta)^{-1} (I - \rho \Psi) (y^{(t)})
\end{align*}
$$

  In other words, AMA is simply an instance of FoBoS.

  Finally, let's relate this reduction from AMA to FoBoS to Proximal Gradient
Descent's reduction. It is worth noting at this point that the dual objective to
$\ref{eqn:objective}$ is

$$
\min_{y} f^{*}(A^T y) + g^{*}(B^T y) - \langle y, c \rangle
$$

  It is easy to see from the dual problem formulation that $\Psi(y) = \nabla_y
f^{*}(A^T y)$ and that $\Theta(y) = \partial (g^{*}(B^T y) - \langle y,
c \rangle)$, just as prescribed in the preceding section on Proximal Gradient
Descent $\ref{eqn:fobos-def}$. Thus we conclude that AMA is identical to
Proximal Gradient Ascent on the dual.


<a name="extensions" href="#extensions">Extensions</a>
======================================================

**Accelerated** As ADMM is so closely related to Proximal Gradient-based
methods, one might ask if there exists an accelerated variant with a better
convergence rate. The answer is a resounding yes, as shown by [Goldstein et
al.][accelerated-admm], though care must be taken for non-strongly convex
objectives. In their article, Goldstein et al. show that a convergence rate of
$O(1/\sqrt{\epsilon})$ can be guaranteed if both $f$ and $g$ are strongly
convex. If this isn't the case, only a rate of $O(1/\epsilon)$ is shown.

**Online** In online learning, one is interested in solving a series of
supervised machine learning instances in sequence with minimal error. At each
iteration, the algorithm is presented with an input $x_t$, to which it responds
with a prediction $\hat{y}_t$. The world then presents the algorithm with the
correct answer $y_t$, and the algorithm suffers loss $l_t(y_t, \hat{y}_t)$. The
goal of the algorithm is to minimize the sum of errors $\sum_{t} l_t(y_t,
\hat{y}_t)$.

  In this setting, [Wang][online-admm] has shown that an online variant to ADMM
can achieve regret competitive with the best possible ($O(\sqrt{T})$ for
convex loss functions, $O(\log(T))$ for strongly convex loss functions).

**Stochastic** In a stochastic setting, one is interested in minimizing the
_average_ value of $f(x)$ via a series of samples. In [Ouyang et
al][stochastic-admm], convergence rates for a linearized variant of ADMM when
$f$ can only be accessed through samples.

**Multi Component** Traditional ADMM considers an objective with only
2 components $f(x)$ and $g(z)$. While applying the same logic to 3 or more is
straightforward, proving convergence for this scenario is more difficult. This
was the task taken by [He et al][multi-admm]. In particular, they showed that
a special variant of ADMM using "Gaussian back substitution" is ensured to
converge.

<a name="references" href="#references">References</a>
======================================================

**ADMM** While ADMM has existed for decades, it has only recently been brought
to light by [Boyd][admm]'s article describing its applications for statistical
machine learning. It is from this work from which I initially learned of ADMM.

**Proof of Convergence** The proof of convergence presented here is a verbose
expansion of that presented in [Wang][online-admm]'s paper on Online ADMM.

**Proximal Gradient Descent and ADMM** I was first made aware of the
relationship between AMA and ADMM in [Chi][convex-clustering]'s article on
convex clustering via ADMM and AMA. The relationship between Proximal Gradient
Descent and FoBoS is taken from [Berkeley's EE227a slides][fobos-slides] and
the relationship between FoBoS and AMA from [Goldstein et
al][accelerated-admm]'s work on Accelerated ADMM and AMA.

<!-- internal references -->
[admm]: {filename}/2012-06-24-admm.markdown
[prox-grad]: {filename}/2013-04-19-proximal-gradient.markdown

<!-- papers -->
[boyd]: http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
[fobos]: http://machinelearning.wustl.edu/mlpapers/paper_files/jmlr10_duchi09a.pdf
[fobos-slides]: http://www.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
[ama]: http://dspace.mit.edu/bitstream/handle/1721.1/3103/P-1836-19477130.pdf

<!-- convergence proofs -->
[hong]: http://arxiv.org/abs/1208.3922
[deng]: ftp://ftp.math.ucla.edu/pub/camreport/cam12-52.pdf
[feng]: http://iqua.ece.toronto.edu/~cfeng/notes/cfeng-admm12.pdf
[he]: http://www.math.hkbu.edu.hk/~xmyuan/Paper/HeYuan-SecondRevision.pdf

<!-- extensions -->
[stochastic-admm]: http://arxiv.org/pdf/1211.0632.pdf
[online-admm]: http://icml.cc/2012/papers/577.pdf
[accelerated-admm]: ftp://ftp.math.ucla.edu/pub/camreport/cam12-35.pdf
[bregman-admm]: http://arxiv.org/abs/1306.3203
[multi-admm]: http://www.optimization-online.org/DB_FILE/2010/12/2871.pdf

[variational-inequality]: http://supernet.isenberg.umass.edu/austria_lectures/fvisli.pdf
[splitting-methods]: http://arxiv.org/abs/1304.0499

<!-- uses -->
[group-sparsity]: http://arxiv.org/pdf/1104.1872.pdf
[basis-pursuit]: http://arxiv.org/pdf/1009.1128.pdf
[low-rank]: http://papers.nips.cc/paper/4434-linearized-alternating-direction-method-with-adaptive-penalty-for-low-rank-representation
[distributed-svm]: http://www.ece.umn.edu/users/alfonso/pubs/jmlr2010.pdf
[map-admm]: http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Martins_150.pdf
[convex-clustering]: http://arxiv.org/abs/1304.0499

<a name="reference-impl" href="#reference-impl">Reference Implementation</a>
============================================================================

```python
```
