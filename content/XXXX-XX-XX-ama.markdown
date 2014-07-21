<!--
Title: Alternating Minimization Algorithm and Proximal Gradient Descent
Date: 2013-07-20 00:00
Category: optimization
Tags: optimization, fobos, admm, ama, proximal
Slug: ama
-->


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

<a name="references" href="#references">References</a>
======================================================

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
