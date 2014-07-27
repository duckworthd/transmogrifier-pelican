Title: From ADMM to Proximal Gradient Descent
Date: 2014-07-26 00:00
Category: optimization
Tags: optimization, fobos, admm, ama, proximal
Slug: admm-to-prox-grad

  At first blush, [ADMM][admm] and [Proximal Gradient Descent][prox-grad]
(ProxGrad) appear to have very little in common. The convergence analyses for
these two methods are unrelated, and the former operates on an Augmented
Lagrangian while the latter directly minimizes the primal objective. In this
post, we'll show that after a slight modification to ADMM, we recover the
proximal gradient algorithm applied to Lagrangian _dual_ of the ADMM objective.

  To be precise, we'll first make a slight modification to ADMM to construct
another algorithm known as the [Alternating Minimization Algorithm][ama] (AMA).
We'll then show this algorithm is an instance of a more general technique for
[Variational Inequality problems][variational-inequality] called
[Forward-Backward Splitting][fobos] (FOBOS). Finally, we'll show that ProxGrad
is also an instance of FOBOS with the exact same form. We conclude that these
two algorithms are equivalent.

<a name="ama" href="#ama">Alternating Minimization Algorithm</a>
================================================================

  The [Alternating Minimization Algorithm][ama] (AMA), originally proposed by
Paul Tseng in 1988, is an algorithm very similar to ADMM. In fact, the only
difference between these two methods is in the first step of each iteration.
Recall the pseudocode for ADMM; whereas ADMM minimizes the _Augmented_
Lagrangian with respect to $x$, AMA minimizes the _Non-Augmented_ Lagrangian,

<div class="pseudocode" markdown>
  **Input** Step size $\rho$, initial primal iterates $x^{(0)}$ and $z^{(0)}$,
            initial dual iterate $y^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Let $x^{(t+1)} = \underset{x}{\text{argmin}} \quad L_{   0}( x        , z^{(t)}, y^{(t)} )$
    3. Let $z^{(t+1)} = \underset{z}{\text{argmin}} \quad L_{\rho}( x^{(t+1)}, z      , y^{(t)} )$
    4. Let $y^{(t+1)} = y^{(t)} + \rho ( Ax^{(t+1)} + Bz^{(t+1)} - c )$
</div>

  Notice the $0$ instead of $\rho$ in the definition of $x^{(t+1)}$. This tiny
change, we'll see, is all that's necessary to turn ADMM into ProxGrad.

<a name="vi" href="#vi">Variational Inequalties</a>
===================================================

  To show the similarity between AMA and ProxGrad, we'll show that both
algorithms are instances of Forward-Backward Splitting (FOBOS). Unlike other
algorithms we've considered, FOBOS isn't about minimizing a real-valued
objective function subject to constraints. Instead, FOBOS solves Variational
Inequality problems, which we'll now describe.

  Variational Inequality (VI) problems involve a vector-to-vector function
$F: \mathbb{R}^n \rightarrow \mathbb{R}^n$ and a convex set $\mathcal{C}$. The
goal is to find an input $w^{*}$ such that,

$$
\begin{equation*}
  \forall w \in \mathcal{C} \quad
  \langle F(w^{*}), w - w^{*} \rangle \ge 0
\end{equation*}
$$

  If $\mathcal{C} = \mathcal{R}^n$, then this inequality can only hold when
$F(w^{*}) = 0$. For example, if $F = \nabla f$ for a differentiable convex
objective function $f$, then finding $F(w^{*}) = 0$ is the same as a finding
$f$'s unconstrained global minimum. Incorporating constraints is as simple as
letting $F(w) = [\nabla_x L(x,y); -\nabla_y L(x,y)]$ for Lagrangian $L(x,y)$
with primal variable $x$ and dual variable $y$ and $w = [x; y]$.

  What is not covered in this setup, however, is the case when $L$ is not
differentiable with respect to all parameters. We can expand on the concept of
Variational Inequalties a bit by letting $F(w)$ be a _subset_ of
$\mathcal{R}^{n}$ instead of a single value (that is,
$F: \mathcal{R}^n \rightarrow 2^{\mathcal{R}^{n}}$). We'll say that $F$ is
a _monotone operator_ if,

$$
\begin{align*}
  \forall w,w' \in \mathcal{C}; \,
  \forall u \in F(w);           \,
  \forall v \in F(w')           \quad
  \langle u-v, w-w' \rangle \ge 0
\end{align*}
$$

  Now if $\mathcal{C} = \mathcal{R}^n$ and
$F = [\partial_x L(x,y); -\partial_y L(x,y)]$, we can see that finding
$0 \in F(w^{*})$ is the same as solving the optimization described by $L$ for
non-smooth objective and constraint functions.

<a name="fobos" href="#fobos">Forward-Backward Splitting</a>
============================================================

  [Forward-Backward Splitting][fobos] FOBOS is an algorithm for finding
a $w^{*}$ that solves VI problems for particular choices of $F$. Namely,
we'll make the following assumptions.

1. $F(w) = \Psi(w) + \Theta(w)$ for [monotone operators][monotone] $\Psi$ and
  $\Theta$.
2. $\Psi(w)$ has exactly one value for each $w$ in its domain.

  Given this, FOBOS will converge to a $w^{*}$ such that $0 \in F(w^{*})$. The
algorithm itself is,

<div class="pseudocode" markdown>
  **Input** Step sizes $\{ \rho_t \}_{t=1}^{\infty}$, initial iterate $w^{(0)}$

  1. For $t = 0, 1, \ldots$
    2. Let $w^{(t+1/2)} = w^{(t)} - \rho_t \Psi(w^{(t)})$
    3. Let $w^{(t+1)}$ be such that $w^{(t+1)} + \rho_t \Theta(w^{(t+1)}) = w^{(t+1/2)}$
</div>

  An equivalent, more concise way to describe FOBOS is with
$w^{(t+1)} = (I + \rho_t \Theta)^{-1} (I - \rho_t \Psi) (w^{(t)})$. With this
formulation in mind, we'll now show that both AMA and ProxGrad are instances of
FOBOS performing the same set of operations.

<a name="reductions" href="#reductions">Reductions to FOBOS</a>
===============================================================

  We'll now show that for the specific optimization problem tackled by ADMM,
AMA is the same as Proximal Gradient Descent on the dual problem. First, recall
the problem ADMM is solving,

$$
\begin{align}
\begin{split}
  \underset{x,z}{\text{minimize}} \qquad
    & f(x) + g(z) \\
  \text{s.t.}                     \qquad
    & Ax + Bz = c \\
\end{split} \label{eqn:primal}
\end{align}
$$

  The dual problem to this is then,

$$
\begin{align}
\begin{split}
  - \underset{y}{\text{minimize}} \qquad
    & f^{*}(A^{T} y) + g^{*}(B^{T} z) - \langle y, c \rangle \\
\end{split} \label{eqn:dual}
\end{align}
$$

  where $f^{*}$ and $g^{*}$ are the [convex conjugates][convex-conjugate] to
$f$ and $g$, respectively. We'll now show that both AMA and Proximal Gradient
Descent are optimizing this same dual.

<a name="prox-grad-to-fobos" href="#prox-grad-to-fobos">Proximal Gradient Descent to FOBOS</a>
----------------------------------------------------------------------------------------------

  Suppose we want to minimize $f^{*}(A^T y) + g^{*}(B^T y)$. If the problem is
unconstrained, this is equivalent to finding

$$
\begin{align*}
  0 \in F(y)
  &= \partial_y \left( f^{*}(A^T y) + g(B^T y) - \langle y, c \rangle \right) \\
  &= A (\nabla_y f^{*})(A^T y) + B (\partial_y g^{*})(B^T y) - c
  \end{align*}
$$

  Let's now define,

$$
\begin{align}
  \Psi(y)   &= A (\nabla_y   f^{*})(A^T y)     &
  \Theta(y) &= B (\partial_y g^{*})(B^T y) - c
  \label{eqn:fobos-def}
\end{align}
$$

  Clearly, $(I - \rho_{t} \Psi)(y) = y - \rho_{t} A (\nabla_y f^{*})(A^T y)$
matches the first part of FOBOS and the "gradient step" part of ProxGrad, but
we also need to show that,

$$
\begin{align*}
  \text{prox}_{\rho_t g^{*}(B^T \cdot) - \langle \cdot, c \rangle}(y)
  & = (I + \rho_{t} \Theta)^{-1}(y)
\end{align*}
$$

  To do this, let's recall the definition of the prox operator,

$$
\begin{align*}
  \bar{y}
  & = \text{prox}_{\rho_t g^{*}(B^T \cdot) - \langle \cdot, c \rangle}(y) \\
  & = \argmin_{y'}  g^{*}(B^T y')
                    - \langle y', c \rangle
                    + \frac{1}{2\rho_t}\norm{y'-y}_2^2
\end{align*}
$$

  Since this is an unconstrained minimization problem, we know that $0$ must be
in the subgradient of this expression at $\bar{y}$.

$$
\begin{align*}
  0
  & \in B (\partial_{\bar{y}} g^{*})(B^T \bar{y}) - c + \frac{1}{\rho_t} (\bar{y}-y)  \\
  y
  & \in \bar{y} + \rho_t \left( B (\partial_{\bar{y}} g^{*})(B^T \bar{y}) - c \right) \\
  & = (I + \rho_t \Theta)(\bar{y})
\end{align*}
$$

  Apply $(I + \rho_t \Theta)^{-1}$ to both sides gives us the desired result,
We now have that for the above choices of $\Psi$ and $\Theta$, ProxGrad can
be reframed as identical to FOBOS,

$$
\begin{align*}
  y^{(t+1)} = (I + \rho_t \Theta)^{-1} (I - \rho_t \Psi) (y^{(t)})
\end{align*}
$$


<a name="ama-to-fobos" href="#ama-to-fobos">AMA to FOBOS</a>
------------------------------------------------------------

  We'll now show that AMA as applied to the ADMM objective is
simply an instance of FOBOS. Similar to the [ProxGrad
reduction](#prox-grad-to-fobos), we'll use the following definitions for
$\Psi$ and $\Theta$,

$$
\begin{align*}
  \Psi(y)   &= A (\nabla   f^{*})(A^T y)      &
  \Theta(y) &= B (\partial g^{*})(B^T y) - c
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

  Notice that this is exactly the same thing we concluded in the reduction from
ProxGrad to FOBOS. Thus, we have shown that both AMA and ProxGrad are the same
algorithm for the ADMM objective.

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
[fobos-convergence]: http://faculty.uml.edu/cbyrne/FBS.pdf
[ama]: http://dspace.mit.edu/bitstream/handle/1721.1/3103/P-1836-19477130.pdf

[variational-inequality]: http://supernet.isenberg.umass.edu/austria_lectures/fvisli.pdf

[nonexpansive]: http://en.wikipedia.org/wiki/Contraction_mapping#Firmly_non-expansive_mapping
[monotone]: http://web.stanford.edu/class/ee364b/lectures/monotone_slides.pdf
[convex-conjugate]: http://en.wikipedia.org/wiki/Convex_conjugate
[convex-clustering]: http://arxiv.org/abs/1304.0499
[accelerated-admm]: ftp://ftp.math.ucla.edu/pub/camreport/cam12-35.pdf
