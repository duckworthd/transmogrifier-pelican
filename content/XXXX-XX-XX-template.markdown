<!--
Title: ADMM, revisited
Date: 2013-07-06 00:00
Category: optimization
Tags: optimization, distributed, admm
Slug: admm-revisited
-->

<div class="pseudocode" markdown>
</div>


<a name="implementation" href="#implementation">How does it work?</a>
=====================================================================


<div class="img-center">
  <img src="/assets/img/frank_wolfe/animation.gif"></img>
  <span class="caption">
  </span>
</div>


<a name="proof" href="#proof">Why does it work?</a>
===================================================


  **Assumptions**

  **Proof Outline**

  **Step 1**

$$
\begin{align*}
  f(x^{+})
  & \le f(x) + \nabla f(x)^T (x^{+} - x) + \frac{\alpha^2}{2} C_f \\
  & = f(x) + \nabla f(x)^T ( (1-\alpha) x + \alpha s^{+} - x ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) + \nabla f(x)^T ( \alpha s^{+} - \alpha x ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) - \alpha \nabla f(x)^T ( x - s^{+} ) + \frac{\alpha^2}{2} C_f \\
  & = f(x) - \alpha g(x) + \frac{\alpha^2}{2} C_f \\
\end{align*}
$$


<a name="usage" href="#usage">When should I use it?</a>
=======================================================

<a name="extensions" href="#extensions">Extensions</a>
======================================================

<a name="references" href="#references">References</a>
======================================================

[frank_wolfe]: http://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
[proximal_gradient]: {filename}/2013-04-19-proximal-gradient.markdown

<a name="reference-impl" href="#reference-impl">Reference Implementation</a>
============================================================================

```python
```
