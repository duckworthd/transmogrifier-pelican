Title: Recent Advances in Stochastic Methods
Date: 3000-01-01 00:00
Category: optimization
Tags: optimization, stochastic
Slug: recent-advances-in-stochastic-methods

In the beginning, there was Gradient Descent, and lo, the optimization gods
looked upon it, and saw that it was good. Then the gods created Newton's
Method, and lo, it was even greater. But soon came the days of supervised
learning and [empirical risk minimization][empirical-risk-minimization], where
even evaluating the objective function required computation _linear_ in the
amount of data available. Thus Stochastic Gradient Descent (SGD) was borne,
a computationally advantageous alternative to Gradient Descent with
per-iteration computation _constant_ with respect to the amount of data.

And lo, this method too was good. In the data-rich setting where generalization
error (not optimization error!) is the goal it can even be [vastly
superior][bottou-2011] to its older sibling, Gradient Descent.

Yet something was amiss. In practice, SGD obtained "good" solutions very
quickly but struggled to reach high accuracy ones. One of the most scalable
open source systems for empirical risk minimization, [Vowpal
Wabbit][vowpwal-wabbit], limits the use of SGD to its initial phase precisely
for this reason. Theory, too, verified what was observed. For strongly convex
objectives, SGD requires $O(1/\epsilon)$ iterations to achieve an
$\epsilon$-accurate solution; Gradient Descent, on the other hand, only
requires $O(\log 1 / \epsilon)$.

For many years, this rift remained. Gradient Descent, the heavy weight,
theoretically fast ideal; Stochastic Gradient Descent, its nimble but
theoretically crippled counterpart. Was there not some way to marry the two? To
obtain the benefits of both?

In 2012, all that changed. Two methods, [Stochastic Dual Coordinate Ascent
(SDCA)][shalev-schwartz-2012] and [Stochastic Average Gradient
(SAG)][le-roux-2012], showed that we can have our cake and eat it too. Since
then, "Incremental Optimization" has matured into a full-fledged research topic
of its own, resulting in tens of methods with different assumptions,
guarantees, and variations.

In the following, I'll do my best to summarize these methods and their
successors. What do they assume? What do they guarantee? Why do they work?
*TODO: Add conclusion sentence*

# The Finite-Sum Perspective


[empirical-risk-minimization]: http://en.m.wikipedia.org/wiki/Empirical_risk_minimization#Empirical_risk_minimization
[bottou-2011]: http://leon.bottou.org/publications/pdf/mloptbook-2011.pdf
[vowpwal-wabbit]: https://github.com/JohnLangford/vowpal_wabbit
[shalev-schwartz-2012]: http://arxiv.org/abs/1209.1873
[le-roux-2012]: http://arxiv.org/abs/1202.6258
[schmidt-2013]: http://arxiv.org/abs/1309.2388
