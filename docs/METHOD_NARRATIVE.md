# Method narrative

This document is meant to become the seed for slides. The goal is to motivate the mathematics and the computational story in small, intuitive steps.

## 1. What are we trying to learn?

We want to learn a **preference function** over poems.

Each poem is represented by an embedding vector, and the user provides a small number of ratings. The hidden object we care about is something like

\[
f(x) = \text{how much this user likes the poem with embedding } x.
\]

We never observe the full function directly. We only observe a few noisy ratings at a few points.

That immediately suggests a regression problem.

## 2. First attempt: ordinary linear regression

A natural first idea is linear regression:

\[
f(x) \approx \beta_0 + x^T \beta.
\]

Why it is attractive:

- simple
- fast
- easy to explain
- easy to fit repeatedly

Why it is not enough for this project:

- it gives a very restrictive functional form
- it mainly supports exploitation, not principled exploration
- it does not naturally express **epistemic uncertainty** about unobserved poems

For a recommender that wants to choose between “show me the poem I think you will like most” and “show me the poem that would teach me the most,” uncertainty matters.

## 3. Second attempt: Bayesian linear regression

Bayesian linear regression improves on this by placing a prior on the coefficients.

That gives us uncertainty in predictions, which is already a major step forward.

But it is still restrictive in an important way: the model is still linear in the chosen features.

So now the bottleneck becomes feature design.

## 4. Could we just engineer more features?

Yes, but then awkward questions appear:

- which hand-engineered features should we add?
- how many interaction terms should we include?
- what nonlinearities matter?
- how do we know when we have engineered enough?

This creates a tension:

- simple features are easy to fit but may underfit
- richer features may help, but they increase design burden and can become arbitrary

## 5. Could we learn features with a neural network instead?

Possibly, but for this setting that creates a different problem.

Neural networks can be very flexible, but standard point-estimate neural nets do **not** naturally provide the kind of calibrated epistemic uncertainty we want for active exploration. There are extensions such as ensembles, Bayesian neural nets, MC dropout, Laplace approximations, and conformal methods, but those introduce extra machinery and pedagogy.

For this project, we want a model that is:

- flexible
- uncertainty-aware
- mathematically interpretable
- and still compact enough to explain clearly

## 6. Enter Gaussian Process Regression

A Gaussian process lets us place a prior directly over functions:

\[
f \sim \mathcal{GP}(m, k).
\]

This means that before seeing any ratings, we specify beliefs about what kinds of functions are plausible. The key modeling object is the kernel

\[
k(x, x').
\]

The kernel says how similar two inputs are, and therefore how strongly their function values should co-vary.

## 7. Why kernels help

The kernel viewpoint is powerful because it lets us model nonlinear relationships without explicitly writing down a huge set of basis functions.

For many kernels, there is an equivalent interpretation as an inner product in a very high-dimensional, sometimes infinite-dimensional, feature space. So instead of manually inventing a giant basis expansion, we specify a kernel and let that determine the induced function class.

Pedagogically, this gives us a nice story:

- linear regression chooses a finite, explicit feature map
- Gaussian processes let us work with a much richer implicit feature space
- the core linear algebra depends primarily on the number of observed rated poems, not the number of explicit basis functions we would otherwise have to build

That last point is one reason the GP story fits this project well.

## 8. Why the RBF kernel is a good teaching choice

For the main pedagogical example, the RBF kernel is the cleanest starting point:

\[
k(x,x') = \sigma_f^2 \exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right).
\]

Students can interpret its hyperparameters immediately:

- \(\sigma_f^2\): vertical scale of variation
- \(\ell\): length scale controlling how quickly similarity decays with distance
- plus a noise variance in the observation model

This makes it easy to explain the prior in plain language:

- nearby poems in embedding space should have similar preference values
- far-away poems should be less correlated
- smaller length scales allow wigglier functions
- larger length scales enforce smoother variation

## 9. But aren't kernels a strong prior?

Yes. That is not a flaw; it is the model.

A Gaussian process does not “discover any function whatsoever.” It expresses uncertainty over a structured family of functions. The kernel is where those assumptions live.

This is a useful teaching moment:

- all models impose structure
- linear regression imposes one kind of structure
- a GP kernel imposes another
- the RBF kernel is strong, but easy to understand

## 10. How do we choose the hyperparameters?

For fixed kernel hyperparameters, GP posterior prediction is available in closed form.

But the hyperparameters themselves are usually chosen by maximizing the **log marginal likelihood**:

\[
\log p(y \mid X, \theta).
\]

This gives another appealing narrative step:

1. choose a kernel family
2. choose hyperparameters by evidence optimization
3. then condition on the observed ratings to get posterior means and variances

This is also where the computation becomes expensive, because each marginal-likelihood evaluation requires rebuilding the kernel matrix and refactorizing it.

## 11. Why GPs are especially attractive for active learning

The GP gives us two quantities we care about at each candidate poem:

- posterior mean: what do we expect the user to like?
- posterior variance: where are we still uncertain?

That gives a natural exploit/explore split:

- **exploit**: choose a poem with high posterior mean
- **explore**: choose a poem with high posterior uncertainty

This is much harder to motivate cleanly with ordinary linear regression, and much more natural with a GP.

## 12. Where the computational bottleneck appears

There are really two heavy parts.

### 12.1 Training / refitting the GP

If the user has rated \(N\) poems, exact GP fitting requires factorizing an \(N \times N\) kernel matrix.

That costs roughly

\[
O(N^3)
\]

for the main dense linear algebra step.

If we also optimize hyperparameters by marginal likelihood, we repeat that solve many times.

### 12.2 Scoring the full catalog

Now suppose there are \(M\) candidate poems remaining.

To recommend the next poem, we may want posterior means and variances for **all** candidates. The variance computation across the full catalog can be especially expensive, and for large candidate pools it can dominate the fit step.

That is what makes this a strong HPC example: we are not just fitting one GP once; we are repeatedly refitting and scoring a large catalog under uncertainty.

## 13. Why this becomes an HPC story

Once the mathematical model is clear, the computational story becomes natural.

A naive implementation is easy to write, but expensive because it repeatedly performs:

- dense kernel construction
- Cholesky factorization
- triangular solves
- large cross-kernel evaluations against the candidate set
- posterior variance calculations across the whole catalog

That sets up an implementation ladder:

1. naive serial exact GP
2. blocked / vectorized exact GP
3. MPI-distributed candidate scoring
4. GPU acceleration or approximate methods as future work

Now the optimization techniques are motivated by a real modeling need, not by an arbitrary toy benchmark.

## 14. The clean storyline for slides

A compact slide narrative could be:

1. We want to learn a user’s hidden preference function over poem space.
2. Linear regression is simple, but too rigid and does not naturally support exploration.
3. Bayesian linear regression gives uncertainty, but is still tied to explicit chosen features.
4. Feature engineering or neural nets increase flexibility, but also increase modeling and uncertainty-estimation complexity.
5. Gaussian processes give a distribution over nonlinear functions directly.
6. The kernel encodes similarity and the prior over functions.
7. The GP posterior gives both expected preference and uncertainty, which supports exploit/explore decisions.
8. Exact GP inference plus full-catalog uncertainty scoring creates a real computational bottleneck.
9. That bottleneck motivates vectorization, distributed scoring, and HPC methods.

## 15. Important nuance for teaching

A few statements should be kept careful and honest:

- standard neural nets do not naturally provide calibrated epistemic uncertainty, but there are important extensions that do
- the kernel does not magically eliminate modeling assumptions; it replaces explicit feature engineering with structured prior design
- exact GPs scale mainly with the number of observed ratings, but kernel evaluation still depends on the embedding dimension
- the RBF kernel is a pedagogical choice, not a claim that it is always the best practical kernel

Those caveats help keep the story accurate without making it too heavy.
