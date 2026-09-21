# LevyProcesses.jl

Lévy jump measures, retained-jump simulation and conditional transitions for linear SDEs.

```julia
using LevyProcesses, Random

p = GammaProcess(1.3, 2.0)
retained = TruncatedLevyProcess(p; l=1e-3)
jumps = sample(MersenneTwister(42), retained, 0.5)
```

## Conventions

- `levy_tail_mass(p, x)` is the intensity of jumps with absolute size greater than `x`.
- `TruncatedLevyProcess(p; l, u)` retains nonzero jumps with `l <= abs(x) <= u`. Sampling requires finite retained intensity and returns jump times in `[0, dt]` and their sizes, without adding drift.
- `levy_drift` uses the Lévy–Khintchine truncation function `h(x) = x * (abs(x) <= 1)`. It is not generally the drift to add to a raw jump sum.
- `StableProcess(α, β, σ)` uses the zero-location `StableDistributions.Stable` convention, with `0 < α < 2`, `α != 1`. `StableSubordinator` requires `0 < α < 1`.
- Truncating a mixture's latent subordinator and truncating its physical jumps are different approximations. Gaussian conditional transitions require an explicit latent mixture representation.

The qualified functions `LevyProcesses.log_normalised_sample_jumps_density` and
`log_unnormalised_sample_jumps_density` use a count and labelled time/mark pairs.
Pass `sorted=true` for a density with respect to chronologically ordered times.
The normalised density includes the Poisson intensity term, which is needed when
inferring process parameters.

## Current scope

Compensation and residual approximation policies are being developed separately;
`approximate_residual=true` currently raises an error. Conditional Gaussian
transition construction still adds the existing covariance jitter; use
`conditional_marginal_parameters` to obtain the unmodified jump contribution.
The stable–Gaussian convolution and variance-gamma marginal density routines
remain research approximations with limited numerical validation.

The experimental `HittingTime` residual sampler remains unvalidated for general
scale and cutoff parameters (issue #3).
