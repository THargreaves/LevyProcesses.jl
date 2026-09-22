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
- `StableProcess(α, β, σ, μ=0)` uses Nolan S0 coordinates for its unit-time marginal, with `0 < α < 2`, including α=1. `StableSubordinator` still requires `0 < α < 1`.
- Truncating a mixture's latent subordinator and truncating its physical jumps are different approximations. Gaussian conditional transitions require an explicit latent mixture representation.

The qualified functions `LevyProcesses.log_normalised_sample_jumps_density` and
`log_unnormalised_sample_jumps_density` use a count and labelled time/mark pairs.
Pass `sorted=true` for a density with respect to chronologically ordered times.
The normalised density includes the Poisson intensity term, which is needed when
inferring process parameters.

## Stable coordinates

S0 keeps the law continuous as α crosses one, making it preferable for parameter
inference ([Pyro's convention](https://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.Stable)).
Its location `μ` is generally neither the expectation nor the physical drift.
`marginal(p, dt)` returns a `StableS0` distribution with the correct increment
location; simply multiplying an S0 location by `dt` would change the process.

Use `from_s1(α, β, σ, μ1)` to preserve a model specified using the previous S1
coordinates, and `to_s1(p)` for a unit-time `StableDistributions.Stable` adapter.
Zero S0 and zero S1 locations generally describe different skewed processes.
S1 conversion becomes ill-conditioned near one; the S0 characteristic function,
sampling and canonical drift use continuous formulas directly. Density and CDF
evaluation near one use numerical integration.

`to_nsm` preserves the S0 location using an explicit deterministic `drift` in the
returned mixture. Preserve this field when truncating its latent subordinator.
This positive-subordinator representation remains restricted to α<1; compensated
Gaussian-mark representations across one are separate work.

## Current scope

Compensation and residual approximation policies are being developed separately;
`approximate_residual=true` currently raises an error. Conditional Gaussian
transition construction still adds the existing covariance jitter; use
`conditional_marginal_parameters` to obtain the unmodified jump contribution.
The S1 stable–Gaussian convolution series and variance-gamma marginal density routines
remain research approximations with limited numerical validation.

The experimental `HittingTime` residual sampler remains unvalidated for general
scale and cutoff parameters (issue #3).
