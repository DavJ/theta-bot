# UBT Core Geometry: Biquaternion-First Formulation

## Overview

This document presents the **core geometric refactoring** of Unified Biquaternion Theory (UBT), establishing biquaternion structures as fundamental with General Relativity emerging as a real projection.

## Key Principles

### 1. Biquaternion Metric is Fundamental

The classical metric tensor g_{μν} is **NOT** fundamental. Instead:

```
g_{μν} := Re(𝓖_{μν})
```

where 𝓖_{μν} ∈ 𝔹 is the fundamental biquaternion metric.

### 2. Tetrad Formalism is Mandatory

The metric must be defined through the biquaternion tetrad:

```
E_μ(x) ∈ 𝔹
𝓖_{μν} := Sc(E_μ E_ν†)
```

Direct introduction of g_{μν} is **forbidden**.

### 3. Biquaternion Connection Replaces Christoffel Symbols

Christoffel symbols Γ^λ_{μν} are derived, not fundamental. The fundamental object is:

```
Ω_μ(x) ∈ 𝔹
```

### 4. All Geometric Objects are Biquaternionic

- **Curvature**: 𝓡_{μν} = ∂_μ Ω_ν - ∂_ν Ω_μ + [Ω_μ, Ω_ν]
- **Ricci tensor**: 𝓡_{νσ} = E^μ 𝓡_{μν} E_σ  
- **Stress-energy**: 𝓣_{μν} = ⟨D_μ Θ, D_ν Θ⟩_𝔹 - ½ 𝓖_{μν} ⟨DΘ, DΘ⟩

Only **after** defining these can we take real projections:

```
R_{μν} := Re(𝓡_{μν})
T_{μν} := Re(𝓣_{μν})
```

### 5. Field Equations

The fundamental field equation is:

```
𝓖_{μν} = κ 𝓣_{μν}
```

Einstein's equations emerge as:

```
Re(𝓖_{μν}) = κ Re(𝓣_{μν})  →  G_{μν} = κ T_{μν}
```

## Exotic Regimes

Solutions with Im(𝓖_{μν}) ≠ 0 are:

- **Physically consistent** within UBT
- **Unobservable** in standard GR
- Responsible for:
  - Pseudo-antigravitational behavior
  - Phase invisibility
  - Local temporal drift
  - Modified causal structure

## Meta-Commentary

**General Relativity arises as the real, commutative projection of fundamental biquaternion spacetime geometry.**

Apparent violations (antigravitation, causal drift) correspond to **non-real sectors** of the metric and curvature, not exotic matter or energy violations.

## Prohibitions

The following are **strictly forbidden**:

1. Using GR as an axiom
2. Simplifying biquaternions to complex numbers
3. Breaking global causality
4. Identifying observable world with fundamental reality
5. Introducing energy ex nihilo

## Compilation

To compile the LaTeX document:

```bash
cd papers/ubt_core_geometry
pdflatex ubt_biquaternion_geometry.tex
```

## References

This formulation supersedes any previous UBT formulations that treated g_{μν} as fundamental.

## Relation to Market Applications

This is a **pure physics/mathematics** formulation. Market applications (as in `papers/ubt_tensor_markets/`) use UBT-*inspired* mathematics but do **NOT** claim markets are physical spacetime. See that paper for clarification of the domain distinction.
