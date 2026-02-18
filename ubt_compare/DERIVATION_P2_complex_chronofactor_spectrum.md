# Symbolic Derivation: Complex Chronofactor τ = t + iψ — Spectral Consequences and Links to S_Θ and Σ_Θ

> **Status:** Paper-grade symbolic derivation.  No numerical simulation required.
> **Version:** P2

---

## 1. Setup and Assumptions

### 1.1 The chronofactor

Let the *chronofactor* be

$$\tau = t + i\psi, \qquad t, \psi \in \mathbb{R},$$

where $t$ is the physical (real) time and $\psi$ is the imaginary displacement.

### 1.2 The matrix field Θ

Let $\Theta(\tau)$ be an $n \times n$ **complex matrix field** depending analytically on $\tau$.
Define the *overlap matrix*

$$O(\tau) = \Theta^\dagger(\tau)\,\Theta(\tau),$$

which is positive semidefinite on the real-time slice $\tau = t$ (i.e. $\psi = 0$).

### 1.3 UBT invariants

The two primary invariants of interest are:

| Symbol | Definition | Interpretation |
|--------|-----------|----------------|
| $S_\Theta$ | $k_B \log \det(\Theta^\dagger \Theta)$ | Logarithmic amplitude entropy |
| $\Sigma_\Theta$ | $k_B \arg \det \Theta$ | Phase / holonomy channel |

Here $k_B$ is a positive dimensional constant (Boltzmann-like).

### 1.4 The generator

Let $G$ be an $n \times n$ complex matrix (the *generator*).  Decompose it as

$$G = A + iB, \qquad A, B \in M_n(\mathbb{R}),$$

where $A = \operatorname{Herm}(G) = \tfrac{1}{2}(G + G^\dagger)$ is the Hermitian part and $iB$ with $B$ real-symmetric
is the anti-Hermitian part.  *(More generally $G$ is arbitrary complex; $A$ and $B$ are the
uniquely defined matrices $A = \tfrac{1}{2}(G+G^\dagger)$ and $B = \tfrac{1}{2i}(G-G^\dagger)$.)*

### 1.5 Analyticity assumption

We assume $\Theta(\tau)$ is **holomorphic** in a strip $|\psi| < \psi_{\max}$ so that $\partial_{\bar\tau}\Theta = 0$,
and thus $\partial_t = \partial_\tau$ and $\partial_\psi = i\partial_\tau$ on holomorphic functions.

---

## 2. Linear Complex-τ Flow: ∂_τ Θ = GΘ

**Definition (Linear Flow).**  A *linear complex-τ flow* is the ODE

$$\boxed{\partial_\tau \Theta(\tau) = G\,\Theta(\tau),}$$

with initial condition $\Theta(0) = \Theta_0$.

Because $G$ is $\tau$-independent, this has a unique global solution.

---

## 3. Solution Θ(τ) = exp(τG)Θ₀ and Spectral Decomposition of G

**Proposition 3.1 (Exact solution).**
The unique holomorphic solution to $\partial_\tau\Theta = G\Theta$ with $\Theta(0)=\Theta_0$ is

$$\boxed{\Theta(\tau) = e^{\tau G}\,\Theta_0.}$$

*Proof.* Define $F(\tau) = e^{\tau G}\Theta_0$.  Then $\partial_\tau F = G e^{\tau G}\Theta_0 = G F$, and $F(0)=\Theta_0$. Uniqueness follows from the Picard–Lindelöf theorem applied to the linear system. $\square$

### 3.1 Spectral decomposition of G

Assume $G$ is **diagonalizable** (the non-diagonalizable case is treated in Section 8).  Write

$$G = V\,\operatorname{diag}(\mu_1,\ldots,\mu_n)\,V^{-1},$$

where $\mu_k = \alpha_k + i\omega_k$ are the (generally complex) eigenvalues of $G$, and $V$ is the matrix of right eigenvectors:

$$G v_k = \mu_k\,v_k, \qquad \mu_k = \alpha_k + i\omega_k, \quad \alpha_k, \omega_k \in \mathbb{R}.$$

Then

$$e^{\tau G} = V\,\operatorname{diag}(e^{\tau\mu_1},\ldots,e^{\tau\mu_n})\,V^{-1}.$$

---

## 4. What Changes When τ Is Complex: Growth/Decay + Oscillation

Setting $\tau = t + i\psi$ and $\mu_k = \alpha_k + i\omega_k$:

$$\boxed{e^{\tau\mu_k} = e^{t\alpha_k - \psi\omega_k}\cdot e^{i(t\omega_k + \psi\alpha_k)}.}$$

**Interpretation:**

- **Magnitude:** $|e^{\tau\mu_k}| = e^{t\alpha_k - \psi\omega_k}$.
  - Real time $t$ drives growth ($\alpha_k > 0$) or decay ($\alpha_k < 0$) at rate $\alpha_k$.
  - Imaginary displacement $\psi$ modulates the magnitude through the imaginary part $\omega_k$ of the eigenvalue.
- **Phase:** $\arg e^{\tau\mu_k} = t\omega_k + \psi\alpha_k$.
  - Real time $t$ drives oscillation at frequency $\omega_k$.
  - Complex τ couples oscillation and growth: $\psi$ contributes to the phase through $\alpha_k$, while $t$ contributes to the magnitude through a term that $\psi$ can amplify via $\omega_k$.

**Key observation:** When $\psi \neq 0$, the imaginary part of each eigenvalue directly controls amplitude, not just phase.  Conversely, the real part of each eigenvalue contributes to the phase angle.  This *coupling* between the growth/decay channel and the oscillation channel is the central spectral signature of a complex chronofactor.

### 4.1 Commuting vs. non-commuting generators

Let $G = A + iB$.  Then

$$e^{\tau G} = e^{(t+i\psi)(A+iB)} = e^{(tA - \psi B) + i(tB + \psi A)}.$$

**Case 1: $[A, B] = 0$.**  The Baker–Campbell–Hausdorff (BCH) formula reduces to the additive exponent:

$$e^{\tau G} = e^{tA - \psi B}\cdot e^{i(tB + \psi A)} \qquad ([A,B]=0).$$

This is the *commuting decomposition*.

**Case 2: $[A, B] \neq 0$.**  BCH gives

$$e^{X}e^{Y} = e^{X + Y + \frac{1}{2}[X,Y] + \frac{1}{12}([X,[X,Y]] - [Y,[X,Y]]) + \cdots},$$

where $X = tA - \psi B$ and $Y = i(tB + \psi A)$.  To first order in $[A,B]$:

$$e^{\tau G} = e^{(tA-\psi B) + i(tB+\psi A) + \frac{i}{2}[(tA-\psi B),\,(tB+\psi A)] + O([A,B]^2)}.$$

Expanding the commutator bracket:

$$[(tA-\psi B),\,(tB+\psi A)] = t^2[A,B] - \psi^2[B,A] + t\psi([A,A] - [B,B]) = (t^2+\psi^2)[A,B].$$

Hence the **BCH first-order correction** is:

$$\boxed{e^{\tau G} = e^{(tA-\psi B)+i(tB+\psi A)}\cdot\exp\!\left(\tfrac{i}{2}|\tau|^2[A,B] + O([A,B]^2)\right),}$$

where $|\tau|^2 = t^2+\psi^2$.  The correction vanishes for normal generators ($[A,B]=0$) and grows with $|\tau|^2$ for non-commuting ones.

---

## 5. Consequences for O(τ) = Θ†Θ on the Real-Time Slice

### 5.1 Evolution of O under real t (ψ fixed)

On the slice where $\psi$ is held fixed, differentiate $O = \Theta^\dagger\Theta$ with respect to $t$:

$$\partial_t O = (\partial_t\Theta^\dagger)\Theta + \Theta^\dagger(\partial_t\Theta).$$

Since $\partial_t = \partial_\tau$ for holomorphic $\Theta$:

$$\partial_t\Theta = G\Theta, \qquad \partial_t\Theta^\dagger = (G\Theta)^\dagger = \Theta^\dagger G^\dagger.$$

Therefore

$$\boxed{\partial_t O = \Theta^\dagger G^\dagger \Theta + \Theta^\dagger G\Theta = \Theta^\dagger(G^\dagger + G)\Theta = 2\,\Theta^\dagger\operatorname{Herm}(G)\,\Theta.}$$

**Corollary 5.1.** If $G$ is **anti-Hermitian** ($G^\dagger = -G$, i.e. $\operatorname{Herm}(G) = A = 0$), then $\partial_t O = 0$ on every real-time slice with $\psi$ fixed.  The eigenvalues of $O$ are conserved; only the phase of $\Theta$ evolves.

**Corollary 5.2.** When $A = \operatorname{Herm}(G) \neq 0$, the eigenvalue spectrum of $O$ drifts monotonically (in the direction determined by the sign of the eigenvalues of $A$).

### 5.2 Additional drift when ψ = ψ(t) is time-dependent

If $\psi = \psi(t)$, then $\tau(t) = t + i\psi(t)$ and the chain rule gives

$$\frac{d}{dt}\Theta(\tau(t)) = \partial_t\Theta + \psi'(t)\,i\,\partial_\tau\Theta = (1 + i\psi'(t))\,G\Theta.$$

Let $\tilde{G}(t) = (1+i\psi'(t))G$ be the **effective generator**.  Then

$$\frac{d}{dt}O = \Theta^\dagger\bigl(\tilde{G}^\dagger + \tilde{G}\bigr)\Theta = 2\,\Theta^\dagger \operatorname{Herm}\!\bigl((1+i\psi')G\bigr)\Theta.$$

Computing:

$$\operatorname{Herm}\bigl((1+i\psi')G\bigr) = A - \psi' B + i\psi' A \cdot 0 = A - \psi' B,$$

(since the Hermitian part of $(1+i\psi')(A+iB) = (A - \psi'B) + i(\psi'A + B)$ is $A - \psi'B$).

$$\boxed{\frac{d}{dt}O\Big|_{\psi=\psi(t)} = 2\,\Theta^\dagger\bigl(A - \psi'(t)B\bigr)\Theta.}$$

**Conclusion:** A non-zero $\psi'(t)$ introduces an additional term $-2\psi'(t)\Theta^\dagger B\Theta$ into the eigenvalue drift of $O$.  Even if $A=0$ (anti-Hermitian generator), a time-varying $\psi$ causes *spectral broadening* through the $B$ component of $G$, i.e., through the anti-Hermitian part of $G$ in a coupling set by $\psi'$.

---

## 6. Determinant Channel: log det Θ and log det(Θ†Θ)

**Proposition 6.1 (log det formula).**  Under the linear flow $\Theta(\tau) = e^{\tau G}\Theta_0$:

$$\det\Theta(\tau) = \det(e^{\tau G})\cdot\det\Theta_0 = e^{\tau\operatorname{Tr}G}\cdot\det\Theta_0.$$

*Proof.* Use $\det(e^M) = e^{\operatorname{Tr}M}$ (Jacobi's formula / matrix determinant lemma), which holds for all square matrices $M$. $\square$

**Corollary 6.2 (log det identity).**

$$\boxed{\log\det\Theta(\tau) = \tau\operatorname{Tr}G + \log\det\Theta_0.}$$

Setting $\tau = t + i\psi$ and $\operatorname{Tr}G = \operatorname{Tr}A + i\operatorname{Tr}B$:

$$\log\det\Theta(\tau) = (t+i\psi)(\operatorname{Tr}A + i\operatorname{Tr}B) + \log\det\Theta_0.$$

Separating real and imaginary parts:

$$\operatorname{Re}\log\det\Theta = t\operatorname{Tr}A - \psi\operatorname{Tr}B + \operatorname{Re}\log\det\Theta_0,$$

$$\operatorname{Im}\log\det\Theta = t\operatorname{Tr}B + \psi\operatorname{Tr}A + \operatorname{Im}\log\det\Theta_0.$$

**The 2×Re relationship.**  On any slice where $\Theta^\dagger$ denotes the conjugate transpose evaluated at the *same* $\tau$:

$$\log\det(\Theta^\dagger\Theta) = \log\det\Theta^\dagger + \log\det\Theta = \overline{\log\det\Theta} + \log\det\Theta = 2\operatorname{Re}\log\det\Theta.$$

This identity $\log\det(\Theta^\dagger\Theta)=2\operatorname{Re}\log\det\Theta$ holds **precisely when** the logarithm is evaluated as a single-valued branch consistent on both $\Theta$ and $\Theta^\dagger$.  On the real-time slice ($\psi=0$, $\Theta^\dagger$ is the standard adjoint), this is exact.

---

## 7. Phase Channel: arg det Θ as Holonomy / Winding Under Im(τ)

**Definition 7.1.**  The *phase invariant* is

$$\Sigma_\Theta(\tau) = k_B\,\arg\det\Theta(\tau) = k_B\,\operatorname{Im}\log\det\Theta(\tau).$$

**Proposition 7.2 (Explicit formula for Σ_Θ).**  Under the linear flow:

$$\boxed{\Sigma_\Theta(\tau) = k_B\bigl(t\operatorname{Tr}B + \psi\operatorname{Tr}A + \operatorname{Im}\log\det\Theta_0\bigr).}$$

Here:
- $\operatorname{Tr}B = \operatorname{Im}\operatorname{Tr}G$ is the imaginary part of the trace.
- $\operatorname{Tr}A = \operatorname{Re}\operatorname{Tr}G$ is the real part of the trace.

**Interpretation.**  When $\tau$ is real ($\psi = 0$), $\Sigma_\Theta$ grows linearly in $t$ at rate $k_B\operatorname{Tr}B$.  When $\tau$ is complex ($\psi \neq 0$), the *real* part of the trace $\operatorname{Tr}A$ also contributes to the phase at rate $k_B\psi$.  This is the holonomy effect of the imaginary displacement.

**Definition 7.3 (Entropy invariant).**

$$S_\Theta(\tau) = k_B\log\det(\Theta^\dagger\Theta) = 2k_B\operatorname{Re}\log\det\Theta(\tau).$$

Explicitly:

$$\boxed{S_\Theta(\tau) = 2k_B\bigl(t\operatorname{Tr}A - \psi\operatorname{Tr}B + \operatorname{Re}\log\det\Theta_0\bigr).}$$

**Winding under Im(τ).**  Consider a loop $\psi: 0 \to \psi \to 0$ at fixed $t$.  The net change in $\Sigma_\Theta$ is zero (it returns to its initial value), consistent with single-valued $\Theta$ — unless a zero of $\det\Theta$ is enclosed in the $\psi$-contour, in which case there is a non-zero winding contribution (monodromy / branch point).

---

## 8. Non-Hermitian Case: Complex Eigenvalues, Bi-Orthogonal Modes, Pseudospectrum

### 8.1 Bi-orthogonal modes for non-Hermitian G

When $G$ is non-Hermitian, the left and right eigenvectors are distinct:

$$G v_k = \mu_k v_k, \qquad w_k^\dagger G = \mu_k w_k^\dagger, \qquad w_j^\dagger v_k = \delta_{jk} \text{ (biorthonormality)}.$$

The spectral expansion of $e^{\tau G}$ becomes

$$e^{\tau G} = \sum_k e^{\tau\mu_k}\,v_k\,w_k^\dagger.$$

### 8.2 Non-normal growth (pseudospectrum)

**Definition 8.1.**  The *$\epsilon$-pseudospectrum* of $G$ is

$$\Lambda_\epsilon(G) = \bigl\{z \in \mathbb{C} : \|(G - zI)^{-1}\| \geq \epsilon^{-1}\bigr\}.$$

**Key point:**  Even if all eigenvalues $\mu_k$ have $\operatorname{Re}\mu_k < 0$ (asymptotic stability), a non-normal $G$ can cause **transient growth** of $\|e^{\tau G}\|$ for finite $t$.  This transient growth is controlled by the geometry of the pseudospectrum, not the spectrum alone.

**Link to spectral broadening of O.**  Transient growth of $\|e^{\tau G}\|$ means the eigenvalues of $O(\tau) = \Theta^\dagger\Theta$ can increase and then decrease, even without a monotonic change of parameters.  This constitutes a *sudden broadening* of the eigenvalue distribution of $O$ that cannot be predicted from eigenvalues of $G$ alone — a signature of non-normality.

### 8.3 Jordan blocks (non-diagonalizable G)

If $G$ is not diagonalizable, the Jordan normal form introduces polynomial factors:

$$e^{\tau G}\Big|_{\text{Jordan block}} = e^{\tau\mu}\!\sum_{j=0}^{m-1}\frac{\tau^j}{j!}N^j,$$

where $N$ is the nilpotent part of the block.  This leads to polynomial-in-$\tau$ envelope factors multiplying the exponential, modifying the growth/decay rates and oscillation frequencies in a $\tau$-dependent way.

---

## 9. Diagnostic Invariants and Discriminators (A/B Test: τ Real vs. τ Complex)

The following three discriminators allow one to test, from observable quantities, whether the effective chronofactor has a non-zero imaginary component.

### D1: Phase–Entropy Coupling Coefficient

$$\boxed{C_{12}(t) = \frac{\partial_t\Sigma_\Theta}{\partial_t S_\Theta}.}$$

- **Prediction under real τ** ($\psi = 0$, $\psi'=0$):
  $\partial_t S_\Theta = 2k_B\operatorname{Tr}A$, $\partial_t\Sigma_\Theta = k_B\operatorname{Tr}B$, so
  $C_{12} = \operatorname{Tr}B / (2\operatorname{Tr}A)$ — a time-independent constant determined solely by the generator.

- **Prediction under complex τ** ($\psi = \psi(t)$ time-varying):
  $\partial_t S_\Theta = 2k_B(\operatorname{Tr}A - \psi'\operatorname{Tr}B)$ and $\partial_t\Sigma_\Theta = k_B(\operatorname{Tr}B + \psi'\operatorname{Tr}A)$, so
  $$C_{12}(t) = \frac{\operatorname{Tr}B + \psi'(t)\operatorname{Tr}A}{2(\operatorname{Tr}A - \psi'(t)\operatorname{Tr}B)},$$
  which is **time-varying** whenever $\psi'(t) \neq 0$.

**Test:** If $C_{12}(t)$ is observed to vary with $t$, this is evidence for a time-varying imaginary component of $\tau$.

### D2: Spectral Conservation Test

$$\boxed{\text{D2: } \lambda_i(O(t)) = \text{const in }t \iff \operatorname{Herm}(G_{\text{eff}}) = 0.}$$

- Under real τ with anti-Hermitian $G$ ($A=0$): eigenvalues of $O$ are exactly conserved.
- Under complex τ with $\psi'(t) \neq 0$: effective Hermitian part is $A - \psi'B$, so eigenvalues of $O$ drift at rate $2\Theta^\dagger(A-\psi'B)\Theta$ even when $A=0$, via the $\psi'B$ term.

**Test:** Observe eigenvalue drift of $O$ while simultaneously checking whether a pure-real generator would predict conservation.  Any drift attributable to the $B$ component of $G$ indicates non-zero $\psi'$.

### D3: Mode Pairing and Oscillatory Signatures in Σ_Θ with Monotonic S_Θ

For real generators $G \in M_n(\mathbb{R})$, complex eigenvalues come in conjugate pairs $\mu_k = \alpha \pm i\omega$.  Under complex τ:
- Each pair contributes $e^{t\alpha - \psi\omega}e^{i(t\omega + \psi\alpha)}$ and its conjugate.
- $S_\Theta$ depends on $\operatorname{Tr}A = \sum_k\alpha_k$ and is monotone if all $\alpha_k$ have the same sign.
- $\Sigma_\Theta$ oscillates with frequency $\operatorname{Tr}B / k_B$ in $t$.

$$\boxed{\text{D3: Oscillatory }\Sigma_\Theta(t)\text{ with monotone }S_\Theta(t) \iff \text{complex-conjugate eigenpairs with } \operatorname{Tr}B \neq 0.}}$$

**Test:** Compute the Fourier spectrum of $\Sigma_\Theta(t)$.  A peak at frequency $f_0 = \operatorname{Tr}B/(2\pi k_B)$ while $S_\Theta(t)$ is monotone is a signature of complex-τ dynamics with conjugate eigenpairs.

---

## Appendix A: 2×2 Worked Example (Closed Form)

Let $n=2$, $\Theta_0 = I_2$, and

$$G = \begin{pmatrix} a+ib & 0 \\ 0 & c+id \end{pmatrix}, \qquad a,b,c,d \in \mathbb{R} \text{ (diagonal, commuting case)}.$$

Then $A = \operatorname{diag}(a,c)$, $B = \operatorname{diag}(b,d)$, $[A,B]=0$.

**Solution:**

$$\Theta(\tau) = e^{\tau G} = \begin{pmatrix} e^{\tau(a+ib)} & 0 \\ 0 & e^{\tau(c+id)} \end{pmatrix}.$$

With $\tau = t + i\psi$:

$$\Theta(\tau) = \begin{pmatrix} e^{ta-\psi b}\,e^{i(tb+\psi a)} & 0 \\ 0 & e^{tc-\psi d}\,e^{i(td+\psi c)} \end{pmatrix}.$$

**det Θ:**

$$\det\Theta(\tau) = e^{\tau(a+ib)}e^{\tau(c+id)} = e^{\tau(a+c+i(b+d))} = e^{\tau\operatorname{Tr}G}.$$

**log det Θ:**

$$\log\det\Theta(\tau) = \tau\operatorname{Tr}G = (t+i\psi)\bigl((a+c) + i(b+d)\bigr).$$

Real and imaginary parts:

$$\operatorname{Re}\log\det\Theta = t(a+c) - \psi(b+d),$$

$$\operatorname{Im}\log\det\Theta = t(b+d) + \psi(a+c).$$

**UBT invariants:**

$$S_\Theta = 2k_B\bigl(t(a+c) - \psi(b+d)\bigr),$$

$$\Sigma_\Theta = k_B\bigl(t(b+d) + \psi(a+c)\bigr).$$

**O = Θ†Θ:**

$$O(\tau) = \begin{pmatrix} e^{2(ta-\psi b)} & 0 \\ 0 & e^{2(tc-\psi d)} \end{pmatrix}.$$

Eigenvalues: $\lambda_1 = e^{2(ta-\psi b)}$, $\lambda_2 = e^{2(tc-\psi d)}$.

$\partial_t\lambda_1 = 2a\lambda_1$: drift controlled by $a=\operatorname{Re}\mu_1$.

$\partial_\psi\lambda_1 = -2b\lambda_1$: imaginary displacement modulates amplitude through $b=\operatorname{Im}\mu_1$.  This confirms the coupling formula of Section 4.

---

## Appendix B: Relation to Heat-Kernel / Tr log Representation

The identity $\log\det\Theta = \operatorname{Tr}\log\Theta$ holds when the logarithm is defined on the appropriate branch.  For the linear flow:

$$\operatorname{Tr}\log\Theta(\tau) = \operatorname{Tr}\log(e^{\tau G}\Theta_0) = \tau\operatorname{Tr}G + \operatorname{Tr}\log\Theta_0.$$

The *heat-kernel representation* of $\operatorname{Tr}\log\Theta$ is:

$$\operatorname{Tr}\log\Theta = -\int_0^\infty \frac{ds}{s}\,\operatorname{Tr}(e^{-s\Theta} - e^{-s}).$$

For the linear flow, $\Theta(\tau) = e^{\tau G}\Theta_0$, this integral can be expressed as a Laplace transform of $\operatorname{Tr}(e^{-se^{\tau G}\Theta_0})$ — a well-defined object when $\Theta_0$ is positive definite.

The symbolic link is: as $\psi$ varies, the eigenvalues of $\Theta$ rotate in the complex plane, and the heat-kernel integrand acquires oscillatory contributions from the imaginary parts of eigenvalues.  These are the same oscillations that appear in $\Sigma_\Theta$ (Section 7).

---

## Appendix C: Regularization Near det Θ → 0 (Why log → −∞)

When one or more eigenvalues $\mu_k$ of $G$ satisfy $\operatorname{Re}(\tau\mu_k) \to -\infty$ (decay dominates), the corresponding mode amplitude $e^{\tau\mu_k} \to 0$, so $\det\Theta(\tau) \to 0$ and $\log\det\Theta \to -\infty$.

**Standard regularizations:**

1. **Tikhonov / spectral cutoff:** Replace $\det\Theta$ by $\det(\Theta + \epsilon I)$ for small $\epsilon > 0$, which shifts eigenvalues away from zero.  The log is then finite, with $\log\det(\Theta+\epsilon I) = \sum_k \log(\lambda_k + \epsilon)$.

2. **Zeta-function regularization:** Define $\log\det\Theta = -\zeta'_\Theta(0)$ where $\zeta_\Theta(s) = \operatorname{Tr}(\Theta^{-s})$.  This analytic continuation avoids the divergence but requires $\operatorname{Re}\mu_k > 0$.

3. **Conditional entropy:** Restrict $S_\Theta$ to the range of $\Theta$ (project out the kernel), which amounts to summing only over non-zero eigenvalues.

**Physical interpretation:** A divergence $S_\Theta \to -\infty$ signals that the matrix $\Theta$ is approaching a singular configuration — a degenerate subspace is collapsing.  Near such a point, the pseudospectrum (Section 8) typically expands dramatically, indicating high sensitivity to perturbations.

---

*End of derivation.*
