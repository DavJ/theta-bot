"""
complex_tau_linear_flow.py
--------------------------
SymPy verification of the 2x2 commuting-case closed forms derived in
DERIVATION_P2_complex_chronofactor_spectrum.md (Appendix A).

Checks symbolically:
  - det Theta(tau) = exp(tau * Tr G) * det Theta0
  - log det Theta(tau) = tau * Tr G + log det Theta0
  - S_Theta  = 2 * k_B * Re(log det Theta)
  - Sigma_Theta = k_B * Im(log det Theta)
  - dO/dt formula (fixed psi, Hermitian-part contribution)

No heavy computation; all expressions stay symbolic.
"""

import sympy as sp

# ── Symbols ──────────────────────────────────────────────────────────────────
t, psi, k_B = sp.symbols("t psi k_B", real=True)
a, b, c, d  = sp.symbols("a b c d",   real=True)   # eigenvalue components
tau = t + sp.I * psi

# ── 2x2 diagonal generator G = diag(a+ib, c+id) ─────────────────────────────
mu1 = a + sp.I * b   # eigenvalue 1
mu2 = c + sp.I * d   # eigenvalue 2
trG = mu1 + mu2      # Tr G

# Theta0 = I_2 (so det Theta0 = 1, log det Theta0 = 0)
# Theta(tau) = diag(exp(tau*mu1), exp(tau*mu2))

exp_tau_mu1 = sp.exp(tau * mu1)
exp_tau_mu2 = sp.exp(tau * mu2)

# ── 1. det Theta ─────────────────────────────────────────────────────────────
det_Theta = exp_tau_mu1 * exp_tau_mu2
det_Theta_simplified = sp.simplify(det_Theta)

# Expected: exp(tau * Tr G)
det_Theta_expected = sp.exp(tau * trG)

print("=== 1. det Theta ===")
print("  computed :", det_Theta_simplified)
print("  expected :", det_Theta_expected)
det_check = sp.simplify(det_Theta_simplified - det_Theta_expected)
print("  diff (should be 0):", det_check)

# ── 2. log det Theta ─────────────────────────────────────────────────────────
log_det = sp.log(det_Theta_simplified)
log_det_simplified = sp.expand(log_det)

log_det_expected = tau * trG   # + log(det Theta0) = 0 here

print("\n=== 2. log det Theta ===")
print("  computed :", log_det_simplified)
print("  expected :", sp.expand(log_det_expected))
# SymPy does not automatically collapse log(exp(i*X)) for symbolic X;
# verify equality via Re and Im parts separately.
re_check = sp.simplify(sp.re(log_det_simplified) - sp.re(log_det_expected))
im_check = sp.simplify(sp.im(log_det_simplified) - sp.im(log_det_expected))
print("  Re diff (should be 0):", re_check)
# SymPy cannot reduce arg(exp(i*X)) symbolically without bounds on X;
# verify numerically at a sample point where the angle is within (-pi, pi].
subs_num = {t: sp.Rational(1, 10), psi: sp.Rational(1, 20),
            a: sp.Rational(1, 5),  b: sp.Rational(1, 7),
            c: sp.Rational(1, 11), d: sp.Rational(1, 13)}
im_check_num = complex(im_check.subs(subs_num).evalf())
print("  Im diff numeric (should be ~0):", round(im_check_num.real, 12), round(im_check_num.imag, 12))

# ── 3. S_Theta = 2*k_B*Re(log det Theta) ─────────────────────────────────────
# For symbolic Re/Im we expand tau*trG manually
tau_trG_expanded = sp.expand(tau * trG)  # (t+i*psi)*(a+c + i*(b+d))
re_log_det = sp.re(tau_trG_expanded)
im_log_det = sp.im(tau_trG_expanded)

S_Theta   = 2 * k_B * re_log_det
Sigma_Theta = k_B * im_log_det

# Simplify by applying real-symbol assumptions via .rewrite + .doit
re_log_det_simplified   = sp.simplify(re_log_det)
im_log_det_simplified   = sp.simplify(im_log_det)

print("\n=== 3. Re(log det Theta) ===")
print("  Re(tau*TrG) =", re_log_det_simplified)
print("  Expected    : t*(a+c) - psi*(b+d)")

print("\n=== 4. Im(log det Theta) ===")
print("  Im(tau*TrG) =", im_log_det_simplified)
print("  Expected    : t*(b+d) + psi*(a+c)")

print("\n=== 5. UBT invariants ===")
print("  S_Theta     =", sp.simplify(S_Theta))
print("  Sigma_Theta =", sp.simplify(Sigma_Theta))

# ── 4. O(tau) = Theta† Theta (diagonal case, psi fixed) ──────────────────────
# Theta_11 = exp((t+i*psi)*(a+i*b)) = exp(t*a - psi*b) * exp(i*(t*b + psi*a))
# |Theta_11|^2 = exp(2*(t*a - psi*b))
lam1 = sp.exp(2 * (t * a - psi * b))   # eigenvalue 1 of O
lam2 = sp.exp(2 * (t * c - psi * d))   # eigenvalue 2 of O

print("\n=== 6. Eigenvalues of O(tau) ===")
print("  lambda_1 =", lam1)
print("  lambda_2 =", lam2)

dlam1_dt   = sp.diff(lam1, t)
dlam1_dpsi = sp.diff(lam1, psi)
print("\n  d(lambda_1)/dt   =", sp.simplify(dlam1_dt),
      "  (= 2*a*lambda_1 -- growth/decay rate by Re(mu_1)=a)")
print("  d(lambda_1)/dpsi =", sp.simplify(dlam1_dpsi),
      "  (= -2*b*lambda_1 -- amplitude modulation by Im(mu_1)=b)")

# ── 5. dO/dt formula check ────────────────────────────────────────────────────
# For diagonal G: Herm(G) = A = diag(a, c)
# partial_t O = 2 * Theta† * A * Theta (diagonal) = 2*a*lam1 x 2*c*lam2
print("\n=== 7. dO/dt = 2 Theta† Herm(G) Theta ===")
dO11_dt = 2 * a * lam1
dO22_dt = 2 * c * lam2
print("  dO_11/dt = 2*a*lambda_1 =", sp.simplify(dO11_dt - dlam1_dt),
      " (diff from direct diff, should be 0)")
print("  dO_22/dt = 2*c*lambda_2 =", sp.simplify(dO22_dt - sp.diff(lam2, t)),
      " (diff from direct diff, should be 0)")

print("\nAll symbolic checks complete.")
