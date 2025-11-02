#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_transform_complex.py
-----------------------------------
Generates a complex theta-field Θ(q, τ, φ) with:
  τ = t + iψ   (complex time)
  φ = φ_r + iφ_i (complex phase)

Supports 3 projections:
  1. amplitude – |Θ|
  2. phase – arg(Θ) shown as color (HSV)
  3. toroidal – (Re(τ), Re(φ)) mapped as torus unfolding

Author: David Jaroš & GPT-5
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpmath import jtheta
from matplotlib.colors import hsv_to_rgb


def theta_field(q_vals, t_vals, psi=0.2, phi_r=0.0, phi_i=0.0):
    """Compute complex theta field Θ(q, τ, φ) over given domains."""
    Q, T = np.meshgrid(q_vals, t_vals)
    tau = T + 1j * psi
    phi = phi_r + 1j * phi_i

    Theta = np.zeros_like(Q, dtype=np.complex128)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            # Using Jacobi theta 3 function (periodic, modular invariant)
            val = complex(jtheta(3, np.pi * Q[i, j] + phi, np.exp(1j * np.pi * tau[i, j])))
            Theta[i, j] = val
    return Theta


def plot_projection(Theta, q_vals, t_vals, mode="amp", outfile="theta_projection.png"):
    """Visualize the theta field projection."""
    if mode == "amp":
        data = np.abs(Theta)
        plt.imshow(data, extent=[q_vals[0], q_vals[-1], t_vals[0], t_vals[-1]], aspect='auto', cmap="viridis")
        plt.title("|Θ(q, τ)| Amplitude Map")
        plt.colorbar(label="|Θ|")

    elif mode == "phase":
        phase = np.angle(Theta)
        norm_phase = (phase + np.pi) / (2 * np.pi)
        hsv_img = np.zeros((*phase.shape, 3))
        hsv_img[..., 0] = norm_phase
        hsv_img[..., 1] = 1.0
        hsv_img[..., 2] = np.abs(Theta) / np.max(np.abs(Theta))
        rgb_img = hsv_to_rgb(hsv_img)
        plt.imshow(rgb_img, extent=[q_vals[0], q_vals[-1], t_vals[0], t_vals[-1]], aspect='auto')
        plt.title("Arg(Θ) Phase Map")

    elif mode == "toroidal":
        # Toroidal unfolding – project Re(τ), Re(φ)
        X = np.cos(t_vals[:, None]) * (1 + np.cos(q_vals[None, :]))
        Y = np.sin(t_vals[:, None]) * (1 + np.cos(q_vals[None, :]))
        data = np.abs(Theta)
        plt.pcolormesh(X, Y, data, shading='auto', cmap='plasma')
        plt.axis('equal')
        plt.title("Toroidal Unfolding of Θ(q, τ)")

    else:
        raise ValueError(f"Unknown projection mode: {mode}")

    plt.xlabel("q (phase)")
    plt.ylabel("t (time)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"✅ Projection saved to {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Complex Theta Transform visualizer")
    parser.add_argument("--projection", type=str, default="amp",
                        choices=["amp", "phase", "toroidal"],
                        help="Projection type (amp, phase, toroidal)")
    parser.add_argument("--qmin", type=float, default=-1.0)
    parser.add_argument("--qmax", type=float, default=1.0)
    parser.add_argument("--tmin", type=float, default=-1.0)
    parser.add_argument("--tmax", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.2, help="Imaginary time component")
    parser.add_argument("--phi_i", type=float, default=0.1, help="Imaginary phase component")
    parser.add_argument("--phi_r", type=float, default=0.0, help="Real phase component")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--outfile", type=str, default="theta_projection.png")
    args = parser.parse_args()

    q_vals = np.linspace(args.qmin, args.qmax, args.res)
    t_vals = np.linspace(args.tmin, args.tmax, args.res)

    print(f"🌀 Generating Θ(q, τ, φ) with ψ={args.psi}, φ_r={args.phi_r}, φ_i={args.phi_i}")
    Theta = theta_field(q_vals, t_vals, psi=args.psi, phi_r=args.phi_r, phi_i=args.phi_i)

    print(f"🎨 Plotting projection mode = {args.projection}")
    plot_projection(Theta, q_vals, t_vals, mode=args.projection, outfile=args.outfile)


if __name__ == "__main__":
    main()

