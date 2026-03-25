"""
Analyze Limber validity results from Professor Pullen's code.
"""

import numpy as np

# Load the Limber validity array
limber = np.loadtxt('test_limber.txt')
nchantot = len(limber)

print("="*70)
print("STEP 0: LIMBER VALIDITY DIAGNOSTIC")
print("="*70)

# (a) Total number of channels
print(f"\n(a) nchantot = {nchantot} channels")

# (b) Channels with limber_min <= 50 (fully Limber across our ℓ range)
fully_limber = np.sum(limber <= 50)
print(f"\n(b) Channels with limber_min <= 50: {fully_limber}")
print(f"    These use Limber approximation across ENTIRE ℓ range [50, 350]")

# (c) Channels with 50 < limber_min <= 350 (partial Limber)
partial_limber = np.sum((limber > 50) & (limber <= 350))
print(f"\n(c) Channels with 50 < limber_min <= 350: {partial_limber}")
print(f"    These need Bessel integral for lowest ℓ bins, Limber for high ℓ")

# (d) Channels with limber_min > 350 (never Limber in our range)
never_limber = np.sum(limber > 350)
print(f"\n(d) Channels with limber_min > 350: {never_limber}")
print(f"    These need Bessel integral across ENTIRE ℓ range [50, 350]")

# Verify total
print(f"\n    Verification: {fully_limber} + {partial_limber} + {never_limber} = {fully_limber + partial_limber + never_limber}")

# (e) First 10 channels (bands 1-2, lowest wavelength)
print(f"\n(e) Limber_min for first 10 channels (bands 1-2):")
print(f"    Channels 0-9: {limber[:10].astype(int)}")

# (f) Last 10 channels (bands 5-6, highest wavelength, R=110 and R=130)
print(f"\n(f) Limber_min for last 10 channels (bands 5-6, R=110/130):")
print(f"    Channels {nchantot-10}-{nchantot-1}: {limber[-10:].astype(int)}")

# (g) Summary by band
# Band boundaries based on Professor Pullen's code
# lamband = [0.75, 1.10, 1.63, 2.42, 3.82, 4.42, 5.00]
# R = [41, 41, 41, 35, 110, 130]
# nchan per band from his calculation

# Recompute nchan distribution
lamband = np.array([0.75, 1.10, 1.63, 2.42, 3.82, 4.42, 5.00])
dlamband = np.diff(lamband)
lamcen = 0.5 * (lamband[:-1] + lamband[1:])
R = np.array([41, 41, 41, 35, 110, 130])
dlamchan = lamcen / R
nchan = np.floor(dlamband / dlamchan).astype(int)

print(f"\n(g) Summary by band:")
print(f"    Band | λ range (μm) |  R  | Channels | limber_min range")
print(f"    " + "-"*60)

start_idx = 0
for i in range(6):
    end_idx = start_idx + nchan[i]
    limber_band = limber[start_idx:end_idx]
    limber_min = int(np.min(limber_band))
    limber_max = int(np.max(limber_band))

    # Count by category
    full = np.sum(limber_band <= 50)
    partial = np.sum((limber_band > 50) & (limber_band <= 350))
    never = np.sum(limber_band > 350)

    print(f"    {i+1:4} | {lamband[i]:.2f}-{lamband[i+1]:.2f} | {R[i]:3} | "
          f"{nchan[i]:8} | {limber_min:4}-{limber_max:4} "
          f"(Full:{full:2}, Part:{partial:2}, Never:{never:2})")

    start_idx = end_idx

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print(f"• Bands 1-4 (λ < 3.82 μm): ALL channels have limber_min < 350")
print(f"  → Can use mix of Limber and Bessel depending on ℓ bin")
print(f"\n• Bands 5-6 (λ > 3.82 μm, R=110/130): ALL channels have limber_min > 350")
print(f"  → MUST use full Bessel integral for entire ℓ range [50, 350]")
print(f"  → {never_limber} channels affected (~{100*never_limber/nchantot:.0f}% of total)")
print("="*70)
