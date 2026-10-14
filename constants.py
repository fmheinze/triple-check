"""Shared physical constants and unit conversions in SI units."""

G_SI = 6.67430e-11
C_SI = 299792458.0
MSUN_SI = 1.98847e30
DAY_SI = 86400.0
AU_SI = 1.495978707e11

# One solar mass in geometric time units: G M_sun / c^3, in seconds.
MSUN_TIME_SI = G_SI * MSUN_SI / C_SI**3
