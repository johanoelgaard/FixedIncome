import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


# plotting spot, forward rates and zcb prices
# Plot spot and forward rates
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T, f_3M, label='3M forward rates', alpha=0.5, marker='.')
scatter2 = ax1.scatter(T, r, label='Spot rates', alpha=0.5, marker='.')
ax1.set_ylim([0, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p, label='ZCB prices', color='red', alpha=0.5, marker='.')
ax2.set_ylim([0, 1.1])
ax2.set_ylabel('Bond Prices')

ax1.yaxis.set_major_locator(MultipleLocator(0.01))
for tick in ax1.get_yticks():
    ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond Prices')
plt.show()


# plotting spot, forward rates and zcb prices
# Plot spot and forward rates
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T_ghi, f_3M_ghi, label='3M forward rates', alpha=0.5, marker='.')
scatter2 = ax1.scatter(T_ghi, r_ghi, label='Spot rates', alpha=0.5, marker='.')
ax1.set_ylim([0, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T_ghi, p_ghi, label='ZCB prices', color='red', alpha=0.5, marker='.')
ax2.set_ylim([0, 1.1])
ax2.set_ylabel('Bond Prices')

ax1.yaxis.set_major_locator(MultipleLocator(0.01))
for tick in ax1.get_yticks():
    ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond Prices')
plt.show()


# plotting spot, forward rates and zcb prices
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T[1:], R[1:],marker = '.', label='Spot rates', alpha=0.5, color='red')
scatter2 = ax1.scatter(T[1:], f_3M[1:], marker = '.', label='3M forward rates', alpha=0.5, color='blue')
ax1.set_ylim([0, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# add dotted lines across the plot at 0.01, 0.02, 0.03, 0.04 and 0.05
for i in range(1, 6):
    ax1.axhline(y=i*0.01, color='gray', linestyle='--')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p, label='ZCB prices', marker = '.', color='black', alpha=0.5)
ax2.set_ylim([0, 1.1])
ax2.set_ylabel('Bond Prices')

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond Prices')
plt.show()
plt.close


# plot interpolated spot and forward rates
plt.figure(figsize=(10, 6), dpi=300)

plt.scatter(T_inter[1:], R_inter[1:],marker = '.', label='Spot rates', alpha=0.5, color='black',s=10)
plt.scatter(T_inter[1:], f_inter[1:], marker = '.', label='Forward rates', alpha=0.5, color='red',s=10)
plt.ylim([0, 0.0625])
plt.xlabel('Time to Maturity')
plt.ylabel('Rates')

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.01))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

textstr = f'method: {interpolation_options["method"]}\ntransition: {interpolation_options["transition"]}'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes,
         verticalalignment='bottom', bbox=props)

plt.legend(loc = 'lower right')

plt.title('Calibrated zero-coupon spot rates and forward rates')
plt.show()
plt.close()


# plotting confidence intervals for Vasicek model
# plot lb and ub and confidence interval
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(T_ci_plot, ub, label='Upper bound', color='blue', marker='.')
plt.scatter(T_ci_plot, lb, label='Lower bound', color='red', marker='.')
plt.plot(T_ci_plot, ub_sd * np.ones(N_ci_plot), label='Upper bound - sd', color='black')
plt.plot(T_ci_plot, lb_sd * np.ones(N_ci_plot), label='Lower bound - sd', color='black')
plt.xlabel('Time to maturity')
plt.title('Confidence interval for Vasicek model')
plt.legend(loc='lower right')

# Set y-axis to start at 0
plt.ylim(bottom=0, top=0.085)


# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.02))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.show()
plt.close()

# plotting confidence intervals for vasicek model and simulated values
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(T_simul, r_euler, label='Euler', color='red', marker='.', s=1)
plt.scatter(T_simul, r_milstein, label='Milstein', color='green', marker='.', s=1)
plt.scatter(T_simul, r_exact, label='Exact', color='blue', marker='.', s=1)
plt.scatter(T_simul, lb_simul, label='Lower bound', color='black', marker='.', s=1)
plt.scatter(T_simul, ub_simul, label='Upper bound', color='black', marker='.', s=1)

plt.xlabel('Time to maturity')
plt.title('Simulation of short rate in CIR')
plt.ylim(bottom=0, top=0.085)

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.02))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)

plt.legend(loc='lower right')

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.show()
plt.close()


# plotting confidence intervals in the CIR model
# plot lb and ub and confidence interval
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(T_ci_plot, ub, label='Upper bound', color='blue', marker='.')
plt.scatter(T_ci_plot, lb, label='Lower bound', color='red', marker='.')
plt.plot(T_ci_plot, ub_sd * np.ones(N_ci_plot), label='Upper bound - sd', color='black')
plt.plot(T_ci_plot, lb_sd * np.ones(N_ci_plot), label='Lower bound - sd', color='black')
plt.xlabel('Time to maturity')
plt.title('Confidence interval for CIR model')
plt.legend(loc='lower right')

# Set y-axis to start at 0
plt.ylim(bottom=0, top=0.085)


# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.02))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.show()
plt.close()


# plotting confidence intervals and simulated values in the CIR model
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(T_simul, r_euler, label='Euler', color='red', marker='.', s=1)
plt.scatter(T_simul, r_milstein, label='Milstein', color='green', marker='.', s=1)
plt.scatter(T_simul, r_exact, label='Exact', color='blue', marker='.', s=1)
plt.scatter(T_simul, lb_simul, label='Lower bound', color='black', marker='.', s=1)
plt.scatter(T_simul, ub_simul, label='Upper bound', color='black', marker='.', s=1)

plt.xlabel('Time to maturity')
plt.title('Simulation of short rate in CIR')
plt.ylim(bottom=0, top=0.085)

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.02))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)

plt.legend(loc='lower right')

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.show()
plt.close()


# plotting histogram of simulated short rates
bins = 50

mu = np.exp(-a*T_3)*r0 + b/a*(1-np.exp(-a*T_3))
std = np.sqrt(sigma**2/(2*a)*(1-np.exp(-2*a*T_3)))

# plot the histogram of the simulated short rates with 100 equally spaced bins
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(r3n, bins=bins, color='blue', edgecolor='black', label='Simulated', density=True)

# add normal distribution with mean and standard deviation of the short rate to the plot
hist, bin_edges = np.histogram(r3n, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
pdf_values = norm.pdf(bin_centers, mu, std)
plt.scatter(bin_centers, pdf_values, color='red', label="Theorectical", s=2)

plt.xlabel('Short rate')
plt.ylabel('Frequency')
plt.title(f'Histogram of simulated short rates in the Vasicek model with {N} simulations')
textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$\n $T = {T_3}$\n method = {method}'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(-0.065, 0.675, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)
plt.legend(loc='upper right')
plt.show()
plt.close()


# plotting histogram of simulated short rates
bins = 100

# plot the histogram of the simulated short rates with 100 equally spaced bins
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(r3n, bins=bins, color='blue', edgecolor='black', label='Simulated', density=True)

# add normal distribution with mean and standard deviation of the short rate to the plot
hist, bin_edges = np.histogram(r3n, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
pdf_values = norm.pdf(bin_centers, mu, std)
plt.scatter(bin_centers, pdf_values, color='red', label="Theorectical", s=2)

plt.xlabel('Short rate')
plt.ylabel('Frequency')
plt.title(f'Histogram of simulated short rates in the Vasicek model with {N} simulations')
textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$\n $T = {T_3}$\n method = {method}'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(-0.065, 0.675, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)
plt.legend(loc='upper right')
plt.show()
plt.close()


# plotting histogram of simulated short rates
bins = 50

df = 4*a*b/sigma**2
noncentrality = (4*a*np.exp(-a*T_3))/(sigma**2*(1-np.exp(-a*T_3)))*r0
scale = (sigma**2 * (1 - np.exp(-a * T_3))) / (4 * a)

# plot the histogram of the simulated short rates with 100 equally spaced bins
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(r3n, bins=bins, color='blue', edgecolor='black', label='Simulated', density=True)

# add chi2 distribution with mean and standard deviation of the short rate to the plot
hist, bin_edges = np.histogram(r3n, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
pdf_values = ncx2.pdf(bin_centers/scale, df=df, nc = noncentrality)/scale
plt.scatter(bin_centers, pdf_values, color='red', label="Theorectical", s=2)

plt.xlabel('Short rate')
plt.ylabel('Frequency')
plt.title(f'Histogram of simulated short rates in the CIR model with {N} simulations')
textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$\n $T = {T_3}$\n method = {method}'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(-0.065, 0.675, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)
plt.legend(loc='upper right')
plt.show()
plt.close()


# plotting histogram of simulated short rates
bins = 100

df = 4*a*b/sigma**2
noncentrality = (4*a*np.exp(-a*T_3))/(sigma**2*(1-np.exp(-a*T_3)))*r0
scale = (sigma**2 * (1 - np.exp(-a * T_3))) / (4 * a)

# plot the histogram of the simulated short rates with 100 equally spaced bins
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(r3n, bins=bins, color='blue', edgecolor='black', label='Simulated', density=True)

# add chi2 distribution with mean and standard deviation of the short rate to the plot
hist, bin_edges = np.histogram(r3n, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
pdf_values = ncx2.pdf(bin_centers/scale, df=df, nc = noncentrality)/scale
plt.scatter(bin_centers, pdf_values, color='red', label="Theorectical", s=2)

plt.xlabel('Short rate')
plt.ylabel('Frequency')
plt.title(f'Histogram of simulated short rates in the CIR model with {N} simulations')
textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$\n $T = {T_3}$\n method = {method}'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(-0.065, 0.675, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)
plt.legend(loc='upper right')
plt.show()
plt.close()


# plotting spot, forward rates and zcb prices
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T[1:], R_vasicek[1:],marker = '.', label='Spot rates', alpha=0.7, color='red')
scatter2 = ax1.scatter(T[1:], f_vasicek[1:], marker = '.', label='Forward rates', alpha=0.7, color='blue')
ax1.set_ylim([0.02, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# add dotted lines across the plot at 0.01, 0.02, 0.03, 0.04 and 0.05
for i in range(1, 6):
    ax1.axhline(y=i*0.01, color='gray', linestyle='--')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p_vasicek, label='ZCB prices', marker = '.', color='black', alpha=0.7)
ax2.set_ylim([0.4, 1.1])
ax2.set_ylabel('Bond Prices')

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond Prices in the Vasicek model')
plt.show()
plt.close()

# plotting spot, forward rates and zcb prices
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T[1:], R_cir[1:],marker = '.', label='Spot rates', alpha=0.7, color='red')
scatter2 = ax1.scatter(T[1:], f_cir[1:], marker = '.', label='Forward rates', alpha=0.7, color='blue')
ax1.set_ylim([0.02, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# add dotted lines across the plot at 0.01, 0.02, 0.03, 0.04 and 0.05
for i in range(1, 6):
    ax1.axhline(y=i*0.01, color='gray', linestyle='--')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p_cir, label='ZCB prices', marker = '.', color='black', alpha=0.7)
ax2.set_ylim([0.4, 1.1])
ax2.set_ylabel('Bond Prices')

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond Prices in the Vasicek model')
plt.show()
plt.close()


# plotting fitted zcb prices and error
fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_vasicek, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)

ax[0].yaxis.set_major_locator(MultipleLocator(base=0.2))
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)

ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('Vasicek model unconstrained to ZCB prices', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_vasicek, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.01, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'left', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('Vasicek model unconstrained to ZCB prices, $\hat{b}=0.12$', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_vasicek, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('Vasicek model bounded to ZCB prices, true values inside bounds', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_vasicek, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('Vasicek model bounded to ZCB prices, true values outside bounds', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_vasicek, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('Vasicek model bounded with non-linear constraint to ZCB prices', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_cir, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('CIR model unbounded fit to ZCB prices', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_cir, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('CIR model unbounded fit to ZCB prices, $\hat{b}=0.08$', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_cir, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('CIR model bounded fit to ZCB prices, true values inside bounds', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_cir, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('CIR model bounded fit to ZCB prices, true values outside bounds', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

# First subplot: ZCB Prices
ax[0].scatter(T, p_star_cir, label='$p^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, p_fit, label='$\hat{p}$', marker='.', color='red', alpha=0.7)
for i in range(0, 3):
    ax[0].axhline(y=i*0.5, color='gray', linestyle='--')
ax[0].set_ylabel('ZCB Prices', fontsize=12)
ax[0].legend(fontsize=12)
textstr = f'$r_0={r0}$\n$a^*={a}$\n$b^*={b}$\n$\\sigma^*={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
ax[0].text(0.01, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)
# Calculate dynamic limits
y_min = min(res)
y_max = max(res)
y_range = y_max - y_min

# Set the y-axis limits with some padding
padding = 0.05 * y_range  # Adjust padding as needed
ax[1].set_ylim([y_min - padding, y_max + padding])

# Create horizontal lines dynamically based on the range
num_lines = 3  # Number of horizontal lines
step = y_range / (num_lines - 1)  # Spacing between lines
for i in range(num_lines):
    ax[1].axhline(y=y_min + i * step, color='gray', linestyle='--')
ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12)

# calculate the mse of the residuals
mse = np.mean(res**2)
txt_mse = f'MSE: {mse:.5e}'
ax[1].text(0.99, 0.075, txt_mse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title('CIR model bounded with non-linear constraint fit to ZCB prices', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T[1:], R_plot[1:],marker = '.', label='Spot rates', alpha=0.7, color='red')
scatter2 = ax1.scatter(T[1:], f_plot[1:], marker = '.', label='Forward rates', alpha=0.7, color='blue')
ax1.set_ylim([0.02, 0.075])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# add dotted lines across the plot at 0.01, 0.02, 0.03, 0.04 and 0.05
ax1.yaxis.set_major_locator(MultipleLocator(base=0.01))
for tick in ax1.get_yticks():
    ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p_plot, label='ZCB prices', marker = '.', color='black', alpha=0.7)
ax2.set_ylim([0.5, 1.05])
ax2.yaxis.set_major_locator(MultipleLocator(base=0.1))
ax2.set_ylabel('Bond Prices')

textstr = f'$f_\infty={f_inf}$\n$a={a_star}$\n$b={b_star}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Nelson-Siegel Spot, Forward Rates and Bond Prices')
plt.show()
plt.close()


fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

scatter1 = ax1.scatter(T[1:], f_plot[1:],marker = '.', label='NS Forward Rate', alpha=0.7, color='red')
scatter2 = ax1.scatter(T[1:], f_star[1:], marker = '.', label='$F^*$', alpha=0.7, color='blue')
ax1.set_ylim([0.02, 0.075])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# add dotted lines across the plot at 0.01, 0.02, 0.03, 0.04 and 0.05
ax1.yaxis.set_major_locator(MultipleLocator(base=0.01))
for tick in ax1.get_yticks():
    ax1.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

textstr = f'$f_\infty={f_inf}$\n$a={a_star}$\n$b={b_star}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

# Combine legends
scatters = [scatter1, scatter2]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('$F^*$ and Nelson-Siegel Forward')
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)

# First subplot: forward rates
ax[0].scatter(T, f_star, label='$f^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, f_fit1, label='$\hat{f}$', marker='.', color='red', alpha=0.7)

ax[0].yaxis.set_major_locator(MultipleLocator(base=0.02))
for tick in ax[0].get_yticks():
    ax[0].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax[0].set_ylabel('Rates', fontsize=12)

ax[0].legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${f_inf_hat1:.8f}\n$\hat a$ ={a_hat1}\n$\hat b $={b_hat1}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
ax[0].text(0.98, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res1, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)

# Create horizontal lines dynamically based on the range
max_residual = np.max(np.abs(res1))
base1 = 2*10 ** (int(np.log10(max_residual))-1)
ax[1].yaxis.set_major_locator(MultipleLocator(base=base1))
for tick in ax[1].get_yticks():
    ax[1].axhline(y=tick, color='gray', linestyle='--', linewidth=1)

ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12, loc='upper right')

# calculate the mse of the residuals
txt_sse = f'SSE: {sse1:.5e}'
ax[1].text(0.98, 0.075, txt_sse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title(f'Empirical and fitted forward rates with the Nelson-Siegel function for $K = {K+1}$', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)

# First subplot: forward rates
ax[0].scatter(T, f_star, label='$f^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, f_fit2, label='$\hat{f}$', marker='.', color='red', alpha=0.7)

ax[0].yaxis.set_major_locator(MultipleLocator(base=0.02))
for tick in ax[0].get_yticks():
    ax[0].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax[0].set_ylabel('Rates', fontsize=12)

ax[0].legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${f_inf_hat2:.8f}\n$\hat a$ ={a_hat2}\n$\hat b $={b_hat2}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
ax[0].text(0.98, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res2, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)

# Create horizontal lines dynamically based on the range
max_residual = np.max(np.abs(res2))
base1 = 2*10 ** (int(np.log10(max_residual))-1)
ax[1].yaxis.set_major_locator(MultipleLocator(base=base1))
for tick in ax[1].get_yticks():
    ax[1].axhline(y=tick, color='gray', linestyle='--', linewidth=1)

ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12, loc='upper right')

# calculate the mse of the residuals
txt_sse = f'SSE: {sse2:.5e}'
ax[1].text(0.98, 0.075, txt_sse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title(f'Empirical and fitted forward rates with the Nelson-Siegel function for $K = {K+1}$', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()


fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)

# First subplot: forward rates
ax[0].scatter(T, f_star, label='$f^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, f_fit3, label='$\hat{f}$', marker='.', color='red', alpha=0.7)

ax[0].yaxis.set_major_locator(MultipleLocator(base=0.02))
for tick in ax[0].get_yticks():
    ax[0].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax[0].set_ylabel('Rates', fontsize=12)

ax[0].legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${f_inf_hat3:.8f}\n$\hat a$ ={a_hat3}\n$\hat b $={b_hat3}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
ax[0].text(0.98, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res3, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)

# Create horizontal lines dynamically based on the range
max_residual = np.max(np.abs(res3))
base1 = 2*10 ** (int(np.log10(max_residual))-1)
ax[1].yaxis.set_major_locator(MultipleLocator(base=base1))
for tick in ax[1].get_yticks():
    ax[1].axhline(y=tick, color='gray', linestyle='--', linewidth=1)

ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12, loc='upper right')

# calculate the mse of the residuals
txt_sse = f'SSE: {sse3:.5e}'
ax[1].text(0.98, 0.075, txt_sse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title(f'Empirical and fitted forward rates with the Nelson-Siegel function for $K = {K+1}$', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

# fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)
plt.figure(figsize=(10, 6), dpi=300)

# First subplot: forward rates
plt.scatter(T, theta_2d, label='$\Theta$', marker='.', color='blue', alpha=0.7)

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.005))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)
plt.ylabel('$\Theta(t)$', fontsize=12)

plt.legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${param_2d[0]:.8f}\n$\hat a$ ={param_2d[1]}\n$\hat b $={param_2d[2]}\n $SSE$ = {sse2:.5e}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
plt.text(0.98, 0.075, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)


plt.title(f'Fitted valued for $\Theta(t)$, $K={K+1}$', fontsize=18)
# plt.tight_layout()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)

# First subplot: forward rates
ax[0].scatter(T, f_star, label='$f^*$', marker='.', color='blue', alpha=0.7)
ax[0].scatter(T, f_fit_NCG, label='$\hat{f}$', marker='.', color='red', alpha=0.7)

ax[0].yaxis.set_major_locator(MultipleLocator(base=0.02))
for tick in ax[0].get_yticks():
    ax[0].axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax[0].set_ylabel('Rates', fontsize=12)

ax[0].legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${f_inf_hat_NCG:.8f}\n$\hat a$ ={a_hat_NCG}\n$\hat b $={b_hat_NCG}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
ax[0].text(0.98, 0.075, textstr, transform=ax[0].transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)

# Second subplot: Residuals
ax[1].scatter(T, res_NCG, label='Residuals', marker='.', color='black', alpha=0.7)
ax[1].set_ylabel('Residuals', fontsize=12)

# Create horizontal lines dynamically based on the range
max_residual = np.max(np.abs(res_NCG))
base1 = 10 ** (int(np.log10(max_residual))-1)
ax[1].yaxis.set_major_locator(MultipleLocator(base=base1))
for tick in ax[1].get_yticks():
    ax[1].axhline(y=tick, color='gray', linestyle='--', linewidth=1)

ax[1].set_xlabel('Time to Maturity', fontsize=12)
ax[1].legend(fontsize=12, loc='upper right')

# calculate the mse of the residuals
txt_sse = f'SSE: {sse_NCG:.5e}'
ax[1].text(0.98, 0.075, txt_sse, transform=ax[1].transAxes, fontsize=12,
         verticalalignment='bottom', horizontalalignment = 'right', bbox=props)

for ax_i in ax:
    ax_i.tick_params(axis='both', which='major', labelsize=12)
    ax_i.yaxis.get_offset_text().set_fontsize(12)  # Adjust the size as needed


ax[0].set_title(f'Empirical and fitted forward rates with the Nelson-Siegel function for $K = {K+1}$ fitted w. Newton-CG', fontsize=18)
plt.tight_layout()
plt.show()
plt.close()

# fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=300, sharex=True)
plt.figure(figsize=(10, 6), dpi=300)

# First subplot: forward rates
plt.scatter(T, theta_2e, label='$\Theta$', marker='.', color='blue', alpha=0.7)

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.005))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)
plt.ylabel('$\Theta(t)$', fontsize=12)

plt.legend(fontsize=12, loc = 'upper right')
textstr = f'$\hat f_\infty =${param_2e[0]:.8f}\n$\hat a$ ={param_2e[1]}\n$\hat b $={param_2e[2]}\n $SSE$ = {sse_NCG:.5e}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
plt.text(0.98, 0.075, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom',horizontalalignment = 'right', bbox=props)


plt.title(f'Fitted valued for $\Theta(t)$, $K={K+1}$', fontsize=18)
# plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(10, 6), dpi=300)
plt.title('Price of the price of the Asian style derivative')
plt.scatter([n for n in range(N3c)], price3c, color='blue', alpha=0.7, s=3)
plt.plot([n for n in range(N3c)], pi_3c*np.ones([N3c]), color='red', alpha=0.7)

ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.005))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

textstr = f'$f_\infty =${f_inf3:.8f}\n$a$ ={a3}\n$b $={b3}\n $\pi$ = {pi_3c:.8f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
plt.text(0.987, 0.975, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='top',horizontalalignment = 'right', bbox=props)

# plt.tight_layout()
plt.show()
plt.close()


plt.figure(figsize=(10, 6), dpi=300)
plt.title('Price of the price of the 1Y4Y payer swaption')
plt.scatter([n for n in range(N3e)], price3e, color='blue', alpha=0.7, s=3)
plt.plot([n for n in range(N3e)], pi_3e*np.ones([N3e]), color='red', alpha=0.7)

ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.01))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)

textstr = f'$f_\infty =${f_inf3:.8f}\n$a$ ={a3}\n$b $={b3}\n $\pi$ = {pi_3e:.8f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey')
plt.text(0.987, 0.975, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='top',horizontalalignment = 'right', bbox=props)

# plt.tight_layout()
plt.show()
plt.close()

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (10,6))   #
fig.suptitle(f"Spot rates and 3M forward rates", fontsize = 20)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,0.5,1,1.5,2]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 12)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Maturity",fontsize = 12)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 12)
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax.set_ylim([0,0.0525])
ax.set_ylabel(f"Rate",fontsize = 12)
p1 = ax.scatter(T[1:], r[1:], s = 10, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T[1:], f_3m[1:], s = 10, color = 'red', marker = ".",label="3M forward rates")
p3 = ax.scatter(T[1:], L[1:], s = 10, color = 'blue', marker = ".",label="3M forward LIBOR rates")

plots = [p1,p2,p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 12)

plt.show()
plt.close()

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))   #
fig.suptitle(f"Spot rates, 3M forward rates and LIBOR rates", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])

xticks = [0,1,2,3,4]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Maturity",fontsize = 6)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 6)
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=1)
ax.set_ylim([0,0.0525])
# ax.set_ylabel(f"",fontsize = 6)
p1 = ax.scatter(T[1:], R_spot[1:], s = 1, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T[1:], f[1:], s = 1, color = 'red', marker = ".",label="3M forward rates")
p3 = ax.scatter(T[1:], L[1:], s = 1, color = 'green', marker = ".",label="3M forward LIBOR rates")

plots = [p1,p2,p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)

plt.show()
plt.close()

# Problem 3 - Simulation of the Libor market model
np.random.seed(2024) # seed not used by JLS but used by me for reproducibility
M_simul, T_simul = 400, 4
sigma_lmm = np.zeros([M-2])
for i in range(0,M-2):
    sigma_lmm[i] = sigma_market_1b[i+2]*np.sqrt(T[i+2]/T[i+1])
rho = np.array([[1,0.5,0.9,0.85],[0.95,1,0.95,0.9],[0.9,0.95,1,0.95],[0.85,0.9,0.95,1]])
rho_sqrt = sqrtm(rho)
L_simul = fid.simul_lmm(L[2:M],T[1:M],sigma_lmm,rho,M_simul)
t_simul = np.array([i*T_simul/M_simul for i in range(0,M_simul+1)])

strike_lmm = 0.0475
N_simul = 1000
caplet_price_3, L_exercise = np.zeros(M-2), np.zeros(M-2)
chi_disc = np.zeros([M-2,N_simul])
for n in range(0,N_simul):
    L_simul = fid.simul_lmm(L[2:M],T[1:M],sigma_lmm,rho,M_simul)
    for j in range(0,M-2):
        L_exercise[j] = L_simul[j,int(j*M_simul/4)+1]
        chi_disc[j,n] = p[-1]*alpha[j+2]*max(L_exercise[j] - strike_lmm,0)
        for k in range(j,M-2):
            chi_disc[j,n] *= (1+alpha[k+2]*L_exercise[k])
for i in range(0,M-2):
    caplet_price_3[i] = sum(chi_disc[i,:])/N_simul*10000
print(f"caplet_price for a strike of {strike_lmm} is {caplet_price_3}, cap price: {sum(caplet_price_3)}")


fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Simulated Libor rates in the LIbor Market Model", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = np.array([0,1,2,3,4])
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]+-0.2,xticks[-1]+0.2])
# ax.set_xlim([xticks[0]+-0.2,2+0.2])
plt.xlabel(f"Time",fontsize = 6)
ax.set_yticks([0,0.02,0.04,0.06,0.08])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08],fontsize = 6)
ax.set_ylim([0,0.0825])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(t_simul, L_simul[0,:], s = 1, color = 'black', marker = ".",label="L_2(t)")
p2 = ax.scatter(t_simul, L_simul[1,:], s = 1, color = 'red', marker = ".",label="L_3(t)")
p3 = ax.scatter(t_simul, L_simul[2,:], s = 1, color = 'blue', marker = ".",label="L_4(t)")
p4 = ax.scatter(t_simul, L_simul[3,:], s = 1, color = 'green', marker = ".",label="L_5(t)")
plots = [p1,p2,p3,p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)
plt.show()
plt.close()

# Plot spot and forward rates
fig, ax1 = plt.subplots()

scatter1 = ax1.scatter(T, f, label='Forward rates', alpha=0.5, marker='.', color='red')
scatter2 = ax1.scatter(T, R, label='Spot rates', alpha=0.5, marker='.', color='blue')
ax1.set_ylim([0, 0.055])
ax1.set_xlabel('Time to Maturity')
ax1.set_ylabel('Rates')

# Add bond prices to the plot on the right axis
ax2 = ax1.twinx()
scatter3 = ax2.scatter(T, p, label='ZCB prices', alpha=0.5, marker='.', color='black')
ax2.set_ylim([0, 1.1])
ax2.set_ylabel('Bond Prices')

# Combine legends
scatters = [scatter1, scatter2, scatter3]
labels = [scatter.get_label() for scatter in scatters]
ax1.legend(scatters, labels, loc='lower right')

plt.title('Spot, Forward Rates and Bond')
plt.show()
plt.close()

# plot the simulation
plt.figure(figsize=(10, 6))
plt.scatter(t_simul, r_simul, label='Euler', color='black', marker='.', s=2)
plt.scatter(t_simul, ub, label='Upper bound', color='red', marker='.', s=2)
plt.scatter(t_simul, lb, label='Lower bound', color='blue', marker='.', s=2)
plt.plot(t_simul, ub_sd * np.ones(t_simul.shape[0]), label='Upper bound - sd', color='black')
plt.plot(t_simul, lb_sd * np.ones(t_simul.shape[0]), label='Lower bound - sd', color='black')

plt.xlabel('Time to maturity')
plt.title('Simulation of short rate in CIR')
plt.ylim(bottom=0, top=0.085)

# Add horizontal lines at each tick mark
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.02))
for tick in ax.get_yticks():
    ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.5)

plt.legend(loc='lower right')

textstr = f'$r_0={r0}$\n$a={a}$\n$b={b}$\n$\\sigma={sigma}$'
props = dict(boxstyle='round', facecolor='white', alpha=0.2)
plt.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
         verticalalignment='bottom', bbox=props)

plt.show()
plt.close()

# PLot of spot and forward rates
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Spot- and forward rates", fontsize = 10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.2,xticks[-1]+0.2])
plt.xlabel(f"Maturity",fontsize = 7)
ax.set_yticks([0,0.02,0.04,0.06,0.08])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08],fontsize = 6)
ax.set_ylim([0,0.081])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(T[1:], R[1:], s = 3, color = 'black', marker = ".",label="Spot rates")
p2 = ax.scatter(T[1:], L[1:], s = 3, color = 'red', marker = ".",label="forward rates")
plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
plt.close()

# Plot of market implied volatilities
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Market implied volatilities", fontsize = 10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = K_swaption_offset
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Strike offset",fontsize = 7)
ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize = 6)
ax.set_ylim([0,0.408])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(K_swaption_offset, iv, s = 3, color = 'black', marker = ".",label="IV market")
plots = [p1]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
plt.close()

# Market and fitted implied volatilities
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Market and fitted implied volatilities", fontsize = 10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = K_swaption_offset
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.02,xticks[-1]+0.02])
plt.xlabel(f"Strike offset",fontsize = 7)
ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize = 6)
ax.set_ylim([0,0.408])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(K_swaption_offset, iv, s = 6, color = 'black', marker = ".",label="IV market")
p2 = ax.scatter(K_swaption_offset, iv_fit, s = 1, color = 'red', marker = ".",label="IV fit")
plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
plt.close()

# PLot of simulated values of the SABR model
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Spot- and forward rates in the SABR model", fontsize = 10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,1,2,3,4,5]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.1,xticks[-1]+0.1])
plt.xlabel(f"Years",fontsize = 7)
ax.set_yticks([0,0.02,0.04,0.06,0.08,0.1])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08,0.1],fontsize = 6)
ax.set_ylim([0,0.101])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Forward swap rate",fontsize = 7)
p1 = ax.scatter(t_simul, F_simul, s = 1, color = 'black', marker = ".",label="Forward swap rate")
ax2 = ax.twinx()
ax2.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax2.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 6)
ax2.set_ylim([0,0.0505])
ax2.set_ylabel(f"Volatility",fontsize = 7)
p2 = ax2.scatter(t_simul, sigma_simul, s = 1, color = 'red', marker = ".",label="Volatility")
plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
plt.close()

# Plot of simulated and theoretical density of the forward par swap rate
fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"PMF of the forwad par swap rate (Maturity = {T_simul_dens})", fontsize = 10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0.02,0.03,0.04,0.05,0.06,0.07,0.08]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.001,xticks[-1]+0.001])
ax.set_xlabel(f"Forward par swap rate",fontsize = 7)
ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
ax.set_yticklabels([0,0.01,0.02,0.03,0.04,0.05],fontsize = 6)
ax.set_ylim([0,0.0503])
ax.set_ylabel(f"PMF",fontsize = 7)
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(bins, black_pmf, s = 1, color = 'black', marker = ".",label="Theoretical PMF")
p2 = ax.scatter(bins, simul_pmf, s = 1, color = 'red', marker = ".",label="Simulated PMF")
plots = [p1,p2]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
plt.show()
plt.close()

fig = plt.figure(constrained_layout=False, dpi = 300, figsize = (5,3))
fig.suptitle(f"Simulated par swap rates in the Swap Market Model", fontsize = 9)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = np.array([0,1,2,3,4,5])
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]+-0.2,xticks[-1]+0.2])
plt.xlabel(f"Time",fontsize = 6)
ax.set_yticks([0,0.02,0.04,0.06,0.08])
ax.set_yticklabels([0,0.02,0.04,0.06,0.08],fontsize = 6)
ax.set_ylim([0,0.0825])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
p1 = ax.scatter(t_simul, R_simul[0,:], s = 1, color = 'black', marker = ".",label="R_0^5(t)")
p2 = ax.scatter(t_simul, R_simul[1,:], s = 1, color = 'red', marker = ".",label="R_1^5(t)")
p3 = ax.scatter(t_simul, R_simul[2,:], s = 1, color = 'orange', marker = ".",label="R_2^5(t)")
p4 = ax.scatter(t_simul, R_simul[3,:], s = 1, color = 'blue', marker = ".",label="R_3^5(t)")
p5 = ax.scatter(t_simul, R_simul[4,:], s = 1, color = 'green', marker = ".",label="R_4^5(t)")
plots = [p1,p2,p3,p4,p5]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 6)
plt.show()
plt.close()

# PLot of zcb prices, spot rates and instantaneous forward rates in the Vasicek model
fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Spot- and forward rates in the Vasicek model",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.01,xticks[-1]+0.01])
plt.xlabel(f"Maturity",fontsize = 7)
yticks1 = [0,0.2,0.4,0.6,0.8,1]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Price",fontsize = 7)
p1 = ax.scatter(T, p, s = 2, color = 'black', marker = ".",label="ZCB prices")
ax2 = ax.twinx()
yticks2 = [0,0.01,0.02,0.03,0.04,0.05]
ax2.set_yticks(yticks2)
ax2.set_yticklabels(yticks2,fontsize = 6)
ax2.set_ylim([yticks2[0],yticks2[-1]+(yticks2[-1]-yticks2[0])*0.02])
ax2.set_ylabel(f"Spot and forward rates",fontsize = 7)
p2 = ax2.scatter(T, R, s = 2, color = 'red', marker = ".",label="Spot rate")
p3 = ax2.scatter(T, f, s = 2, color = 'blue', marker = ".",label="Instantaneous forward rate")
p4 = ax2.scatter(T, R_swap_plot, s = 2, color = 'green', marker = ".",label="Par swap rate")
p5 = ax2.scatter(T, f_6m, s = 2, color = 'orange', marker = ".",label="6M forward rate")
plots = [p1,p2,p3,p4,p5]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)
# fig.savefig("C:/")
plt.show()
plt.close()


# PLot of simulated short rates in the Vasicek model
fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Simulated spot rates in the Vasicek model",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,2,4,6,8,10]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.01,xticks[-1]+0.01])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0,0.02,0.04,0.06,0.08,0.1]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Short rate",fontsize = 7)
p1 = ax.scatter(t_simul, r_simul, s = 1, color = 'black', marker = ".",label="Simulated short rate")
p2 = ax.scatter(t_simul, lb, s = 1, color = 'red', marker = ".",label=f"Lower {size_ci} CB")
p3 = ax.scatter(t_simul, ub, s = 1, color = 'red', marker = ".",label=f"Upper {size_ci} CB")
p4 = ax.scatter(t_simul, lb_sd*np.ones([M_simul+1]), s = 1, color = 'blue', marker = ".",label=f"Lower {size_ci} CB Stat. dist.")
p5 = ax.scatter(t_simul, ub_sd*np.ones([M_simul+1]), s = 1, color = 'blue', marker = ".",label=f"Upper {size_ci} CB Stat. dist.")
plots = [p1,p2,p3,p4,p5]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
# fig.savefig("C:/")
plt.show()
plt.close()

# Plot of option prices as a function of N_simul
fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Swaption price as a function of number of simulations",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*N_simul),int((2/5)*N_simul),int((3/5)*N_simul),int((4/5)*N_simul),int(N_simul)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-2,xticks[-1]+2])
plt.xlabel(f"Number of simulations",fontsize = 7)
yticks1 = [0,0.01,0.02,0.03,0.04,0.05]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Swaption price",fontsize = 7)
p1 = ax.scatter([i for i in range(1,N_simul+1)], price_swaption_plot, s = 1, color = 'black', marker = ".",label="Swaption price")
plots = [p1]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper right",fontsize = 5)
# fig.savefig("C:/")
plt.show()
plt.close()