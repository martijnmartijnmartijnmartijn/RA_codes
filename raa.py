from sys import float_repr_style
import matplotlib.pyplot as plt
import numpy as np
from filecache import filecache
from math import sqrt
from random import Random
from matplotlib.colors import LogNorm
from sympy import Matrix
from sympy.core.logic import fuzzy_or
import LinearBlockCode
from re import findall

get_A = lambda n : np.array([[1 if j <= i else 0 for j in range(n)] for i in range(n)])
get_A_inv = lambda n : np.array([[1 if j == i or j == i + 1 else 0 for j in range(n)] for i in range(n)]).T
to_bitstring = lambda s, n : np.array([int(val) for val in np.binary_repr(s, width=n)])


def get_wt_of_code(G):
    """
    Given an n x k generator matrix G for a linear code, computes its minimum
    distance, and the weight distribution of all its codewords.

    Input:
    - G : np.array with shape (n, k) representing the codes' generator matrix.

    Output:
    - np.array of length n where entry i denotes the # codewords with wt i.
    - integer denoting the minimum distance of the code.
    """
    # Create data structure to save results to.
    n, k = G.shape
    wt_dist = np.zeros(n)
    min_wt = n

    # Generate the codebook.
    for s in range(2**k):
        c = G.dot(to_bitstring(s, k)) % 2
        w = c.sum()
        wt_dist[int(w)-1] += 1

        # Save the minimum weight of this code.
        if 0 < w and w < min_wt:
            min_wt = w
    return wt_dist, min_wt

@filecache(60 * 60 * 24 * 365)
def get_wt_of_codes(mode, k, r, n_samples, seed):
    """
    Helper function of plot_weights: actually samples the codes in the given
    mode of the given block length. Returns the mean weight distribution and
    minimum weights.

    Input:
    - mode : a string of A's and D's. The codes that are sampled have the form
                ... Pi_3 [A or D] Pi_2 [A or D] Pi_1 F_r
             and this string fills in whether A or D is used in any round (and
             implicitely, the number of A/D's determines the number of rounds).
             For example, the string "A" would yield an RA code, the string "AA"
             would yield an RAA code, the string "D" would yield the dual of an
             RA code, the string "DD" would give the dual of an RAA code, etc.
    - k : message length.
    - r : repetion factor. i.e. block length is n = k * r.
            r*k_max are considered.
    - n_samples : number of codes to be sampled.
    - seed : seed used by prng to sample permutations Pi_1, Pi_2, ...

    Output:
    - np.array of length n=k*r s.t. entry i is the mean # codewords with wt i.
    - np.array of length n_samples denoting the minimum distance of each code.
    """
    n = k * r
    min_weights = np.zeros(n_samples)
    mean_wt_dist = np.zeros(n)
    prng = Random(seed)

    # Create fixed components of generator matrices.
    F_r = np.repeat(np.eye(k), r, axis=0)
    A = get_A(n)
    D = get_A_inv(n)

    # Parse mode and sample a code of the indicated form.
    for i in range(n_samples):
        G = F_r
        for M in findall(r'(A\^T|D\^T|A|D|)', mode):
            temp = A if 'A' in M else D
            M = temp.T if "^T" in M else temp
            G = M @ np.eye(n)[prng.sample(range(n),n)] @ G % 2
        # G_dual = get_dual(G) # Alternative way to compute dual...
        # print("G_2^T G == [0]?\n{}".format(G_dual.T @ G % 2))
        # Compute min wt and wt distribution of sampled code.
        wt_dist, min_wt = get_wt_of_code(G)
        mean_wt_dist += wt_dist
        min_weights[i] = min_wt
    return mean_wt_dist, min_weights

def plot_weights(mode, k_min, k_max, r=2, n_samples=100, bins=10, seed=Random.randint(Random(), 0, 2**15)):
    """
    Samples RA-like codes and plots their relative weight distribution over the
    block length, and the minimum distance over the block length.

    Input:
    - mode : a string of A's and D's. The codes that are sampled have the form
             ... Pi_3 [A or D] Pi_2 [A or D] Pi_1 F_r
             and this string fills in whether A or D is used in any round (and
             implicitely, the number of A/D's determines the number of rounds).
             For example, the string "A" would yield an RA code, the string "AA"
             would yield an RAA code, the string "D" would yield the dual of an
             RA code, the string "DD" would give the dual of an RAA code, etc.
    - k_min : smallest message length considered.
    - k_max : largest message length considered. that is, message lengths
              k_min, k_min + 1, ..., k_max are considered.
    - r : repetion factor. i.e. block lengths r*k_min, r*(k_min + 1), ...,
          r*k_max are considered.
    - n_samples : number of codes to be sampled.
    - bins : number of bins in weight distribution histograms.
    - seed : seed used by prng to sample permutations Pi_1, Pi_2, ...
    """
    # Set up data structures to save wt dist and min wts.
    k_s = np.arange(k_min, k_max + 1)
    n_s = k_s * r
    mean_wt_dists = np.zeros((len(k_s), bins))
    mean_min_wt = np.zeros(len(k_s))
    std_min_wt = np.zeros(len(k_s))

    # Calculate wt increases and min wt for randomly sampled bitstrings.
    for k in k_s:
        mean_wt_dist, min_weights = get_wt_of_codes(mode, k, r, n_samples, seed)

        # Normalise mean wt dist, scale to # of bins, compute mean/std of min wt
        mean_wt_dists[k - k_min] = np.array([part.sum() for part in np.array_split(mean_wt_dist / n_samples / 2**k, bins)])
        mean_min_wt[k - k_min] = min_weights.mean() / (k * r)
        std_min_wt[k - k_min] = min_weights.std() / (k * r)

        # Debug prints.
        print(min_weights)
        print("n={} histogram: {}".format(k * r, mean_wt_dists[k - k_min]))

    # Plot squence of histograms showing mean relative wt distibution over n.
    plt.figure(figsize=[12, 6])
    plt.ylim(0-0.5, bins-0.5)
    plt.imshow(mean_wt_dists.T, aspect='auto', origin="lower", cmap='Oranges', interpolation='nearest', norm=LogNorm())
    plt.colorbar()

    # Scale to x-axis=[1,2,...,k_max-k_min] and y-axis=[0,1,...bins]-0.5
    mean = mean_min_wt * bins - 0.5
    std = std_min_wt * bins - 0.5

    # Plot (linear least squares fit of) mean, std dev of min wt.
    plt.scatter(k_s - k_min, mean, label="Mean min. distance", color="grey")
    plt.fill_between(k_s - k_min, mean+std, mean-std, alpha=0.5, color="grey")
    # a, b = np.polyfit(n_s, mean, 1)
    # label = create_quadratic_label(0, a, b)
    # plt.plot(k_s - k_min, a * n_s + b, label=label, c="grey", linestyle="--")

    # Set-up axis labels, axis ticks, title and legends.
    plt.subplots_adjust(right=0.8, top=0.83)
    plt.xlabel('Block length $n = {}k$'.format(r))
    plt.ylabel('Relative weight')
    plt.xticks(np.arange(len(k_s)), n_s)
    plt.yticks(np.arange(bins) + 0.5, [f'{1/bins * (i+1):.2f}' for i in range(bins)])
    plt.title("Mean min. distance and mean relative weight distribution of codewords\nover block length for {} uniformly random $R{}$ codes with repetition factor {}.".format(n_samples, mode, r), pad=20)
    plt.legend(bbox_to_anchor=(1.175,1), loc="upper left")
    plt.savefig("plots/R{}_r={}_n={}_samples={}.pdf".format(mode, r, k_max * r, n_samples))
    # plt.show()


# Global params.
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 15})
r = 2
bins = 20
seed = 12345

# Cached.
n_samples = 100
k_min = 2
k_max = 13

# 12 minutes to go from 18 to 20.
# i.e. 4 minutes for n=38 and 8 minutes for n=40.
#
plot_weights("A", k_min, k_max, r, n_samples, bins, seed)
plot_weights("D^T", k_min, k_max, r, n_samples, bins, seed)

plot_weights("A^T", k_min, k_max, r, n_samples, bins, seed)
plot_weights("D", k_min, k_max, r, n_samples, bins, seed)

# 2 rounds.
plot_weights("AA", k_min, k_max, r, n_samples, bins, seed)
plot_weights("D^TD^T", k_min, k_max, r, n_samples, bins, seed)

plot_weights("DA", k_min, k_max, r, n_samples, bins, seed)
plot_weights("A^TD^T", k_min, k_max, r, n_samples, bins, seed)

plot_weights("AD", k_min, k_max, r, n_samples, bins, seed)
plot_weights("D^TA^T", k_min, k_max, r, n_samples, bins, seed)

plot_weights("DD", k_min, k_max, r, n_samples, bins, seed)
plot_weights("A^TA^T", k_min, k_max, r, n_samples, bins, seed)

# 3 rounds.
plot_weights("ADA", k_min, k_max, r, n_samples, bins, seed)
plot_weights("DAD", k_min, k_max, r, n_samples, bins, seed)

# 4 rounds
plot_weights("ADAD", k_min, k_max, r, n_samples, bins, seed)
plot_weights("DADA", k_min, k_max, r, n_samples, bins, seed)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Not used anymore
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_weight_k_vector(k, n):
    """
    Returns a uniformly random length n bitstring with weight k.
    """
    v = np.zeros(n)
    v[Random.sample(Random(), range(n), weight)] = 1
    return v

def create_quadratic_label(quad, slope, intercept):
    label = "$"
    round = float("{:.2g}".format(quad))
    label += "{}x^2 + ".format(round) if round != 0 else ""
    round = float("{:.2g}".format(slope))
    label += "{}x + ".format(round) if round != 0 else ""
    round = float("{:.2g}".format(intercept))
    label += "{}$".format(round) if round != 0 else " $"
    return label

def to_sys_form(G):
    """
    Rewrites the given n x k generator matrix to systematic form.
    TODO this doens't work sometimes when rref gives something where there isn't
    a perfect identiy matrix.
    """
    G = G.T # Transpose so that we reduce cols, not rows.
    G_rref, pivots = Matrix(G).rref()
    G_rref = np.array(G_rref.tolist()).astype(int) % 2
    return G_rref.T # Transpose to go back to G in sys. form.

def get_dual(G):
    """
    Given the n x k generator matrix of a linear code, returns the genetaor
    matrix for its dual code. That is, fist rewrites G in systematic form. Then
    use that to create the (n-k) x n parity check matrix. Then transpose that
    to get the generator of the dual code.
    """
    n, k = G.shape
    G = to_sys_form(G)
    H = np.zeros((n-k, n))
    H[:k, :k] = G[n-k:, :]
    H[:, k:] = np.eye(n-k)
    return H.T # Transpose parity check matrix to get generator for dual code.
