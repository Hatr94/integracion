import numpy as np

paths = {
    "NPU0_fixed": "/Users/hat/integracion/run_test_jj_20260309_100436/antikt_R0.4/jets_test_jj_antikt_R0.4.npy",
    "NPU5_fixed": "/Users/hat/integracion/run_test_jj_20260309_100733/antikt_R0.4/jets_test_jj_antikt_R0.4.npy",
    "NPU20_fixed": "/Users/hat/integracion/run_test_jj_20260309_100942/antikt_R0.4/jets_test_jj_antikt_R0.4.npy",
    "NPU50_fixed": "/Users/hat/integracion/run_test_jj_20260309_101148/antikt_R0.4/jets_test_jj_antikt_R0.4.npy",
    "Poisson_mu20": "/Users/hat/integracion/run_test_jj_20260309_103802/antikt_R0.4/jets_test_jj_antikt_R0.4.npy",
}

cols = {
    "pt_gen": 0,
    "m_gen": 3,
    "recoPt": 6,
    "recoMass": 21,
    "n_const": 10,
    "ncharged": 17,
    "nneutral": 18,
    "nef": 11,
    "nhf": 12,
    "cef": 13,
    "chf": 14,
}

def stats(a):
    return np.mean(a), np.median(a), np.percentile(a, 90)

print("=" * 110)
print(
    f"{'sample':<15} {'jets':>8} "
    f"{'pt_mean':>10} {'m_mean':>10} {'recoPt_mean':>12} {'recoM_mean':>12} "
    f"{'nconst_mean':>12} {'nch_mean':>10} {'nneu_mean':>10}"
)
print("=" * 110)

for name, path in paths.items():
    x = np.load(path)
    jets = len(x)

    pt_mean = np.mean(x[:, cols["pt_gen"]])
    m_mean = np.mean(x[:, cols["m_gen"]])
    rpt_mean = np.mean(x[:, cols["recoPt"]])
    rm_mean = np.mean(x[:, cols["recoMass"]])
    nc_mean = np.mean(x[:, cols["n_const"]])
    nch_mean = np.mean(x[:, cols["ncharged"]])
    nneu_mean = np.mean(x[:, cols["nneutral"]])

    print(
        f"{name:<15} {jets:>8d} "
        f"{pt_mean:>10.3f} {m_mean:>10.3f} {rpt_mean:>12.3f} {rm_mean:>12.3f} "
        f"{nc_mean:>12.3f} {nch_mean:>10.3f} {nneu_mean:>10.3f}"
    )

print("=" * 110)
print("\nFracciones energéticas medias")
print("=" * 70)
print(f"{'sample':<15} {'nef':>10} {'nhf':>10} {'cef':>10} {'chf':>10}")
print("=" * 70)

for name, path in paths.items():
    x = np.load(path)
    nef = np.mean(x[:, cols["nef"]])
    nhf = np.mean(x[:, cols["nhf"]])
    cef = np.mean(x[:, cols["cef"]])
    chf = np.mean(x[:, cols["chf"]])

    print(f"{name:<15} {nef:>10.4f} {nhf:>10.4f} {cef:>10.4f} {chf:>10.4f}")

print("=" * 70)