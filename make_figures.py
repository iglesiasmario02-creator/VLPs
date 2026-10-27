"""Presentation figures (English labels) for the VLP–protein analysis.

Reads the outputs of Projecte_VLP_Proteins_final.ipynb (resultats/), the raw
.lif movies and the u-track files, and writes PNG (300 dpi) + PDF to figures/.
The helper functions mirror the ones in the notebook.

    python make_figures.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from readlif.reader import LifFile
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import ransac, regionprops
from skimage.segmentation import expand_labels
from skimage.transform import AffineTransform, warp

BASE = Path(__file__).resolve().parent
RES = BASE / "resultats"
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

PROJECTS = {
    "P1": ("20241026  Project 1  E1E2 HiLO.lif", "20241026  Project 1  E1E2 HiLO"),
    "P2": ("20241026 Project  2 E1E2 HiLO.lif", "20241026 Project  2 E1E2 HiLO"),
}
HALF = 600
VALID_COLS = (14, 590)
PIXEL_UM = 1 / 9.118252
BLOCK = 5
TOLERANCE = 2
MIN_CONTACT = 3
EXAMPLE = ("P2", 3)   #movie used for the example images (enrichment and D ratio close to the median)

#colours: validated categorical slots 1-2 (blue/orange) + neutral grey for the control
C_CONTACT = "#eb6834"
C_NOCONTACT = "#2a78d6"
C_CONTROL = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "lines.linewidth": 2,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", name)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- data helpers
_lifs = {}
def lif(proj):
    if proj not in _lifs:
        _lifs[proj] = LifFile(str(BASE / PROJECTS[proj][0]))
    return _lifs[proj]

def frame(img, t):
    return np.asarray(img.get_frame(z=0, t=t, c=0), dtype=np.float32)

def normalise(img, sigma_bg=15):
    res = img - ndi.gaussian_filter(img, sigma_bg)
    sd = 1.4826 * np.median(np.abs(res - np.median(res)))
    return res / max(sd, 1e-6)

def tracks_file(proj, s):
    return next((BASE / PROJECTS[proj][1]).glob(f"seq_s{s:02d}/analysis/*_tracking.txt"))

def masks(proj, s):
    return np.load(RES / "mascares" / f"{proj}_s{s:02d}.npz")["etiquetes"]

def detect_spots(img, n_sigma=8, sigma_bg=10):
    res = img - ndi.gaussian_filter(img, sigma_bg)
    sd = 1.4826 * np.median(np.abs(res - np.median(res)))
    pk = peak_local_max(ndi.gaussian_filter(res, 1), min_distance=3, threshold_abs=n_sigma * sd, exclude_border=3)
    yy, xx = np.mgrid[-2:3, -2:3]
    out = []
    for y, x in pk:
        c = res[y - 2:y + 3, x - 2:x + 3]
        c = c - c.min()
        out.append((x + (c * xx).sum() / c.sum(), y + (c * yy).sum() / c.sum()))
    return np.array(out)

def bead_registration():
    lf = lif("P1")
    idx = [i for i, im in enumerate(lf.get_iter_image()) if "beads" in im.name.lower()][0]
    beads = frame(lf.get_image(idx), 0)
    pl, pr = detect_spots(beads[:, :HALF]), detect_spots(beads[:, HALF:])
    A = AffineTransform()
    for tol in [4, 2, 1.5]:
        d, j = cKDTree(pl).query(A(pr), distance_upper_bound=tol)
        ok = np.isfinite(d)
        A, inl = ransac((pr[ok], pl[j[ok]]), AffineTransform, min_samples=3,
                        residual_threshold=1.5, max_trials=2000, rng=0)
    m = re.search(r'RoiLeftHightY="(\d+)"', lf.xml_header)
    y_roi = int(m.group(1)) if m else 344
    S = np.array([[1, 0, 0], [0, 1, y_roi], [0, 0, 1]], float)
    return beads, A, AffineTransform(matrix=np.linalg.inv(S) @ A.params @ S)

def load_proteins(txt, A):
    df = pd.read_csv(txt, sep=r"\s+", header=None, names=["idx", "track", "frame", "x", "y", "c5", "c6", "amp"])
    df = df[["track", "frame", "x", "y"]].copy()
    df["frame"] -= 1
    df[["x", "y"]] -= 0.5
    x0, x1 = VALID_COLS
    inside = df["x"].between(HALF + x0, HALF + x1 - 1)
    df = df[inside.groupby(df["track"]).transform("all")].copy()
    df["x"] -= HALF
    xy = A(df[["x", "y"]].to_numpy())
    df["xv"], df["yv"] = xy[:, 0], xy[:, 1]
    return df.sort_values(["track", "frame"]).reset_index(drop=True)

def contact(df, labels_exp):
    nb, H, W = labels_exp.shape
    xi, yi = np.round(df["xv"]).astype(int).to_numpy(), np.round(df["yv"]).astype(int).to_numpy()
    b = np.minimum(df["frame"].to_numpy() // BLOCK, nb - 1)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    v = np.zeros(len(df), np.int32)
    v[ok] = labels_exp[b[ok], yi[ok], xi[ok]]
    return v

def classify(df):
    c = (df["vlp"] > 0).to_numpy()
    new = (df["track"].ne(df["track"].shift()) | df["frame"].diff().ne(1)
           | pd.Series(c).ne(pd.Series(c).shift()).to_numpy())
    grp = new.cumsum()[c]
    longest = df[c].groupby(grp.to_numpy())["track"].agg(["first", "size"]).groupby("first")["size"].max()
    m = df["track"].map(longest).fillna(0)
    return np.where(m >= MIN_CONTACT, "contact", np.where(m > 0, "brief", "none"))


# --------------------------------------------------------------------------- figures
def fig_raw_frame():
    img = lif(EXAMPLE[0]).get_image(EXAMPLE[1] - 1)
    f = frame(img, 0)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    for a, half, title in [(ax[0], f[:, :HALF], "VLP channel (left half)"),
                           (ax[1], f[:, HALF:], "Protein channel (right half)")]:
        a.imshow(half, cmap="gray", vmin=np.percentile(half, 1), vmax=np.percentile(half, 99.7))
        a.set_title(title)
        a.axis("off")
    scalebar(ax[0], 10)
    save(fig, "01_raw_frame")

def scalebar(ax, um, color="white"):
    px = um / PIXEL_UM
    ax.plot([20, 20 + px], [490, 490], color=color, lw=4, solid_capstyle="butt")
    ax.text(20 + px / 2, 478, f"{um} µm", color=color, ha="center", va="bottom", fontsize=14)

def fig_sim_vs_real():
    import deeptrack as dt
    IMG = 128
    optics = dt.Fluorescence(wavelength=600 * dt.units.nm, NA=0.75, magnification=1,
                             resolution=PIXEL_UM * dt.units.um, output_region=(0, 0, IMG, IMG))
    amp1 = np.array(optics(dt.PointParticle(position=(IMG / 2, IMG / 2), intensity=1, z=0))()).max()
    np.random.seed(3)
    particle = dt.PointParticle(position=lambda: np.random.uniform(2, IMG - 2, size=2),
                                intensity=lambda: np.exp(np.random.uniform(np.log(4), np.log(80))) / amp1,
                                z=lambda: np.random.uniform(-0.3, 0.3) * dt.units.um)
    particles = particle ^ (lambda: np.random.randint(3, 8))
    def background_noise(image):
        image = np.asarray(image, dtype=np.float32)
        yy, xx = np.mgrid[:IMG, :IMG]
        bg = np.zeros((IMG, IMG), np.float32)
        for _ in range(3):
            cy, cx = np.random.uniform(0, IMG, 2)
            s = np.random.uniform(4, 20)
            bg += np.random.uniform(0.5, 8) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        return image + bg[..., None] + np.random.normal(0, 1, image.shape).astype(np.float32)
    sim = normalise(np.array((optics(particles) >> background_noise)())[..., 0])

    img = lif(EXAMPLE[0]).get_image(EXAMPLE[1] - 1)
    x0, x1 = VALID_COLS
    real_full = normalise(np.mean([frame(img, t)[:, x0:x1] for t in range(BLOCK)], 0))
    lab0 = masks(*EXAMPLE)[0][:, x0:x1]
    best = max(regionprops(lab0, intensity_image=real_full), key=lambda p: p.intensity_max)
    cy, cx = np.clip(np.round(best.centroid).astype(int), IMG // 2, np.array(real_full.shape) - IMG // 2)
    real = real_full[cy - IMG // 2:cy + IMG // 2, cx - IMG // 2:cx + IMG // 2]

    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.25)
    for k, (im, title) in enumerate([(real, "Real VLPs"), (sim, "Simulated VLPs")]):
        a = fig.add_subplot(gs[k])
        a.imshow(im, cmap="gray", vmin=-3, vmax=15)
        a.set_title(title)
        a.axis("off")
    a = fig.add_subplot(gs[2])
    bins = np.linspace(-5, 30, 80)
    a.hist(real_full.ravel(), bins=bins, density=True, color=C_NOCONTACT, alpha=0.55, label="Real")
    a.hist(sim.ravel(), bins=bins, density=True, histtype="step", color=INK, lw=2, label="Simulated")
    a.set_yscale("log")
    a.set_xlabel("Pixel value (signal / noise)")
    a.set_ylabel("Density")
    a.legend()
    a.set_title("Same intensity distribution")
    save(fig, "02_sim_vs_real")

def fig_vlp_masks():
    img = lif(EXAMPLE[0]).get_image(EXAMPLE[1] - 1)
    x0, x1 = VALID_COLS
    mean = np.mean([frame(img, t)[:, :HALF] for t in range(0, img.dims.t, 10)], 0)
    lab = masks(*EXAMPLE)
    norm = normalise(mean[:, x0:x1])
    ext = (x0 - 0.5, x1 - 0.5, mean.shape[0] - 0.5, -0.5)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6))
    for a in ax:
        a.imshow(norm, cmap="gray", vmin=-3, vmax=20, extent=ext)
        a.axis("off")
    ax[1].contour(np.arange(HALF), np.arange(mean.shape[0]), (lab > 0).any(0), [0.5], colors=C_CONTACT, linewidths=1.2)
    ax[0].set_title("VLP channel (movie average)")
    ax[1].set_title(f"UNet masks ({len(np.unique(lab)) - 1} VLPs)")
    save(fig, "03_vlp_masks")

def fig_registration(beads, A):
    reg = warp(beads[:, HALF:], A.inverse, preserve_range=True)
    def sc(a):
        lo, hi = np.percentile(a, [1, 99.8])
        return np.clip((a - lo) / (hi - lo), 0, 1)
    left = sc(beads[:, :HALF])
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    for a, right, title in [(ax[0], beads[:, HALF:], "Before registration"), (ax[1], reg, "After registration")]:
        rgb = np.dstack([left, sc(right), left])[400:700, 200:500]
        a.imshow(rgb)
        a.set_title(title)
        a.axis("off")
    fig.text(0.5, 0.02, "Beads: left channel in magenta, right channel in green (overlap = white)",
             ha="center", color=INK2, fontsize=14)
    save(fig, "04_bead_registration")

def fig_trajectories(A):
    proj, s = EXAMPLE
    img = lif(proj).get_image(s - 1)
    lab = masks(proj, s)
    df = load_proteins(tracks_file(proj, s), A)
    df["vlp"] = contact(df, np.stack([expand_labels(e, TOLERANCE) for e in lab]))
    df["cls"] = classify(df)
    mean = np.mean([frame(img, t)[:, :HALF] for t in range(0, img.dims.t, 10)], 0)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(normalise(mean), cmap="gray", vmin=-3, vmax=20)
    rng = np.random.default_rng(0)
    for tr, g in df.groupby("track"):
        cl = g["cls"].iat[0]
        if cl == "none" and rng.random() > 0.1:
            continue
        ax.plot(g["xv"], g["yv"], lw=0.9 if cl != "none" else 0.6,
                color={"contact": C_CONTACT, "brief": "#eda100", "none": C_NOCONTACT}[cl])
    ax.set_xlim(VALID_COLS)
    ax.set_ylim(img.dims.y, 0)
    ax.axis("off")
    for lbl, col in [("In contact with a VLP (≥ 3 frames)", C_CONTACT), ("Brief contact", "#eda100"),
                     ("No contact (1 in 10 shown)", C_NOCONTACT)]:
        ax.plot([], [], color=col, lw=3, label=lbl)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=14, handlelength=1.5)
    scalebar(ax, 10)
    save(fig, "05_trajectories")
    return df

def fig_enrichment(res):
    piv = res.pivot_table(index=["projecte", "serie"], columns="cas", values="enriquiment")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (proj, s), r in piv.iterrows():
        ax.plot([0, 1], [r["control"], r["real"]], "-", color=GRID, lw=1.5, zorder=1)
    ax.scatter(np.zeros(len(piv)), piv["control"], s=70, color=C_CONTROL, zorder=2, edgecolor="white", lw=1.5)
    ax.scatter(np.ones(len(piv)), piv["real"], s=70, color=C_CONTACT, zorder=2, edgecolor="white", lw=1.5)
    ax.axhline(1, ls="--", color=INK2, lw=1)
    ax.text(1.13, 1, "no\npreference", va="center", color=INK2, fontsize=13)
    p = stats.wilcoxon(piv["real"], piv["control"]).pvalue
    n_up = int((piv["real"] > piv["control"]).sum())
    ax.set_xticks([0, 1], ["Control\n(rotated 180°)", "Real VLPs"])
    ax.set_xlim(-0.35, 1.5)
    ax.set_ylabel("Protein enrichment on VLPs\n(density inside / outside)")
    ax.set_title(f"Higher than control in {n_up} of {len(piv)} movies\nWilcoxon p = {p:.1g}", fontsize=15)
    ygrid(ax)
    save(fig, "06_enrichment")

def fig_D_distribution(tt):
    groups = [("real", "interacciona", "In contact", C_CONTACT),
              ("real", "cap contacte", "No contact", C_NOCONTACT),
              ("control", "interacciona", "Control:\n\"in contact\"", C_CONTROL)]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for a, proj, title in [(ax[0], "P1", "Project 1"), (ax[1], "P2", "Project 2")]:
        g = tt[tt["projecte"] == proj]
        data = [np.log10(g[(g["cas"] == c) & (g["classe"] == cl)]["D"]) for c, cl, _, _ in groups]
        vp = a.violinplot(data, showextrema=False, widths=0.8)
        for body, (_, _, _, col) in zip(vp["bodies"], groups):
            body.set_facecolor(col)
            body.set_alpha(0.35)
            body.set_edgecolor("none")
        for k, (d, (_, _, _, col)) in enumerate(zip(data, groups), start=1):
            med = np.median(d)
            a.plot([k - 0.25, k + 0.25], [med, med], color=col, lw=3)
            a.text(k + 0.3, med, f"{10 ** med:.3f}", va="center", fontsize=13, color=INK)
        a.set_xticks(range(1, 4), [f"{lbl}\n(n = {len(d)})" for (_, _, lbl, _), d in zip(groups, data)], fontsize=13)
        a.set_title(title)
        ygrid(a)
    ax[0].set_ylabel("Diffusion coefficient D (µm²/s)")
    ticks = np.arange(-4, 1)
    ax[0].set_yticks(ticks, [f"$10^{{{t}}}$" for t in ticks])
    ax[0].set_ylim(-4, 0.7)
    fig.suptitle("Diffusion coefficient per trajectory (bar and number = median, µm²/s)", y=1.03, fontsize=15)
    save(fig, "07_D_distribution")

def fig_D_ratio(tt):
    med = tt.groupby(["projecte", "serie", "cas", "classe"])["D"].median().unstack(["cas", "classe"])
    real = med[("real", "interacciona")] / med[("real", "cap contacte")]
    ctrl = med[("control", "interacciona")] / med[("control", "cap contacte")]
    p = stats.wilcoxon(real, ctrl).pvalue
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for a_, b_ in zip(ctrl, real):
        ax.plot([0, 1], [a_, b_], "-", color=GRID, lw=1.5, zorder=1)
    ax.scatter(np.zeros(len(ctrl)), ctrl, s=70, color=C_CONTROL, zorder=2, edgecolor="white", lw=1.5)
    ax.scatter(np.ones(len(real)), real, s=70, color=C_CONTACT, zorder=2, edgecolor="white", lw=1.5)
    ax.axhline(1, ls="--", color=INK2, lw=1)
    ax.text(1.13, 1, "same\nspeed", va="center", color=INK2, fontsize=13)
    ax.set_xticks([0, 1], ["Control\n(rotated 180°)", "Real VLPs"])
    ax.set_xlim(-0.35, 1.5)
    ax.set_ylabel("D (in contact) / D (no contact)\nmedian per movie")
    ax.set_title(f"Slower in contact: lower than control in {int((real < ctrl).sum())} of {len(real)} movies\n"
                 f"Wilcoxon p = {p:.1g}", fontsize=15)
    ygrid(ax)
    save(fig, "08_D_ratio_per_movie")

def fig_steps(A, series):
    rows = []
    for proj, s, dt_s in series:
        lab = masks(proj, s)
        df = load_proteins(tracks_file(proj, s), A)
        df["vlp"] = contact(df, np.stack([expand_labels(e, TOLERANCE) for e in lab]))
        df["cls"] = classify(df)
        g = df.groupby("track")
        ok = (g["frame"].diff() == 1).to_numpy()
        r2 = ((g["x"].diff() ** 2 + g["y"].diff() ** 2) * PIXEL_UM ** 2).to_numpy()
        on = (df["vlp"] > 0).to_numpy() & (g["vlp"].shift() > 0).to_numpy()
        off = (df["vlp"] == 0).to_numpy() & (g["vlp"].shift() == 0).to_numpy()
        for name, sel in [("On a VLP", on & ok),
                          ("Same proteins,\noff the VLP", off & ok & (df["cls"] == "contact").to_numpy()),
                          ("Never in contact", off & ok & (df["cls"] == "none").to_numpy())]:
            rows.append(dict(projecte=proj, serie=s, tipus=name, D=np.mean(r2[sel]) / (4 * dt_s)))
        print(f"  steps {proj} s{s:02d}")
    st = pd.DataFrame(rows)
    order = ["On a VLP", "Same proteins,\noff the VLP", "Never in contact"]
    cols = [C_CONTACT, "#f2a27f", C_NOCONTACT]
    piv = st.pivot_table(index=["projecte", "serie"], columns="tipus", values="D")[order]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, r in piv.iterrows():
        ax.plot(range(3), r.values, "-", color=GRID, lw=1.2, zorder=1)
    for k, (name, col) in enumerate(zip(order, cols)):
        ax.scatter(np.full(len(piv), k), piv[name], s=60, color=col, zorder=2, edgecolor="white", lw=1.5)
        m = piv[name].median()
        ax.plot([k - 0.22, k + 0.22], [m, m], color=INK, lw=2.5, zorder=3)
        ax.text(k + 0.26, m, f"{m:.2f}", va="center", fontsize=13)
    ax.set_xticks(range(3), order)
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("Step-based D (µm²/s), mean per movie")
    ax.set_title("The slowdown happens only on the VLP", fontsize=15)
    ygrid(ax)
    save(fig, "09_steps_on_off_vlp")

def fig_durations(ep):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for a, proj, title in [(ax[0], "P1", "Project 1"), (ax[1], "P2", "Project 2")]:
        for cas, col, lbl in [("real", C_CONTACT, "Real VLPs"), ("control", C_CONTROL, "Control (chance)")]:
            d = np.sort(ep[(ep["projecte"] == proj) & (ep["cas"] == cas)]["durada_s"].to_numpy())
            a.step(d, 1 - np.arange(len(d)) / len(d), where="post", color=col,
                   label=f"{lbl}: mean {d.mean():.2f} s")
        a.set_yscale("log")
        a.set_xlim(0, 6)
        a.set_xlabel("Contact duration (s)")
        a.set_title(title)
        a.legend(fontsize=13)
        ygrid(a)
    ax[0].set_ylabel("Fraction of contacts lasting longer")
    save(fig, "10_contact_durations")


if __name__ == "__main__":
    res = pd.read_csv(RES / "resum_series.csv")
    tt = pd.read_csv(RES / "trajectories.csv")
    tt = tt[tt["D"] > 0].dropna(subset=["D"])
    ep = pd.read_csv(RES / "episodis_contacte.csv")

    beads, A_beads, A_video = bead_registration()
    fig_raw_frame()
    fig_sim_vs_real()
    fig_vlp_masks()
    fig_registration(beads, A_beads)
    fig_trajectories(A_video)
    fig_enrichment(res)
    fig_D_distribution(tt)
    fig_D_ratio(tt)
    fig_durations(ep)
    series = [(p, int(s), 1 / lif(p).get_image(int(s) - 1).scale[3])
              for p, s in res[["projecte", "serie"]].drop_duplicates().itertuples(index=False)]
    fig_steps(A_video, series)
