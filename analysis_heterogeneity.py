import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
male_data = pd.read_csv("focal_user_df_males.csv")
female_data = pd.read_csv("focal_user_df_females.csv")

print("Males: {:,} obs, Females: {:,} obs".format(
    male_data.shape[0], female_data.shape[0]))

# Pre-treatment variables and outcomes
pre_vars = [
    "view_sent_cnt_1", "view_rcvd_cnt_1",
    "msg_sent_cnt_1", "msg_rcvd_cnt_1",
    "match_sent_cnt_1", "match_rcvd_cnt_1",
]
outcomes = [
    "view_sent_cnt_2", "view_rcvd_cnt_2",
    "msg_sent_cnt_2", "msg_rcvd_cnt_2",
    "match_sent_cnt_2", "match_rcvd_cnt_2",
]
outcome_labels = [
    "ViewsSent", "ViewsReceived",
    "MessagesSent", "MessagesReceived",
    "MatchesSent", "MatchesReceived",
]


def add_pc1(df):
    """Compute PC1 from pre-treatment activity."""
    df = df.copy()
    mat = df[pre_vars].values
    mask = np.all(np.isfinite(mat), axis=1)
    scaler = StandardScaler()
    pc1 = np.full(len(df), np.nan)
    scaled = scaler.fit_transform(mat[mask])
    pca = PCA(n_components=1)
    pc1[mask] = pca.fit_transform(scaled).ravel()
    df["pc1_pre"] = pc1
    return df


male_data = add_pc1(male_data)
female_data = add_pc1(female_data)


def add_activity_tercile(df):
    """Pre-treatment activity tercile."""
    df = df.copy()
    total_pre = df[pre_vars].sum(axis=1)
    df["pre_activity_total"] = total_pre
    df["activity_terc"] = pd.qcut(total_pre, 3, labels=["low", "mid", "high"])
    return df


male_data = add_activity_tercile(male_data)
female_data = add_activity_tercile(female_data)


def simplify_ethnicity(df):
    """Ethnicity simplification (top groups + other)."""
    df = df.copy()
    top = ["white", "black", "asian", "latin"]
    df["eth_group"] = df["ethnicity"].apply(
        lambda x: x if x in top else "other_eth")
    return df


male_data = simplify_ethnicity(male_data)
female_data = simplify_ethnicity(female_data)


def simplify_edu(df):
    """Education simplification."""
    df = df.copy()
    mapping = {
        "HighSchool": "HS_or_less",
        "TwoYear": "HS_or_less",
        "University": "University",
        "PostGrad": "PostGrad",
        "SpaceCamp": "PostGrad",
        "unknown": "unknown_edu",
    }
    df["edu_group"] = df["edu_level"].map(mapping)
    return df


male_data = simplify_edu(male_data)
female_data = simplify_edu(female_data)


def fit_nb_glm(df, formula, outcome):
    """Fit GLM with NB family. First estimate alpha, then refit."""
    dfw = df.dropna(subset=[outcome, "pc1_pre"]).copy()
    try:
        pois = smf.glm(
            formula, data=dfw, family=sm.families.Poisson()).fit()
        mu = pois.fittedvalues
        y = dfw[outcome].values
        alpha_hat = max(((((y - mu)**2 - mu) / mu**2).mean()), 0.01)
    except Exception:
        alpha_hat = 1.0
    try:
        model = smf.glm(
            formula, data=dfw,
            family=sm.families.NegativeBinomial(alpha=alpha_hat))
        result = model.fit()
        return result, len(dfw)
    except Exception:
        return None, len(dfw)


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_coef(coef, se, pval):
    if pd.isna(coef) or pd.isna(se):
        return "  --  "
    return "{:+.3f}{:3s} ({:.3f})".format(coef, stars(pval), se)


def run_subgroup_analysis(datasets, group_var, group_levels, title):
    """Run NB GLM per subgroup, print treatment coefficient table."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("Each cell: NB GLM treatment coefficient (SE). "
          "Controls: body_type, edu_level, ethnicity, pc1_pre")
    print("*p<0.1, **p<0.05, ***p<0.01\n")

    for gender_label, data in datasets:
        print("--- {} ---".format(gender_label))
        dist = data[group_var].value_counts()
        print("  Distribution: {}".format(dict(dist)))

        header = "  {:15s} {:>6s}".format("Group", "N")
        for lab in outcome_labels:
            header += "  {:>22s}".format(lab)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for grp in group_levels:
            sub = data[data[group_var] == grp].copy()
            n = len(sub)
            row = "  {:15s} {:6d}".format(str(grp), n)
            for out in outcomes:
                formula = (
                    "{} ~ manipulation + C(body_type) "
                    "+ C(edu_level) + C(ethnicity) + pc1_pre".format(out))
                result, nobs = fit_nb_glm(sub, formula, out)
                if (result is not None
                        and "manipulation" in result.params.index):
                    c = result.params["manipulation"]
                    s = result.bse["manipulation"]
                    p = result.pvalues["manipulation"]
                    row += "  {:>22s}".format(fmt_coef(c, s, p))
                else:
                    row += "  {:>22s}".format("FAIL")
            print(row)
        print()


def run_interaction_analysis(
        datasets, interact_var, ref_level, other_levels, title):
    """Run NB GLM with treatment x interact_var interaction."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("Model: outcome ~ manipulation * C({}, ref='{}')".format(
        interact_var, ref_level))
    print("         + C(body_type) + C(edu_level) + C(ethnicity) + pc1_pre")
    print("*p<0.1, **p<0.05, ***p<0.01\n")

    for gender_label, data in datasets:
        print("--- {} (N={:,}) ---".format(gender_label, len(data)))

        rows = []
        for out, label in zip(outcomes, outcome_labels):
            data_c = data.copy()
            data_c[interact_var] = pd.Categorical(
                data_c[interact_var],
                categories=[ref_level] + other_levels)
            formula = (
                "{} ~ manipulation * C({}) "
                "+ C(body_type) + C(edu_level) "
                "+ C(ethnicity) + pc1_pre".format(out, interact_var))
            result, nobs = fit_nb_glm(data_c, formula, out)
            if result is None:
                continue

            params = result.params
            bse = result.bse
            pvals = result.pvalues

            row = {"outcome": label, "N": nobs}
            if "manipulation" in params.index:
                row["Treatment (base)"] = fmt_coef(
                    params["manipulation"],
                    bse["manipulation"],
                    pvals["manipulation"])
            for lev in other_levels:
                key = "manipulation:C({})[T.{}]".format(interact_var, lev)
                if key in params.index:
                    row["Trt x {}".format(lev)] = fmt_coef(
                        params[key], bse[key], pvals[key])
                else:
                    row["Trt x {}".format(lev)] = "--"
            rows.append(row)

        df_res = pd.DataFrame(rows).set_index("outcome")
        print(df_res.to_string())
        print()


# Define datasets
datasets = [("FEMALE", female_data), ("MALE", male_data)]

# 1. ETHNICITY
run_subgroup_analysis(
    datasets, "eth_group",
    ["white", "black", "asian", "latin", "other_eth"],
    "ANALYSIS 1a: SUBGROUP TREATMENT EFFECTS BY ETHNICITY")

run_interaction_analysis(
    datasets, "eth_group", "white",
    ["black", "asian", "latin", "other_eth"],
    "ANALYSIS 1b: INTERACTION MODEL -- TREATMENT x ETHNICITY")

# 2. EDUCATION
run_subgroup_analysis(
    datasets, "edu_group",
    ["HS_or_less", "University", "PostGrad", "unknown_edu"],
    "ANALYSIS 2a: SUBGROUP TREATMENT EFFECTS BY EDUCATION")

run_interaction_analysis(
    datasets, "edu_group", "HS_or_less",
    ["University", "PostGrad", "unknown_edu"],
    "ANALYSIS 2b: INTERACTION MODEL -- TREATMENT x EDUCATION")

# 3. PRE-TREATMENT ACTIVITY LEVEL
run_subgroup_analysis(
    datasets, "activity_terc",
    ["low", "mid", "high"],
    "ANALYSIS 3a: SUBGROUP TREATMENT EFFECTS BY PRE-TREATMENT ACTIVITY")

run_interaction_analysis(
    datasets, "activity_terc", "low",
    ["mid", "high"],
    "ANALYSIS 3b: INTERACTION MODEL -- TREATMENT x ACTIVITY LEVEL")

# 4. OWN DESIRABILITY
run_subgroup_analysis(
    datasets, "atrct_own_msg_tert",
    ["low", "mid", "high"],
    "ANALYSIS 4a: SUBGROUP TREATMENT EFFECTS BY OWN DESIRABILITY")

run_interaction_analysis(
    datasets, "atrct_own_msg_tert", "low",
    ["mid", "high"],
    "ANALYSIS 4b: INTERACTION MODEL -- TREATMENT x OWN DESIRABILITY")


# 5. SUMMARY
print("\n" + "=" * 90)
print("SUMMARY: COLLECTING ALL SUBGROUP TREATMENT EFFECTS")
print("=" * 90)
print("Goal: identify the groups with highest and lowest treatment effects\n")

all_results = []

for gender_label, data in datasets:
    for dim_name, dim_var, dim_levels in [
        ("Ethnicity", "eth_group",
         ["white", "black", "asian", "latin", "other_eth"]),
        ("Education", "edu_group",
         ["HS_or_less", "University", "PostGrad", "unknown_edu"]),
        ("Activity", "activity_terc",
         ["low", "mid", "high"]),
        ("Desirability", "atrct_own_msg_tert",
         ["low", "mid", "high"]),
    ]:
        for grp in dim_levels:
            sub = data[data[dim_var] == grp].copy()
            for out, label in zip(outcomes, outcome_labels):
                formula = (
                    "{} ~ manipulation + C(body_type) "
                    "+ C(edu_level) + C(ethnicity) + pc1_pre".format(out))
                result, nobs = fit_nb_glm(sub, formula, out)
                if (result is not None
                        and "manipulation" in result.params.index):
                    all_results.append({
                        "gender": gender_label,
                        "dimension": dim_name,
                        "group": grp,
                        "outcome": label,
                        "coef": result.params["manipulation"],
                        "se": result.bse["manipulation"],
                        "pval": result.pvalues["manipulation"],
                        "n": nobs,
                    })

df_all = pd.DataFrame(all_results)
df_all["sig"] = df_all["pval"].apply(stars)

print("\n=== STRONGEST POSITIVE TREATMENT EFFECTS (significant) ===")
sig = df_all[df_all["pval"] < 0.10].sort_values("coef", ascending=False)
for _, r in sig.head(20).iterrows():
    print("  {:6s} | {:12s} | {:15s} | {:18s} | coef={:+.3f}{:3s} "
          "(SE={:.3f}) | N={}".format(
              r["gender"], r["dimension"], r["group"], r["outcome"],
              r["coef"], r["sig"], r["se"], r["n"]))

print("\n=== STRONGEST NEGATIVE TREATMENT EFFECTS (significant) ===")
sig_neg = df_all[df_all["pval"] < 0.10].sort_values("coef", ascending=True)
for _, r in sig_neg.head(20).iterrows():
    print("  {:6s} | {:12s} | {:15s} | {:18s} | coef={:+.3f}{:3s} "
          "(SE={:.3f}) | N={}".format(
              r["gender"], r["dimension"], r["group"], r["outcome"],
              r["coef"], r["sig"], r["se"], r["n"]))

print("\n\n" + "=" * 90)
print("ALL ANALYSES COMPLETE")
print("=" * 90)