# TECH-GB 3345 (Digital and AI Economics)
# Submitted to: Prof. Arun Sundararajan
# Submitted by: Qiyu Dai, Ilias Triantafyllopoulos
# Note: Age Heterogeneity Analysis — NB Models, All Outcomes (Sent + Received)

rm(list = ls())
setwd("/Users/qiyudai/Dropbox/coursework/PHD course/Yr1 Spring/IS seminar/Assignment 5b/Copy of Bapna et al 2023 Data")

PackageList <- c('data.table', 'MASS', 'modelsummary', 'kableExtra', 'sandwich', 'lmtest')
NewPackages <- PackageList[!(PackageList %in% installed.packages()[,"Package"])]
if (length(NewPackages)) install.packages(NewPackages)
lapply(PackageList, require, character.only = TRUE)

# ── Load data ──────────────────────────────────────────────────────────────────
male_data   <- fread("focal_user_df_males.csv")
female_data <- fread("focal_user_df_females.csv")

# ── Age groups (Young 18-29 / Mid 30-49 / Old 50+) ───────────────────────────
create_age_group <- function(dt) {
  dt <- copy(dt)
  dt[, age_group3 := fcase(
    age_raw >= 18 & age_raw < 29, "young",
    age_raw >= 30 & age_raw < 49, "mid",
    age_raw >= 50,                 "old",
    default = NA_character_
  )]
  dt[, age_group3 := factor(age_group3, levels = c("young", "mid", "old"))]
  dt
}
female_data <- create_age_group(female_data)
male_data   <- create_age_group(male_data)

# ── Pre-treatment PC1 ─────────────────────────────────────────────────────────
compute_pc1 <- function(dt) {
  dt <- copy(dt)
  pre_vars <- c("view_sent_cnt_1", "view_rcvd_cnt_1",
                "msg_sent_cnt_1",  "msg_rcvd_cnt_1",
                "match_sent_cnt_1","match_rcvd_cnt_1")
  mat <- as.matrix(dt[, ..pre_vars])
  idx <- complete.cases(mat)
  pc1 <- rep(NA_real_, nrow(mat))
  pc1[idx] <- prcomp(mat[idx, ], scale. = TRUE)$x[, 1]
  dt[, pc1_pre := pc1]
  dt
}
female_data <- compute_pc1(female_data)
male_data   <- compute_pc1(male_data)

# ── Outcomes: all 6 (sent + received) ─────────────────────────────────────────
outcomes <- c(
  "view_sent_cnt_2",  "view_rcvd_cnt_2",
  "msg_sent_cnt_2",   "msg_rcvd_cnt_2",
  "match_sent_cnt_2", "match_rcvd_cnt_2"
)
outcome_labels <- c(
  "ViewsSent",     "ViewsReceived",
  "MessagesSent",  "MessagesReceived",
  "MatchesSent",   "MatchesReceived"
)

# ── Fit NB model with treatment x age_group interaction ───────────────────────
fit_nb <- function(data, outcome) {
  df  <- as.data.frame(data[!is.na(age_group3) & !is.na(get(outcome))])
  fml <- as.formula(paste0(
    outcome,
    " ~ manipulation * age_group3 +
      body_type + edu_level + ethnicity + pc1_pre"
  ))
  glm.nb(fml, data = df)
}

# Fit all 12 models (6 outcomes x 2 genders)
models_f <- lapply(outcomes, fit_nb, data = female_data)
models_m <- lapply(outcomes, fit_nb, data = male_data)
names(models_f) <- outcome_labels
names(models_m) <- outcome_labels

# ── Coefficient labels ────────────────────────────────────────────────────────
coef_map <- c(
  "manipulation"               = "Treatment",
  "age_group3mid"              = "Age: Mid (30-49)",
  "age_group3old"              = "Age: Old (50+)",
  "manipulation:age_group3mid" = "Treatment x Mid (30-49)",
  "manipulation:age_group3old" = "Treatment x Old (50+)"
)

# ── GOF rows ──────────────────────────────────────────────────────────────────
gof_map <- data.frame(
  raw   = "nobs",
  clean = "Observations",
  fmt   = 0,
  stringsAsFactors = FALSE
)

# ── Control variable checkmark rows ───────────────────────────────────────────
make_ctrl_rows <- function() {
  ctrl <- data.frame(
    term             = c("Body Types",
                         "Education Levels",
                         "Ethnicities",
                         "Pre-treatment activities (PC1 only)"),
    ViewsSent        = rep("\\checkmark", 4),
    ViewsReceived    = rep("\\checkmark", 4),
    MessagesSent     = rep("\\checkmark", 4),
    MessagesReceived = rep("\\checkmark", 4),
    MatchesSent      = rep("\\checkmark", 4),
    MatchesReceived  = rep("\\checkmark", 4),
    stringsAsFactors = FALSE
  )
  attr(ctrl, "position") <- c(11, 12, 13, 14)
  ctrl
}

# ── Print to console (markdown) ───────────────────────────────────────────────
print_table <- function(model_list, title_str) {
  modelsummary(
    model_list,
    vcov     = lapply(model_list, function(m) vcovHC(m, type = "HC3")),
    coef_map = coef_map,
    gof_map  = gof_map,
    stars    = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
    add_rows = make_ctrl_rows(),
    title    = title_str,
    notes    = "Robust SE in parentheses. Baseline age group = Young (18-29).",
    output   = "markdown"
  )
}

# ── Save as .tex ──────────────────────────────────────────────────────────────
save_table <- function(model_list, title_str, filename) {
  modelsummary(
    model_list,
    vcov     = lapply(model_list, function(m) vcovHC(m, type = "HC3")),
    coef_map = coef_map,
    gof_map  = gof_map,
    stars    = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
    add_rows = make_ctrl_rows(),
    title    = title_str,
    notes    = "Robust standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.1. Baseline age group = Young (18--29).",
    output   = filename
  )
  cat("Saved:", filename, "\n")
}

# ── Run ───────────────────────────────────────────────────────────────────────
cat("\n=== TABLE A: FEMALE ===\n")
print_table(models_f, "Table A - Female: NB Model for Age Heterogeneity (All Outcomes)")

cat("\n=== TABLE B: MALE ===\n")
print_table(models_m, "Table B - Male: NB Model for Age Heterogeneity (All Outcomes)")

save_table(models_f,
           "Table A - Female: NB Model for Age Heterogeneity (All Outcomes)",
           "table_age_female.tex")
save_table(models_m,
           "Table B - Male: NB Model for Age Heterogeneity (All Outcomes)",
           "table_age_male.tex")

