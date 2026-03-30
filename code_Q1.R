# TECH-GB 3345 (Digital and AI Economics)
# Submitted to: Prof. Arun Sundararajan
# Submitted by: Qiyu Dai, Ilias Triantafyllopoulos

# Note: This script contains the code for our answer in Q1

# Some preliminary commands
rm(list = ls()) # clearing the workspace
setwd("/Users/qiyudai/Dropbox/coursework/PHD course/Yr1 Spring/IS seminar/Assignment 5b/Copy of Bapna et al 2023 Data")

PackageList =c('data.table', 'tidyverse', 'fixest','kableExtra','dplyr')
NewPackages=PackageList[!(PackageList %in%
                            installed.packages()[,"Package"])]
if(length(NewPackages)) install.packages(NewPackages)
lapply(PackageList,require,character.only=TRUE)

# ── Load data ──────────────────────────────────────────────────────────────────
male_data   <- fread("focal_user_df_males.csv")
female_data <- fread("focal_user_df_females.csv")

# ── Helper: run t-test for one outcome × manipulation group ───────────────────
run_ttest <- function(data, outcome_var, gender_label) {
  treat <- data[manipulation == 1, get(outcome_var)]
  ctrl  <- data[manipulation == 0, get(outcome_var)]
  
  tt <- t.test(treat, ctrl, var.equal = FALSE)   # Welch t-test
  
  data.frame(
    Gender          = gender_label,
    BasicOutcomes   = outcome_var,
    Manipulation    = c("Treatment", "Control"),
    Observations    = c(length(treat),   length(ctrl)),
    Mean            = c(mean(treat),     mean(ctrl)),
    Std.Error       = c(sd(treat) / sqrt(length(treat)),
                        sd(ctrl)  / sqrt(length(ctrl))),
    t.statistic     = c(round(tt$statistic, 4), NA),
    p.value         = c(round(tt$p.value,   4), NA),
    stringsAsFactors = FALSE
  )
}

# ── Outcomes to test (column names in the data) ───────────────────────────────
# Adjust these names if your CSV columns differ
outcomes <- c(
  "view_sent_cnt_1",    # ViewsSent
  "view_rcvd_cnt_1",    # ViewsReceived
  "msg_sent_cnt_1",     # MessagesSent
  "msg_rcvd_cnt_1",     # MessagesReceived
  "match_sent_cnt_1",   # MatchesSent
  "match_rcvd_cnt_1"    # MatchesReceived
)

pretty_names <- c(
  "ViewsSent", "ViewsReceived",
  "MessagesSent", "MessagesReceived",
  "MatchesSent", "MatchesReceived"
)

# ── Build results for females and males ───────────────────────────────────────
build_results <- function(data, gender_label) {
  rows <- lapply(outcomes, run_ttest, data = data, gender_label = gender_label)
  do.call(rbind, rows)
}

female_results <- build_results(female_data, "Females")
male_results   <- build_results(male_data,   "Males")

all_results <- rbind(female_results, male_results)

# Replace internal column names with pretty outcome names
all_results$BasicOutcomes <- rep(
  rep(pretty_names, each = 2), 2   # 2 rows (treat/ctrl) × 6 outcomes × 2 genders
)

# ── Round numeric columns (base R) ────────────────────────────────────────────
all_results$Mean        <- round(all_results$Mean,        3)
all_results$Std.Error   <- round(all_results$Std.Error,   3)
all_results$t.statistic <- round(all_results$t.statistic, 4)
all_results$p.value     <- round(all_results$p.value,     4)

# ── Print a clean table to console ────────────────────────────────────────────
cat("\nTable 1 – Randomization Check (Activity in Pre-Treatment Month)\n")
cat(strrep("-", 90), "\n")
print(all_results, row.names = FALSE)

