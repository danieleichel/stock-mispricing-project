# ==============================================================================
# Step 6b (R): robustness check of Step 6 with panel regressions
# ==============================================================================
#
# Step 6 (Python) used Fama-MacBeth: one cross-sectional regression per day,
# then the average slope with Newey-West standard errors.
#
# This script re-estimates the same tests a second, standard way:
#
#     forward residual return = slope * s-score + day fixed effect + error
#
# with standard errors clustered two ways, by stock and by day:
#   - by day:   stocks move together on the same day
#   - by stock: forward returns over h days overlap from one day to the next
#
# If both methods reach the same conclusions, the results do not depend on
# one estimation choice. Slopes are in basis points per 1 SD of s-score;
# negative = mispricings correct.
#
# Before touching the real data, the script runs a self-test on simulated
# stocks with a known answer and stops if it does not recover it.
#
# Setup (once, in R or RStudio):
#   install.packages(c("arrow", "fixest", "data.table"))
#
# Run from the project folder:
#   Rscript src/06b_panel_robustness.R              PCA-5 and FF-5
#   Rscript src/06b_panel_robustness.R pca5         one model
#   Rscript src/06b_panel_robustness.R --selftest   only the self-test
#
# Needs: data/processed/conditional_panel_<model>.parquet (from Step 6)
#
# Creates:
#   outputs/tables/panel_robustness_<model>.csv
#   outputs/tables/panel_vs_famamacbeth_<model>.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(fixest)
  library(arrow)
})

setFixest_notes(FALSE)

HORIZONS <- c(1, 5, 10, 20)
CONDITIONS <- c("news", "size", "sector", "stress", "ou_fit", "year")
DIFFERENCES <- list(
  list("news",   "news-like move in last 5 days", "no news-like move"),
  list("size",   "small (bottom third)",          "large (top third)"),
  list("stress", "stressed market",               "calm market"),
  list("ou_fit", "OU fit significant",            "OU fit not significant")
)
MIN_ROWS <- 500
PROCESSED_DIR <- file.path("data", "processed")
TABLES_DIR <- file.path("outputs", "tables")


# ------------------------------------------------------------------------------
# Estimation helpers
# ------------------------------------------------------------------------------

# Slope of forward return on s-score, with day fixed effects and two-way clustering.
fit_slope <- function(data, outcome) {
  data <- data[!is.na(get(outcome))]
  if (nrow(data) < MIN_ROWS) return(NULL)
  model <- tryCatch(
    feols(as.formula(paste(outcome, "~ s_score | date")),
          data = data, cluster = ~ date + ticker),
    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  ct <- coeftable(model)
  data.table(estimate_bps = ct["s_score", 1] * 1e4,
             se_bps       = ct["s_score", 2] * 1e4,
             t_stat       = ct["s_score", 3],
             p_value      = ct["s_score", 4],
             observations = nobs(model))
}

# Difference in slopes between group a and group b (interaction term).
fit_difference <- function(data, condition, a, b, outcome) {
  data <- data[get(condition) %in% c(a, b) & !is.na(get(outcome))]
  if (nrow(data) < MIN_ROWS) return(NULL)
  data[, in_a := as.integer(get(condition) == a)]
  model <- tryCatch(
    feols(as.formula(paste(outcome, "~ s_score + s_score:in_a | date")),
          data = data, cluster = ~ date + ticker),
    error = function(e) NULL)
  if (is.null(model)) return(NULL)
  ct <- coeftable(model)
  data.table(estimate_bps = ct["s_score:in_a", 1] * 1e4,
             se_bps       = ct["s_score:in_a", 2] * 1e4,
             t_stat       = ct["s_score:in_a", 3],
             p_value      = ct["s_score:in_a", 4],
             observations = nobs(model))
}

run_all_tests <- function(panel) {
  results <- list()

  for (h in HORIZONS) {
    outcome <- paste0("fwd_", h, "d")

    overall <- fit_slope(panel, outcome)
    if (!is.null(overall)) {
      results[[length(results) + 1]] <- cbind(
        data.table(type = "slope", condition = "all", group = "all stock-days",
                   horizon_days = h), overall)
    }

    for (condition in CONDITIONS) {
      for (group in sort(unique(na.omit(panel[[condition]])))) {
        est <- fit_slope(panel[get(condition) == group], outcome)
        if (!is.null(est)) {
          results[[length(results) + 1]] <- cbind(
            data.table(type = "slope", condition = condition, group = group,
                       horizon_days = h), est)
        }
      }
    }

    for (d in DIFFERENCES) {
      est <- fit_difference(panel, d[[1]], d[[2]], d[[3]], outcome)
      if (!is.null(est)) {
        results[[length(results) + 1]] <- cbind(
          data.table(type = "difference", condition = d[[1]],
                     group = paste(d[[2]], "minus", d[[3]]), horizon_days = h), est)
      }
    }
  }

  results <- rbindlist(results)
  # Benjamini-Hochberg q-values, separately for slopes and for differences,
  # matching the two families in Step 6.
  results[, q_value_bh := p.adjust(p_value, method = "BH"), by = type]
  results[]
}


# ------------------------------------------------------------------------------
# Self-test on simulated data with a known answer
# ------------------------------------------------------------------------------

self_test <- function() {
  cat("Self-test on simulated data... ")
  set.seed(1)
  n_days <- 400
  n_stocks <- 300
  sim <- CJ(date = as.Date("2023-01-02") + 0:(n_days - 1), ticker = sprintf("S%03d", 1:n_stocks))
  sim[, s_score := rnorm(.N)]
  sim[, group := fifelse(as.integer(substr(ticker, 2, 4)) <= n_stocks / 2, "A", "B")]
  day_shock <- sim[, .(shock = rnorm(1, sd = 0.02)), by = date]
  sim <- merge(sim, day_shock, by = "date")
  # True slopes: -10 bps per SD in group A, +10 bps in group B.
  true_slope <- fifelse(sim$group == "A", -0.0010, 0.0010)
  sim[, fwd_5d := true_slope * s_score + shock + rnorm(.N, sd = 0.01)]

  a <- fit_slope(sim[group == "A"], "fwd_5d")
  b <- fit_slope(sim[group == "B"], "fwd_5d")
  setnames(sim, "group", "test_group")
  diff <- fit_difference(sim, "test_group", "A", "B", "fwd_5d")

  ok <- abs(a$estimate_bps + 10) < 1.5 &&
        abs(b$estimate_bps - 10) < 1.5 &&
        abs(diff$estimate_bps + 20) < 2.5 &&
        a$t_stat < -10 && b$t_stat > 10
  if (!ok) {
    print(rbind(a, b, diff))
    stop("Self-test FAILED: the regressions did not recover the known slopes.")
  }
  cat(sprintf("passed (A: %.1f bps, B: %.1f bps, A minus B: %.1f bps; true: -10, +10, -20)\n",
              a$estimate_bps, b$estimate_bps, diff$estimate_bps))
}


# ------------------------------------------------------------------------------
# Compare with Step 6 (Fama-MacBeth)
# ------------------------------------------------------------------------------

compare_with_step6 <- function(results, model) {
  slopes_file <- file.path(TABLES_DIR, paste0("conditional_slopes_", model, ".csv"))
  diffs_file  <- file.path(TABLES_DIR, paste0("conditional_differences_", model, ".csv"))
  if (!file.exists(slopes_file) || !file.exists(diffs_file)) return(NULL)

  fm_slopes <- fread(slopes_file)[, .(type = "slope", condition, group = as.character(group),
                                      horizon_days, fm_bps = slope_bps_per_unit_s,
                                      fm_t = nw_t_stat, fm_q = q_value_bh)]
  fm_diffs <- fread(diffs_file)[, .(type = "difference", condition, group = comparison,
                                    horizon_days, fm_bps = difference_bps_per_unit_s,
                                    fm_t = nw_t_stat, fm_q = q_value_bh)]
  fm <- rbind(fm_slopes, fm_diffs)

  both <- merge(results[, .(type, condition, group, horizon_days,
                            panel_bps = estimate_bps, panel_t = t_stat, panel_q = q_value_bh)],
                fm, by = c("type", "condition", "group", "horizon_days"))

  # Keep tests that both methods could estimate (e.g. Fama-MacBeth needs 30
  # days, so a group that only exists for a few weeks has no Step 6 result).
  both <- both[!is.na(fm_t) & !is.na(panel_t)]

  # Same conclusion: both significant with the same sign, or both not significant
  # (using |t| >= 2 as the cutoff for both methods).
  both[, fm_sig := abs(fm_t) >= 2]
  both[, panel_sig := abs(panel_t) >= 2]
  both[, same_conclusion := (fm_sig == panel_sig) &
                            (!fm_sig | sign(fm_bps) == sign(panel_bps))]
  both[]
}


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

cat(strrep("=", 70), "\n")
cat("STEP 6b (R) - PANEL REGRESSION ROBUSTNESS CHECK\n")
cat(strrep("=", 70), "\n\n")

self_test()
if ("--selftest" %in% args) quit(save = "no")

models <- setdiff(args, "--selftest")
if (length(models) == 0) models <- c("pca5", "ff5")
dir.create(TABLES_DIR, recursive = TRUE, showWarnings = FALSE)

for (model in models) {

  cat("\n", strrep("=", 70), "\n", sep = "")
  cat("MODEL:", model, "\n")
  cat(strrep("=", 70), "\n")

  path <- file.path(PROCESSED_DIR, paste0("conditional_panel_", model, ".parquet"))
  if (!file.exists(path)) {
    stop("Missing ", path, ". Run python src/06_when_do_mispricings_correct.py first, ",
         "and run this script from the project folder.")
  }

  panel <- as.data.table(read_parquet(path))
  panel[, date := as.Date(date)]
  for (condition in CONDITIONS) set(panel, j = condition, value = as.character(panel[[condition]]))
  cat(sprintf("  %s stock-days, %s stocks, %s to %s\n",
              format(nrow(panel), big.mark = ","), format(uniqueN(panel$ticker), big.mark = ","),
              min(panel$date), max(panel$date)))

  cat("  Estimating panel regressions (two-way clustered)...\n")
  results <- run_all_tests(panel)
  fwrite(results, file.path(TABLES_DIR, paste0("panel_robustness_", model, ".csv")))

  starred <- results[q_value_bh < 0.10]
  cat("\n  PANEL RESULTS SIGNIFICANT AFTER MULTIPLE-TESTING CORRECTION (q < 0.10)\n")
  if (nrow(starred) == 0) {
    cat("  none\n")
  } else {
    print(starred[, .(type, condition, group, horizon_days,
                      bps = round(estimate_bps, 2), t = round(t_stat, 2),
                      q = round(q_value_bh, 3))], row.names = FALSE)
  }

  comparison <- compare_with_step6(results, model)
  if (!is.null(comparison)) {
    fwrite(comparison, file.path(TABLES_DIR, paste0("panel_vs_famamacbeth_", model, ".csv")))
    cat(sprintf("\n  AGREEMENT WITH STEP 6 (Fama-MacBeth): %d of %d tests reach the same conclusion (%.0f%%)\n",
                sum(comparison$same_conclusion), nrow(comparison),
                100 * mean(comparison$same_conclusion)))
    disagree <- comparison[same_conclusion == FALSE]
    if (nrow(disagree) > 0) {
      cat("  Tests where the two methods disagree (|t| >= 2 in one but not the other):\n")
      print(disagree[, .(condition, group, h = horizon_days,
                         fm_bps = round(fm_bps, 2), fm_t = round(fm_t, 2),
                         panel_bps = round(panel_bps, 2), panel_t = round(panel_t, 2))],
            row.names = FALSE)
    }
  }
}

cat("\n", strrep("=", 70), "\n", sep = "")
cat("DONE\n")
cat(strrep("=", 70), "\n")
cat("Tables:", normalizePath(TABLES_DIR), "\n")
