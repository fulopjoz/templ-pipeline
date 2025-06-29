# Updated scoring thresholds for shape/color/combo scores
# These values are appropriate for the scoring method used in TEMPL
# Shape/color scores typically range from 0.0 to 0.5 for good matches

# Scoring thresholds
# Note: These are for shape/color/combo scores, not Tanimoto similarity
SCORE_EXCELLENT = 0.35  # Excellent match (top 10%)
SCORE_GOOD = 0.25      # Good match (top 25%)
SCORE_FAIR = 0.15      # Fair match (acceptable)
SCORE_POOR = 0.0       # Poor match (below threshold)
