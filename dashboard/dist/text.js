const decisionDashboard = createDecisionDashboard({
  dataFile: "text-data.json",
  defaultDataset: "sst2",
  datasetKind: "text",
  csvPrefix: "jev-text",
  showPartialMarkers: true,
  pendingChangesLabel: true,
  describeFeatures: dataset => dataset.feature_description || "Text · TF-IDF for classical models",
  statusMessage: (costs, completion) => {
    if (["blocked_no_credit", "halted", "blocked_billing", "billing_paused"].includes(costs.status)) {
      return "Jev review awaits API credits; unfinished scores are unavailable";
    }
    if (completion.complete_runs < completion.expected_runs) {
      return "text study in progress; unfinished scores are unavailable";
    }
    return "two text datasets · no model calls from this dashboard";
  },
  trainingNote: "Full prepared training uses 10,000 SST-2 labels or 4,886 TREC labels; it is a separate reference from the matched 8 or 24 labels.",
  findingsUrl: "https://github.com/statsguysam/jev-classification-benchmark/blob/main/results/text_extension/FINDINGS.md"
});
decisionDashboard.ready = decisionDashboard.boot();
