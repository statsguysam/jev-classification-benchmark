const decisionDashboard = createDecisionDashboard({
  dataFile: "numeric-data.json",
  defaultDataset: "breast_cancer",
  datasetKind: "numerical",
  csvPrefix: "jev-numeric",
  showPartialMarkers: false,
  pendingChangesLabel: false,
  describeFeatures: dataset => `${dataset.n_features} numeric features`,
  statusMessage: costs => costs.status === "halted" ? "Jev review paused by billing errors; unfinished scores are unavailable" : "two numerical datasets · no model calls from this dashboard",
  trainingNote: "The total also excludes the later text extension.",
  findingsUrl: "https://github.com/statsguysam/jev-classification-benchmark/blob/main/results/numeric_expansion/FINDINGS.md"
});
decisionDashboard.ready = decisionDashboard.boot();
