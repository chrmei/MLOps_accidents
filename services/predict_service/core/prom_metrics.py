from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

registry = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

model_accuracy_score = Gauge(
    "model_accuracy_score", "Current accuracy of the model", registry=registry
)

model_precision_score = Gauge(
    "model_precision_score", "Precision score of the model", registry=registry
)

model_recall_score = Gauge(
    "model_recall_score", "Recall score of the model", registry=registry
)

model_f1_score = Gauge("model_f1_score", "F1 score of the model", registry=registry)

evidently_col_drift_share = Gauge(
    "column_drift_share", "Column Drift Share", registry=registry
)
