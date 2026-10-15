# debt-recovery-mlops

The bank assigns a more intensive recovery strategy once a customer's expected
recovery amount reaches $1000. This project measures whether that extra effort
recovers more money than it costs, using a regression discontinuity design.

It is an end-to-end MLOps rebuild of my recovery-threshold analysis: data
versioned with DVC, experiments tracked in MLflow on DagsHub, the policy served
by a FastAPI app in Docker, and every change tested by GitHub Actions before it
reaches main.
