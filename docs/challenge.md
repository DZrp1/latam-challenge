# Part I: Model Transcription and Operationalization

## Model Selection

I identified that the dataset showed a severe class imbalance, where baseline models had around 98% accuracy but only \~2% recall for class 1 (delayed flights), rendering them useless for the business objective. I selected and implemented XGBoost with `scale_pos_weight` for automatic class balancing, as it improved the recall for the positive class up to 69% while maintaining acceptable overall metrics.

I chose to use the top 10 most important features since there was no performance degradation, which optimizes latency and reduces overfitting.

**XGBoost vs Logistic Regression**: While both models demonstrated similar performance on the dataset, I chose XGBoost due to:

* Greater ability to capture non-linear relationships
* Industry standard for tabular data
* Better scalability and robustness with new data

## Technical Implementation

* Designed the `DelayModel` class to encapsulate preprocessing, training, and prediction logic
* Implemented logic to ensure post-flight variables (`Fecha-O`) are used only for target creation and never as features
* Developed a pipeline to ensure alignment with the selected 10 features
* Separated concerns into `model.py` (definition) and `model_train.py` (execution)
* Implemented model serialization for deployment
* Developed robust handling for missing data and unseen categories

The resulting model prioritizes recall over precision to minimize false negatives (undetected delayed flights), aligning with the business requirements.


# Part II: API with FastAPI

* I implemented model loading during startup (`@app.on_event("startup")`) instead of per request to minimize prediction latency.
* I used Pydantic models (`Flight`, `FlightPayload`) for automatic type and structure validation, preventing runtime errors.
* To keep the `predict()` method signature immutable, I accessed the request body via `inspect.currentframe()`. Although this isn't a standard practice for production, I used it to comply with the constraint of not altering the `post_predict()` method signature.


# Part III: Deployment on GCP

* I implemented Artifact Registry + Cloud Build instead of local builds to lay a strong foundation for automated CI/CD.
* Deployed to `southamerica-west1` (Santiago) to minimize latency for end users and comply with data sovereignty requirements.

**Resources**:
* Artifact Registry: `latam-challenge-repo`
* Cloud Run Service: `flight-delay-api`


# Part IV: CI/CD Automation with GitHub Actions

**CI Pipeline**: I set up automated testing on the `develop` branch for fast feedback during feature development and to prevent regressions before merging.

**CD Pipeline**: I implemented automatic deployment from the `main` branch with:

* Secure authentication via Service Account and GitHub Secrets
* Automated build and push to Artifact Registry
* Zero-downtime deployment to Cloud Run