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

