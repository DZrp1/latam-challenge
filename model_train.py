import pandas as pd
from joblib import dump
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from challenge.model import DelayModel

def main():
    """
    Main function to train the model and save the artifact.
    """
    Path("models").mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path("data") / "data.csv"
    data = pd.read_csv(data_path, dtype={'Vlo-I': str, 'Vlo-O': str})

    # Preprocess data
    model = DelayModel()
    features, target = model.preprocess(data=data, target_column="delay")

    # Train model
    model.fit(features=features, target=target)

    # Save model artifact
    model_path = Path("models") / "trained_model.joblib"
    dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    main()