import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, Union, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DelayModel:

    def __init__(
        self
    ):
        """Initialize DelayModel with default configuration."""
        logger.info("Initializing DelayModel")
        
        self._model = xgb.XGBClassifier(
            random_state=42, 
            eval_metric='logloss'
        )
        self.top_10_features = [
            "OPERA_Latin American Wings", "MES_7", "MES_10", 
            "OPERA_Grupo LATAM", "MES_12", "TIPOVUELO_I", 
            "MES_4", "MES_11", "OPERA_Sky Airline", "OPERA_Copa Air"
        ]
        self._is_fitted = False
        self.delay_threshold = 15
        self.high_season_ranges = [
            ('15-Dec', '31-Dec'), ('1-Jan', '3-Mar'),
            ('15-Jul', '31-Jul'), ('11-Sep', '30-Sep')
        ]
        self.time_periods = {
            'mañana': ('05:00', '11:59'),
            'tarde': ('12:00', '18:59')
        }
        self.airlines = None
        self.types = None
        self.months = None
    
    def _get_period_day(self, date: str) -> str:
        try:
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        except ValueError as e:
            raise ValueError(f"Invalid date format for '{date}'. Expected YYYY-MM-DD HH:MM:SS.") from e

        morning_range = self.time_periods['mañana']
        afternoon_range = self.time_periods['tarde']

        morning_min = datetime.strptime(morning_range[0], '%H:%M').time()
        morning_max = datetime.strptime(morning_range[1], '%H:%M').time()
        afternoon_min = datetime.strptime(afternoon_range[0], '%H:%M').time()
        afternoon_max = datetime.strptime(afternoon_range[1], '%H:%M').time()

        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        else:
            return 'noche'

    def _is_high_season(self, fecha: str) -> int:
        try:
            fecha_dt = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            fecha_año = fecha_dt.year
        except ValueError as e:
            raise ValueError(f"Invalid date format for '{fecha}'. Expected YYYY-MM-DD HH:MM:SS.") from e

        for start_date_str, end_date_str in self.high_season_ranges:
            range_min = datetime.strptime(start_date_str, '%d-%b').replace(year=fecha_año)
            range_max = datetime.strptime(end_date_str, '%d-%b').replace(year=fecha_año)
            if range_min <= fecha_dt <= range_max:
                return 1
        return 0

    @staticmethod
    def _get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        This method separates the creation of the target variable (for training)
        from the creation of model features (for both training and prediction),
        ensuring consistency and preventing data leakage.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        logger.info(f"Preprocessing data with {len(data)} rows.")

        base_feature_cols = ['OPERA', 'TIPOVUELO', 'MES']

        # Training-specific logic: Target variable creation
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data['min_diff'] > self.delay_threshold, 1, 0)
            
            self.airlines = data['OPERA'].unique().tolist()
            self.types = data['TIPOVUELO'].unique().tolist()
            self.months = data['MES'].unique().tolist()

        # Select only the features available in production to prevent data leakage.
        features = pd.get_dummies(data[base_feature_cols], columns=base_feature_cols)

        # Align columns with the predefined top features learned during exploration.
        for col in self.top_10_features:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.top_10_features]

        if target_column:
            target = data[[target_column]]
            return features, target
        
        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        logger.info(f"Fitting model with {len(features)} features and {len(target)} target values.")
        target_series = target.iloc[:, 0]
        n_y0 = (target_series == 0).sum()
        n_y1 = (target_series == 1).sum()
        scale_pos_weight = n_y0 / n_y1 if n_y1 > 0 else 1
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        self._model.set_params(scale_pos_weight=scale_pos_weight)
        self._model.fit(features, target_series)
        self._is_fitted = True
        logger.info("Model fitting completed.")

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        predictions = self._model.predict(features)
        return predictions.tolist()