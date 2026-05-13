"""
AlphaSentinel Regime Detection Engine
Hybrid regime detection using HMM + LSTM + Isolation Forest.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from src.utils import logger
from src.data_loader import create_hmm_sequences, create_lstm_sequences

# ============================================================================
# HIDDEN MARKOV MODEL
# ============================================================================

class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    
    States:
        0: Calm trending (low vol, positive drift)
        1: High volatility (elevated vol, mixed drift)
        2: Crisis (extreme vol, negative drift)
    """
    
    def __init__(self, n_states: int = config.HMM_STATES):
        """Initialize HMM regime detector."""
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """
        Fit HMM to historical features.
        
        Args:
            features: Array of shape (n_samples, n_features)
        """
        logger.info(f"Training HMM with {self.n_states} states...")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=config.HMM_COVARIANCE_TYPE,
            n_iter=config.HMM_N_ITER,
            random_state=config.HMM_RANDOM_STATE
        )
        self.model.fit(features_scaled)
        
        self.is_fitted = True
        logger.info(f"✓ HMM fitted. Converged: {self.model.monitor_.converged}")
        
        return self
    
    def predict_states(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime states.
        
        Args:
            features: Array of shape (n_samples, n_features)
            
        Returns:
            Array of predicted states of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction")
        
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)
        
        return states
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get regime probability distribution.
        
        Args:
            features: Array of shape (n_samples, n_features)
            
        Returns:
            Array of state probabilities, shape (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted before prediction")
        
        features_scaled = self.scaler.transform(features)
        _, posteriors = self.model.score_samples(features_scaled)
        
        return posteriors
    
    def save(self, path: str = config.MODELS_DIR / "hmm_model.pkl"):
        """Save fitted HMM."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'n_states': self.n_states
            }, f)
        logger.info(f"✓ HMM saved to {path}")
    
    def load(self, path: str = config.MODELS_DIR / "hmm_model.pkl"):
        """Load pre-trained HMM."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.n_states = data['n_states']
        self.is_fitted = True
        logger.info(f"✓ HMM loaded from {path}")


# ============================================================================
# LSTM NEURAL NETWORK
# ============================================================================

class LSTMRegimeDetector:
    """
    LSTM neural network for regime detection.
    
    Captures temporal patterns and non-linear dependencies in market data.
    """
    
    def __init__(self, lookback: int = config.LSTM_LOOKBACK, n_states: int = config.HMM_STATES):
        """Initialize LSTM detector."""
        self.lookback = lookback
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture."""
        inputs = keras.Input(shape=(self.lookback, 5))  # 5 features
        
        # LSTM layers with dropout
        x = layers.LSTM(config.LSTM_UNITS, return_sequences=True)(inputs)
        x = layers.Dropout(config.LSTM_DROPOUT)(x)
        x = layers.LSTM(config.LSTM_UNITS, return_sequences=False)(x)
        x = layers.Dropout(config.LSTM_DROPOUT)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(config.LSTM_DROPOUT)(x)
        
        # Output layer: regime classification
        outputs = layers.Dense(self.n_states, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✓ LSTM model built")
    
    def fit(self, X: np.ndarray, y_regimes: np.ndarray, epochs: int = config.LSTM_EPOCHS):
        """
        Train LSTM on regime-labeled sequences.
        
        Args:
            X: Sequences of shape (n_samples, lookback, n_features)
            y_regimes: Regime labels of shape (n_samples, n_states) one-hot encoded
            epochs: Number of training epochs
        """
        logger.info(f"Training LSTM for {epochs} epochs...")
        
        # Validate input shapes
        if len(X) < config.LSTM_BATCH_SIZE:
            logger.warning(f"Insufficient samples ({len(X)}) for batch size {config.LSTM_BATCH_SIZE}. Using {max(2, len(X)//2)}")
            batch_size = max(2, len(X) // 2)
        else:
            batch_size = config.LSTM_BATCH_SIZE
        
        try:
            history = self.model.fit(
                X, y_regimes,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=config.LSTM_VALIDATION_SPLIT,
                verbose=0  # Changed to 0 for cleaner logging
            )
            
            self.is_fitted = True
            logger.info("✓ LSTM training complete")
            
            return history
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            self.is_fitted = False
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get regime probability distribution.
        
        Args:
            X: Sequences of shape (n_samples, lookback, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_states)
        """
        if not self.is_fitted:
            raise ValueError("LSTM must be fitted before prediction")
        
        return self.model.predict(X, verbose=0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely regime."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def save(self, path: str = config.LSTM_MODEL_PATH):
        """Save LSTM model."""
        self.model.save(path)
        logger.info(f"✓ LSTM saved to {path}")
    
    def load(self, path: str = config.LSTM_MODEL_PATH):
        """Load pre-trained LSTM."""
        self.model = keras.models.load_model(path)
        self.is_fitted = True
        logger.info(f"✓ LSTM loaded from {path}")


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

class AnomalyDetector:
    """
    Isolation Forest for detecting market anomalies.
    
    Flags unusual market conditions that violate model assumptions.
    """
    
    def __init__(self, contamination: float = config.ISOLATION_FOREST_CONTAMINATION):
        """Initialize anomaly detector."""
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=config.ISOLATION_FOREST_RANDOM_STATE
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit Isolation Forest.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
        """
        logger.info(f"Training Isolation Forest (contamination={self.contamination})...")
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("✓ Isolation Forest fitted")
        
        return self
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in data.
        
        Args:
            X: Feature array
            
        Returns:
            Array: 1 for normal, -1 for anomaly
        """
        if not self.is_fitted:
            raise ValueError("Anomaly detector must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Anomaly detector must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


# ============================================================================
# ENSEMBLE REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """
    Ensemble regime detection combining HMM, LSTM, and Anomaly Detection.
    
    This is the main entry point for regime detection.
    """
    
    def __init__(self):
        """Initialize ensemble detector."""
        self.hmm = HMMRegimeDetector(n_states=config.HMM_STATES)
        self.lstm = LSTMRegimeDetector(
            lookback=config.LSTM_LOOKBACK,
            n_states=config.HMM_STATES
        )
        self.anomaly_detector = AnomalyDetector(
            contamination=config.ISOLATION_FOREST_CONTAMINATION
        )
        self.is_fitted = False
        self.regime_names = ["Calm", "Volatile", "Crisis"]
    
    def fit(self, features: pd.DataFrame):
        """
        Train all three components.
        
        Args:
            features: DataFrame with columns [returns, volatility, skewness, kurtosis, momentum]
        """
        logger.info("Training Regime Detection Ensemble...")
        
        # Extract numpy arrays
        feature_cols = ['returns', 'volatility', 'skewness', 'kurtosis', 'momentum']
        available_cols = [col for col in feature_cols if col in features.columns]
        
        if len(available_cols) < 3:
            logger.error(f"Insufficient features. Found: {available_cols}")
            raise ValueError(f"Need at least 3 features, got {len(available_cols)}")
        
        X = features[available_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Handle NaN/inf
        
        if len(X) < config.HMM_STATES + 5:
            logger.error(f"Insufficient training samples: {len(X)} < {config.HMM_STATES + 5}")
            raise ValueError(f"Need at least {config.HMM_STATES + 5} samples")
        
        # 1. Fit HMM
        try:
            self.hmm.fit(X)
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            raise
        
        # 2. Fit LSTM
        try:
            X_lstm, y_regimes = create_lstm_sequences(features, config.LSTM_LOOKBACK)
            
            if len(X_lstm) < config.LSTM_BATCH_SIZE:
                logger.warning(f"LSTM sequences ({len(X_lstm)}) less than batch size ({config.LSTM_BATCH_SIZE})")
                logger.warning("Skipping LSTM training - insufficient data")
            else:
                # Get HMM regime labels for training
                hmm_states = self.hmm.predict_states(X)
                
                # Create one-hot encoded labels
                y_one_hot = np.zeros((len(X_lstm), config.HMM_STATES))
                for i in range(len(X_lstm)):
                    if config.LSTM_LOOKBACK + i < len(hmm_states):
                        regime_idx = int(hmm_states[config.LSTM_LOOKBACK + i])
                        y_one_hot[i, regime_idx] = 1
                
                self.lstm.fit(X_lstm, y_one_hot, epochs=config.LSTM_EPOCHS)
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}. Continuing with HMM only...")
        
        # 3. Fit Anomaly Detector
        try:
            hmm_proba = self.hmm.predict_proba(X)
            hmm_uncertainty = 1.0 - hmm_proba.max(axis=1)
            self.anomaly_detector.fit(hmm_uncertainty.reshape(-1, 1))
        except Exception as e:
            logger.warning(f"Anomaly detector fitting failed: {e}")
        
        self.is_fitted = True
        logger.info("✓ Ensemble fitted successfully")
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Predict current regime with confidence.
        
        Returns:
            Dict with keys:
            - regime: Predicted regime (0, 1, or 2)
            - regime_name: Human-readable name
            - confidence: Confidence score (0-1)
            - probabilities: Full probability distribution
            - is_anomaly: Boolean flag for anomaly
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        feature_cols = ['returns', 'volatility', 'skewness', 'kurtosis', 'momentum']
        available_cols = [col for col in feature_cols if col in features.columns]
        
        if len(available_cols) < 3:
            logger.warning(f"Insufficient features for prediction: {available_cols}")
            # Return neutral prediction
            return {
                'regime': 1,
                'regime_name': self.regime_names[1],
                'confidence': 0.33,
                'probabilities': [0.33, 0.33, 0.33],
                'is_anomaly': False,
                'hmm_regime': 1,
                'lstm_regime': 1,
            }
        
        try:
            X = features[available_cols].values[-config.LSTM_LOOKBACK:]
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(X) < 10:  # Not enough data
                return {
                    'regime': 1,
                    'regime_name': self.regime_names[1],
                    'confidence': 0.33,
                    'probabilities': [0.33, 0.33, 0.33],
                    'is_anomaly': False,
                    'hmm_regime': 1,
                    'lstm_regime': 1,
                }
            
            # Get latest features for anomaly check
            X_latest = X[-1:, :]
            
            # 1. HMM prediction
            hmm_proba = self.hmm.predict_proba(X_latest)[0]
            hmm_regime = np.argmax(hmm_proba)
            hmm_confidence = hmm_proba[hmm_regime]
            
            # 2. LSTM prediction (if fitted)
            lstm_proba = np.array([1/3, 1/3, 1/3])  # Default uniform
            lstm_regime = 1
            
            if self.lstm.is_fitted:
                try:
                    X_lstm = X.reshape(1, config.LSTM_LOOKBACK, -1)
                    lstm_proba = self.lstm.predict_proba(X_lstm)[0]
                    lstm_regime = np.argmax(lstm_proba)
                except:
                    logger.debug("LSTM prediction failed, using HMM only")
            
            # 3. Anomaly detection
            is_anomaly = False
            try:
                hmm_uncertainty = 1.0 - hmm_proba.max()
                is_anomaly = (self.anomaly_detector.detect_anomalies(
                    hmm_uncertainty.reshape(1, -1)
                )[0] == -1)
            except:
                logger.debug("Anomaly detection failed")
            
            # Ensemble consensus (weighted average)
            w_hmm = 0.4
            w_lstm = 0.5
            w_anomaly = 0.1
            
            ensemble_proba = (
                w_hmm * hmm_proba +
                w_lstm * lstm_proba
            )
            
            if is_anomaly:
                ensemble_proba = ensemble_proba * (1 - config.ANOMALY_ALERT_FLAG)
            
            ensemble_regime = np.argmax(ensemble_proba)
            ensemble_confidence = ensemble_proba[ensemble_regime]
            
            return {
                'regime': int(ensemble_regime),
                'regime_name': self.regime_names[ensemble_regime],
                'confidence': float(ensemble_confidence),
                'probabilities': ensemble_proba.tolist(),
                'is_anomaly': bool(is_anomaly),
                'hmm_regime': int(hmm_regime),
                'lstm_regime': int(lstm_regime),
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return neutral prediction on error
            return {
                'regime': 1,
                'regime_name': self.regime_names[1],
                'confidence': 0.33,
                'probabilities': [0.33, 0.33, 0.33],
                'is_anomaly': False,
                'hmm_regime': 1,
                'lstm_regime': 1,
            }
    
    def predict_timeseries(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes for entire timeseries.
        
        Args:
            features: DataFrame with all historical features
            
        Returns:
            DataFrame with predicted regimes and probabilities
        """
        results = []
        
        for i in range(config.LSTM_LOOKBACK, len(features)):
            window = features.iloc[i - config.LSTM_LOOKBACK:i]
            pred = self.predict(window)
            pred['date'] = features.index[i]
            results.append(pred)
        
        df_results = pd.DataFrame(results)
        df_results.set_index('date', inplace=True)
        
        logger.info(f"✓ Predicted regimes for {len(df_results)} dates")
        
        return df_results
    
    def save(self):
        """Save all ensemble components."""
        self.hmm.save()
        self.lstm.save()
        logger.info("✓ Regime detector ensemble saved")
    
    def load(self):
        """Load all ensemble components."""
        if config.USE_PRETRAINED_HMM:
            self.hmm.load()
        if config.USE_PRETRAINED_LSTM:
            self.lstm.load()
        self.is_fitted = True
        logger.info("✓ Regime detector ensemble loaded")
