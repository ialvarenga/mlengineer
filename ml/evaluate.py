"""
Model Evaluation Module

This module provides comprehensive model evaluation including:
- Performance metrics
- Overfitting/Underfitting analysis
- Feature importance analysis
- Suggestions for improvement
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    HOUSING_DATA_PATH,
    ZIPCODE_DATA_PATH,
    MODEL_METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE
)
from ml.model import HousingPriceModel


class ModelEvaluator:
    """
    Comprehensive model evaluation for the housing price predictor.
    """
    
    def __init__(self, model: HousingPriceModel):
        self.model = model
        self.evaluation_results = {}
    
    def evaluate_all(self, X_train, y_train, X_test, y_test) -> dict:
        """
        Run all evaluation metrics.
        """
        results = {}
        
        # Basic metrics
        results['basic_metrics'] = self._calculate_basic_metrics(X_train, y_train, X_test, y_test)
        
        # Overfitting analysis
        results['overfitting_analysis'] = self._analyze_overfitting(results['basic_metrics'])
        
        # Feature importance
        results['feature_importance'] = self.model.get_feature_importance()
        
        # Improvement suggestions
        results['improvement_suggestions'] = self._generate_suggestions(results)
        
        self.evaluation_results = results
        return results
    
    def _calculate_basic_metrics(self, X_train, y_train, X_test, y_test) -> dict:
        """Calculate basic performance metrics."""
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        return {
            'train': {
                'r2_score': float(r2_score(y_train, train_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
                'mae': float(mean_absolute_error(y_train, train_pred)),
                'mape': float(np.mean(np.abs((y_train - train_pred) / y_train)) * 100)
            },
            'test': {
                'r2_score': float(r2_score(y_test, test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
                'mae': float(mean_absolute_error(y_test, test_pred)),
                'mape': float(np.mean(np.abs((y_test - test_pred) / y_test)) * 100)
            }
        }
    
    def _analyze_overfitting(self, metrics: dict) -> dict:
        """Analyze if the model is overfitting or underfitting."""
        train_r2 = metrics['train']['r2_score']
        test_r2 = metrics['test']['r2_score']
        r2_gap = train_r2 - test_r2
        
        analysis = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'r2_gap': r2_gap,
            'status': 'good_fit'
        }
        
        if r2_gap > 0.15:
            analysis['status'] = 'overfitting'
            analysis['description'] = (
                "The model shows signs of OVERFITTING. "
                f"Train R² ({train_r2:.3f}) is significantly higher than Test R² ({test_r2:.3f}). "
                "The model may be memorizing training data rather than learning generalizable patterns."
            )
        elif test_r2 < 0.5:
            analysis['status'] = 'underfitting'
            analysis['description'] = (
                "The model shows signs of UNDERFITTING. "
                f"Test R² ({test_r2:.3f}) is low, indicating the model is too simple "
                "to capture the underlying patterns in the data."
            )
        elif r2_gap > 0.05:
            analysis['status'] = 'slight_overfitting'
            analysis['description'] = (
                "The model shows SLIGHT OVERFITTING. "
                f"Train R² ({train_r2:.3f}) is somewhat higher than Test R² ({test_r2:.3f}). "
                "Consider regularization or reducing model complexity."
            )
        else:
            analysis['description'] = (
                "The model appears to be WELL-FITTED. "
                f"Train R² ({train_r2:.3f}) and Test R² ({test_r2:.3f}) are well-balanced, "
                "indicating good generalization ability."
            )
        
        return analysis
    
    def _generate_suggestions(self, results: dict) -> list:
        """Generate improvement suggestions based on evaluation results."""
        suggestions = []
        
        overfitting = results['overfitting_analysis']
        metrics = results['basic_metrics']
        
        # Based on overfitting status
        if overfitting['status'] == 'overfitting':
            suggestions.extend([
                {
                    'category': 'Model Complexity',
                    'suggestion': 'Reduce model complexity by decreasing max_depth or n_estimators',
                    'priority': 'high'
                },
                {
                    'category': 'Regularization',
                    'suggestion': 'Add regularization (increase min_samples_split, min_samples_leaf)',
                    'priority': 'high'
                },
                {
                    'category': 'Data',
                    'suggestion': 'Collect more training data to reduce overfitting',
                    'priority': 'medium'
                }
            ])
        elif overfitting['status'] == 'underfitting':
            suggestions.extend([
                {
                    'category': 'Model Complexity',
                    'suggestion': 'Increase model complexity (more trees, deeper trees)',
                    'priority': 'high'
                },
                {
                    'category': 'Features',
                    'suggestion': 'Engineer more features or add polynomial features',
                    'priority': 'high'
                },
                {
                    'category': 'Model Type',
                    'suggestion': 'Try a more complex model like Gradient Boosting or XGBoost',
                    'priority': 'medium'
                }
            ])
        
        # General suggestions
        suggestions.extend([
            {
                'category': 'Feature Engineering',
                'suggestion': 'Add more demographic features (crime rate, school ratings)',
                'priority': 'medium'
            },
            {
                'category': 'Feature Engineering',
                'suggestion': 'Create interaction features (e.g., bedrooms * bathrooms)',
                'priority': 'low'
            },
            {
                'category': 'Hyperparameter Tuning',
                'suggestion': 'Use GridSearchCV or RandomizedSearchCV for hyperparameter optimization',
                'priority': 'medium'
            },
            {
                'category': 'Ensemble Methods',
                'suggestion': 'Try stacking or blending multiple models',
                'priority': 'low'
            },
            {
                'category': 'Data Quality',
                'suggestion': 'Handle outliers in price data more carefully',
                'priority': 'medium'
            }
        ])
        
        return suggestions
    
    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.evaluation_results:
            logger.warning("No evaluation results. Run evaluate_all() first.")
            return
        
        results = self.evaluation_results
        
        print("\n" + "=" * 70)
        print("MODEL EVALUATION REPORT")
        print("=" * 70)
        
        # Basic Metrics
        print("\n📊 PERFORMANCE METRICS")
        print("-" * 40)
        metrics = results['basic_metrics']
        print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        print(f"{'R² Score':<20} {metrics['train']['r2_score']:.4f}         {metrics['test']['r2_score']:.4f}")
        print(f"{'RMSE':<20} ${metrics['train']['rmse']:,.0f}       ${metrics['test']['rmse']:,.0f}")
        print(f"{'MAE':<20} ${metrics['train']['mae']:,.0f}       ${metrics['test']['mae']:,.0f}")
        print(f"{'MAPE':<20} {metrics['train']['mape']:.2f}%         {metrics['test']['mape']:.2f}%")
        
        # Overfitting Analysis
        print("\n🔍 OVERFITTING/UNDERFITTING ANALYSIS")
        print("-" * 40)
        analysis = results['overfitting_analysis']
        status_emoji = {
            'good_fit': '✅',
            'slight_overfitting': '⚠️',
            'overfitting': '❌',
            'underfitting': '❌'
        }
        print(f"Status: {status_emoji.get(analysis['status'], '❓')} {analysis['status'].upper()}")
        print(f"\n{analysis['description']}")
        
        # Feature Importance
        print("\n📈 TOP FEATURE IMPORTANCE")
        print("-" * 40)
        importance = results['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_features[:5], 1):
            bar = "█" * int(score * 50)
            print(f"{i}. {feature:<25} {bar} {score:.4f}")
        
        # Suggestions
        print("\n💡 IMPROVEMENT SUGGESTIONS")
        print("-" * 40)
        for i, suggestion in enumerate(results['improvement_suggestions'][:5], 1):
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            print(f"{i}. [{priority_emoji[suggestion['priority']]}] {suggestion['category']}")
            print(f"   → {suggestion['suggestion']}")
        
        print("\n" + "=" * 70)


def main():
    """Run model evaluation."""
    from ml.train import load_and_prepare_data
    
    logger.info("Loading data...")
    X, y = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    logger.info("Loading model...")
    model = HousingPriceModel.load()
    
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate_all(X_train, y_train, X_test, y_test)
    
    evaluator.print_report()
    
    # Save results
    with open(MODEL_METRICS_PATH, 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    logger.info(f"Evaluation results saved to {MODEL_METRICS_PATH}")


if __name__ == "__main__":
    main()
