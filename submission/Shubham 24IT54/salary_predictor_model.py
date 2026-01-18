import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

DATA_PATH = '../../dataset/salary_data.csv'
OUTPUT_DIR = './'

def load_data():
    print("Loading data...")
    salary_shenanigans = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {salary_shenanigans.shape}")
    print(f"Columns: {salary_shenanigans.columns.tolist()}")
    print(f"\nFirst few rows:\n{salary_shenanigans.head()}")
    print(f"\nDataset statistics:\n{salary_shenanigans.describe()}")
    return salary_shenanigans

def prepare_data(salary_shenanigans):
    X = salary_shenanigans[['YearsExperience']].values
    y = salary_shenanigans['Salary($)'].values
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def train_model(X_train, y_train):
    print("\nTraining Linear Regression model...")
    money_wizard = LinearRegression()
    money_wizard.fit(X_train, y_train)
    
    print(f"Model coefficients:")
    print(f"  Intercept: {money_wizard.intercept_:.2f}")
    print(f"  Slope: {money_wizard.coef_[0]:.2f}")
    
    return money_wizard

def evaluate_model(money_wizard, X_test, y_test):
    print("\nEvaluating model...")
    
    crystal_ball = money_wizard.predict(X_test)
    
    r2 = r2_score(y_test, crystal_ball)
    rmse = np.sqrt(mean_squared_error(y_test, crystal_ball))
    
    print(f"\nModel Performance Metrics:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.2f}")
    
    print(f"\nRequirement Checks:")
    print(f"  R² Score >= 0.90: {'PASS' if r2 >= 0.90 else 'FAIL'} (Current: {r2:.4f})")
    print(f"  RMSE < 7000: {'PASS' if rmse < 7000 else 'FAIL'} (Current: {rmse:.2f})")
    
    return crystal_ball, r2, rmse

def create_visualization(y_test, crystal_ball, output_path):
    print("\nCreating visualization...")
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, crystal_ball, alpha=0.6, color='blue', label='Predictions')
    
    min_val = min(min(y_test), min(crystal_ball))
    max_val = max(max(y_test), max(crystal_ball))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Salary ($)', fontsize=12)
    plt.ylabel('Predicted Salary ($)', fontsize=12)
    plt.title('Actual vs Predicted Salary', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    r2 = r2_score(y_test, crystal_ball)
    plt.text(0.05, 0.95, f'R² Score: {r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()

def save_metrics(r2, rmse, output_path):
    print("\nSaving metrics...")
    
    with open(output_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"R² Score: {r2:.6f}\n")
        f.write(f"RMSE: {rmse:.2f}\n\n")
        
        f.write("-" * 50 + "\n")
        f.write("REQUIREMENT CHECKS\n")
        f.write("-" * 50 + "\n\n")
        
        f.write(f"Test 1 - R² Score >= 0.90: {'PASS' if r2 >= 0.90 else 'FAIL'}\n")
        f.write(f"  Current R² Score: {r2:.6f}\n")
        f.write(f"  Required: >= 0.90\n\n")
        
        f.write(f"Test 2 - Actual vs Predicted Plot: PASS\n")
        f.write(f"  Plot saved as: actual_vs_pred.png\n\n")
        
        f.write(f"Test 3 - RMSE < 7000: {'PASS' if rmse < 7000 else 'FAIL'}\n")
        f.write(f"  Current RMSE: {rmse:.2f}\n")
        f.write(f"  Required: < 7000\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write("Model Type: Linear Regression\n")
        f.write("Feature: YearsExperience\n")
        f.write("Target: Salary($)\n")
    
    print(f"Metrics saved to: {output_path}")

def main():
    print("=" * 60)
    print("SALARY PREDICTOR MODEL - TRAINING PIPELINE")
    print("=" * 60)
    
    salary_shenanigans = load_data()
    
    X, y = prepare_data(salary_shenanigans)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    money_wizard = train_model(X_train, y_train)
    
    crystal_ball, r2, rmse = evaluate_model(money_wizard, X_test, y_test)
    
    create_visualization(y_test, crystal_ball, os.path.join(OUTPUT_DIR, 'actual_vs_pred.png'))
    
    save_metrics(r2, rmse, os.path.join(OUTPUT_DIR, 'model_performance.txt'))
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. actual_vs_pred.png - Visualization plot")
    print(f"  2. model_performance.txt - Performance metrics")
    print("\n")

if __name__ == "__main__":
    main()
