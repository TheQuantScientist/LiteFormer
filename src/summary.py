import pandas as pd
import numpy as np
from tabulate import tabulate

# Read the results CSV file
results_df = pd.read_csv('model_comparison_results.csv')

# Initialize lists to store analysis results
analysis_results = []

# Group by Stock and analyze each group
for stock, group in results_df.groupby('Stock'):
    # Sort by RMSE to find best and second-best models
    sorted_group = group.sort_values(by='RMSE', ascending=True)
    
    # Best model (lowest RMSE)
    best_model = sorted_group.iloc[0]
    best_model_name = best_model['Model']
    best_rmse = best_model['RMSE']
    best_mae = best_model['MAE']
    best_test_loss = best_model['Test Loss']
    
    # Second-best model (second-lowest RMSE)
    if len(sorted_group) > 1:
        second_best_model = sorted_group.iloc[1]
        second_best_model_name = second_best_model['Model']
        second_best_rmse = second_best_model['RMSE']
        second_best_mae = second_best_model['MAE']
        second_best_test_loss = second_best_model['Test Loss']
    else:
        # In case there's only one model (unlikely given 4 models per stock)
        second_best_model_name = 'N/A'
        second_best_rmse = np.nan
        second_best_mae = np.nan
        second_best_test_loss = np.nan
    
    # Append results for this stock
    analysis_results.append({
        'Stock': stock,
        'Best Model': best_model_name,
        'Best RMSE': best_rmse,
        'Best MAE': best_mae,
        'Best Test Loss': best_test_loss,
        'Second Best Model': second_best_model_name,
        'Second Best RMSE': second_best_rmse,
        'Second Best MAE': second_best_mae,
        'Second Best Test Loss': second_best_test_loss
    })

# Create DataFrame from analysis results
analysis_df = pd.DataFrame(analysis_results)

# Print results in a formatted table
print("\nModel Ranking Analysis (Best and Second-Best Models per Stock)")
print(tabulate(analysis_df, headers='keys', tablefmt='psql', floatfmt='.6f', showindex=False))

# Save analysis to CSV
analysis_df.to_csv('model_ranking_analysis.csv', index=False)
print("\nAnalysis saved to 'model_ranking_analysis.csv'")