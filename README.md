# Sabre Fencing Win & Scoreline Prediction (2023/2024 Season)

Predict match outcome and scoreline by assigning Elo ratings to fencers and using an ensemble of Random Forest and XGBoost models to predict win probabilities and expected scorelines for head-to-head matches.

The project uses data from the 2023/2024 season, including:

- **Elimination rounds**: Scores and winners of head-to-head matches.
- **Pool results**: Performance in preliminary rounds.
- **Final rankings**: Official FIE rankings.
- **Athlete data**: Personal information like age, nationality, and points.

The goal is to provide:

- Head-to-head win probability
- Predicted scoreline
- Insights into match competitiveness

## 2. Data Structure

The project uses four main datasets for each tournament:

### **2.1 Elimination Rounds**
| Column            | Description                                   |
|------------------|-----------------------------------------------|
| Round             | Round number in elimination (e.g., 32, 16)   |
| Fencer A Name     | Name of Fencer A                              |
| Fencer A Country  | Country of Fencer A                           |
| Fencer A Score    | Score of Fencer A                             |
| Fencer B Name     | Name of Fencer B                              |
| Fencer B Country  | Country of Fencer B                           |
| Fencer B Score    | Score of Fencer B                             |
| Winner            | Name of the winning fencer                    |

### **2.2 Pool Results**
| Column | Description |
|--------|------------|
| Name   | Fencer's name |
| V      | Number of victories in pools |
| M      | Number of matches fenced in pools |
| Rk.    | Pool ranking |
| TD     | Touches scored |
| TR     | Touches received |
| Ind.   | Indicator or points |
| Diff.  | Touch difference (TD - TR) |
| Q      | Qualified for elimination (Yes/No) |

### **2.3 Final Rankings**
| Column      | Description |
|-------------|------------|
| Name        | Fencer's name |
| Rank        | Final rank in the competition |
| Points      | Points accumulated in the season |
| Age         | Athlete's age |
| Nationality | Athlete's nationality |

### **2.4 Athletes**
| Column      | Description |
|-------------|------------|
| Name        | Athlete's name |
| Rank        | Current world rank |
| Points      | Total FIE points |
| Age         | Athlete's age |
| Nationality | Athlete's nationality |

## 3. Data Cleaning & Feature Engineering

This section handles preprocessing the raw datasets and creating features for ML models.

### **3.1 Cleaning Column Names**
- Removes extra spaces, BOM characters, and zero-width spaces.
- Ensures uniform column names across datasets.

### **3.2 Elimination Data Processing**
- Converts fencer names and winner to title-case strings.
- Drops rows with missing critical values (`Fencer A/B Name`, `Winner`, `Fencer A/B Score`).
- Converts `Round` to integer (extracts numeric part).
- Converts `Fencer A/B Score` to float.

### **3.3 Pool Data Processing**
- Converts columns like `V`, `M`, `TD`, `TR`, `Diff.` to integers.
- Converts `Ind.` to float and `Q` to string.
- Cleans `Name` to title-case.

### **3.4 Ranking and Athletes Data**
- Converts `Name` to title-case.
- Converts `Rank`, `Age` to int, `Points` to float, `Nationality` to string.

### **3.5 Top 16 and Preliminary Athletes**
- `top_16_athletes`: Top 16 athletes based on world rank.
- `pool_qualifiers`: Top 16 athletes from pool results.
- `main_round_athletes = top_16_athletes + pool_qualifiers`
- `preliminary_athletes`: All other athletes not in main round.

### **3.6 Feature Engineering**
- **Pool Performance:** Win rate (`V/M`) and average rank.
- **Elimination Performance:** Count of wins, total matches, elimination win rate.
- **Elo Ratings:** Initialized from ranking, updated after each elimination match.
- **Final Features per Fencer:**
  - `Win_Rate` (from pool)
  - `Elim_Wins`
  - `Elim_Matches`
 
## 4. Head-to-Head Feature Engineering

This section creates features for each match between two fencers based on their individual statistics.

### **4.1 Feature Calculation for Each Match**
For each elimination match:
- `Elo_Diff` = Fencer A Elo - Fencer B Elo
- `Win_Rate_Diff` = Fencer A Pool Win Rate - Fencer B Pool Win Rate
- `Elim_Win_Rate_Diff` = Fencer A Elim Win Rate - Fencer B Elim Win Rate
- `Rank_Diff` = Fencer A Rank - Fencer B Rank
- `Elim_Wins_Diff` = Fencer A Elim Wins - Fencer B Elim Wins
- `IsPreliminary_A` = 1 if Fencer A is a preliminary fencer, else 0
- `IsPreliminary_B` = 1 if Fencer B is a preliminary fencer, else 0
- `Result` = 1 if Fencer A wins, 0 if Fencer B wins
- `Score_Diff` = Fencer A Score - Fencer B Score

### **4.2 Output**
- `head_to_head_df`: DataFrame where each row corresponds to a match and contains the features listed above.
- Ready to be used as input for ML models.

  - `Elim_Win_Rate`
  - `Elo`
  - `Rank`
  - `IsPreliminary` (1 if in preliminary round, 0 otherwise)


## 5. Machine Learning Model Training

This section trains machine learning models to predict match outcomes (win probability) and score differences.

### **5.1 Dataset Preparation**
- Features (`X`):  
  - `Elo_Diff`  
  - `Win_Rate_Diff`  
  - `Elim_Win_Rate_Diff`  
  - `Rank_Diff`  
  - `Elim_Wins_Diff`  
  - `IsPreliminary_A`  
  - `IsPreliminary_B`  
- Target variables:  
  - `y_win` = Match Result (1 if Fencer A wins, 0 if Fencer B wins)  
  - `y_score_diff` = Score Difference

- Train-test split: 80% training, 20% testing.
- Standardize features using `StandardScaler`.

### **5.2 Models**
1. **RandomForestRegressor**
   - For predicting win probability (`y_win`)
   - For predicting score difference (`y_score_diff`)
   - Hyperparameter tuning using `GridSearchCV`:
     - `n_estimators`: [100, 200, 300]  
     - `max_depth`: [10, 20, 30, None]  
     - `min_samples_split`: [2, 5, 10]  

2. **XGBoost Regressor**
   - For predicting win probability (`y_win`)
   - For predicting score difference (`y_score_diff`)
   - Hyperparameter tuning using `GridSearchCV`:
     - `n_estimators`: [100, 200, 300]  
     - `max_depth`: [3, 5, 7]  
     - `learning_rate`: [0.01, 0.1, 0.3]  

### **5.3 Evaluation**
- Mean Squared Error (MSE) calculated for each model.
- Best models from grid search are saved for predictions.

## 6. Head-to-Head Prediction Function

This section defines the function to predict the outcome and scoreline of a match between two fencers.

### **6.1 Function: `predict_head_to_head`**
- **Inputs**:
  - `fencer_a_name`, `fencer_b_name`: Names of the two fencers.
  - `combined_features`: DataFrame with fencer features.
  - `fencer_elos`: Dictionary of Elo ratings for each fencer.
  - `rf_win_best_model`, `xgb_win_best_model`: Trained ML models for win probability.
  - `rf_score_best_model`, `xgb_score_best_model`: Trained ML models for score difference.
  - `scaler`: StandardScaler used on feature data.

- **Steps**:
  1. Normalize fencer names.
  2. Retrieve feature data for both fencers.
  3. Create input features for prediction:
     - Differences in Elo, Win Rate, Elim Win Rate, Rank, Elim Wins
     - Preliminary status flags
  4. Scale features using `scaler`.
  5. Predict win probability:
     - RandomForest prediction
     - XGBoost prediction
     - Elo expected score
     - Average all three predictions
  6. Predict score difference using RandomForest and XGBoost, then average.
  7. Determine predicted scoreline (winner = 15, loser 0–14).
  8. Generate a visual win probability bar chart.
  9. Provide textual analysis of match outcome.

- **Output**:
  - A formatted string containing:
    - Head-to-head matchup
    - Win probability for each fencer
    - Predicted scoreline
    - Analytical summary

## 7. User Interface for Predictions

This section handles interactive user input to select fencers and display head-to-head predictions.

### **7.1 Function: `choose_fencer`**
- **Purpose**: Allow the user to search for and select a fencer by typing part of their name.
- **Inputs**:
  - `prompt`: String prompt for the user (“Fencer A” or “Fencer B”)
- **Steps**:
  1. Prompt user input.
  2. Options:
     - `all`: display the full list of fencers with index numbers.
     - Search text: display matching fencers.
     - `exit`: stop the input loop.
  3. User selects fencer by number.
  4. Validates selection and returns chosen fencer name.
- **Output**: Selected fencer name (string) or `None` if exited.

### **7.2 Main Loop**
- Continuously allow users to predict multiple matches.
- Steps:
  1. Call `choose_fencer` for Fencer A.
  2. Call `choose_fencer` for Fencer B.
  3. Call `predict_head_to_head` with selected fencers.
  4. Print formatted prediction output.
  5. Ask user if they want to predict another match.
  6. Exit loop if user answers `no` or types `exit`.

## 8. Training Machine Learning Models

This section explains how the models are trained to predict win probability and score differences using historical fencing data.

### 8.1 Preparing Data

- Combine features and elimination datasets from all tournaments into single datasets.
- Create a head-to-head dataset that captures differences in Elo ratings, win rates, ranks, elimination wins, and preliminary participation flags between two fencers.
- Define the feature matrix (`X`) and target variables:
  - `y_win`: binary outcome (1 if Fencer A wins, 0 if Fencer B wins)
  - `y_score_diff`: score difference between Fencer A and Fencer B

### 8.2 Train-Test Split

- Split the data into training and testing sets, typically using an 80/20 ratio.
- Standardize the features to ensure all variables are on a comparable scale.

### 8.3 RandomForestRegressor

- Trains models to predict both win probability and score difference.
- Uses hyperparameter tuning to optimize the number of trees, tree depth, and minimum samples per split.
- Provides a baseline ensemble model for performance comparison.

### 8.4 XGBoost Regressor

- Trains gradient-boosted tree models for win probability and score difference prediction.
- Includes hyperparameter tuning for the number of estimators, tree depth, and learning rate.
- Helps capture complex relationships in the data that RandomForest may not fully capture.

### 8.5 Summary

- The combination of RandomForest, XGBoost, and Elo-based features provides a robust prediction framework.
- Models are trained to output probabilities and score differences, which are later combined for final head-to-head predictions.
- This approach allows the system to account for both historical performance (Elo, rankings) and match-specific statistics (pool and elimination features).

## 9. Head-to-Head Prediction Function

This section describes how the system predicts the outcome of a match between two fencers using the trained models.

### 9.1 Input Handling

- Accepts the names of two fencers from the user.
- Cleans and standardizes the names to match the feature dataset.
- Checks if both fencers exist in the combined features dataset.

### 9.2 Feature Construction

- Computes differences between the two fencers for key metrics:
  - Elo rating
  - Win rate from pools
  - Elimination win rate
  - Rank
  - Number of elimination wins
  - Flags indicating if the fencer is from preliminary rounds

### 9.3 Prediction

- Standardizes the input features using the same scaler from training.
- Uses trained RandomForest and XGBoost models to predict:
  - Win probability
  - Score difference
- Combines predictions from the ML models and Elo expected score to compute a final win probability.

### 9.4 Scoreline Estimation

- Converts win probability and predicted score difference into a predicted match score.
- Ensures score rules are respected (winner = 15, loser between 0–14).

### 9.5 Output Summary

- Displays a textual head-to-head summary with:
  - Win probabilities as a visual bar
  - Predicted scoreline
  - Analysis of match competitiveness (strong favorite, slight advantage, or closely contested)
- Designed for user-friendly interpretation and multiple sequential predictions.

## 10. User Interaction and Main Loop

This section handles the interactive part of the program where the user selects fencers and receives predictions.

### 10.1 Fencer Selection

- Users can input part of a fencer's name to search.
- Provides a numbered list of matches if multiple fencers are found.
- Users select the fencer by entering the corresponding number.
- Options:
  - Type "all" to display the full list of fencers.
  - Type "exit" to quit the selection process.

### 10.2 Main Prediction Loop

- Prompts the user to choose two fencers (Fencer A and Fencer B).
- Calls the head-to-head prediction function with the selected fencers.
- Displays the prediction results including:
  - Win probability
  - Predicted scoreline
  - Analysis summary
- Allows the user to predict another match or exit the loop.

### 10.3 Design Considerations

- Provides error handling for invalid or non-existent fencer names.
- Ensures that users can interactively explore match outcomes.
- Modular design allows multiple predictions without restarting the program.

## 11. Model Training and Evaluation

This section focuses on training machine learning models to predict the outcome and score difference of fencer matches.

### 11.1 Feature Preparation

- Uses head-to-head features created from fencer statistics:
  - Elo rating difference
  - Pool win rate difference
  - Elimination win rate difference
  - Rank difference
  - Total elimination wins difference
  - Preliminary round indicators for each fencer
- Splits the dataset into features (`X`) and targets:
  - `y_win` for match result (win/loss)
  - `y_score_diff` for predicted score difference

### 11.2 Data Splitting and Scaling

- Splits the data into training and test sets using an 80-20 split.
- Standardizes the features using `StandardScaler` for model stability and performance.

### 11.3 Machine Learning Models

- **RandomForestRegressor**:
  - Trained for both win probability and score difference.
  - Hyperparameter tuning using `GridSearchCV`:
    - Number of trees (`n_estimators`)
    - Maximum depth (`max_depth`)
    - Minimum samples for split (`min_samples_split`)
- **XGBoost Regressor**:
  - Trained for both win probability and score difference.
  - Hyperparameter tuning using `GridSearchCV`:
    - Number of estimators
    - Maximum depth
    - Learning rate

### 11.4 Evaluation Metrics

- Evaluates model performance using Mean Squared Error (MSE) on the test set.
- Prints the best parameters found for each model along with MSE scores.
- Ensures that both RandomForest and XGBoost models are optimized for prediction accuracy.

### 11.5 Purpose

- Provides a robust prediction framework for sabre matches.
- Combines multiple models to improve reliability of predicted win probabilities and scorelines.
- Lays the foundation for interactive head-to-head predictions.

## 12. Head-to-Head Prediction Function

This section describes the function used to predict the outcome and scoreline of a match between two fencers.

### 12.1 Input Parameters

- `fencer_a_name` and `fencer_b_name`: Names of the two fencers.
- `combined_features`: DataFrame containing all fencer features.
- `fencer_elos`: Dictionary of Elo ratings for each fencer.
- Trained machine learning models:
  - `rf_win_best_model` and `xgb_win_best_model` for win probability.
  - `rf_score_best_model` and `xgb_score_best_model` for score difference.
- `scaler`: StandardScaler used to standardize input features.

### 12.2 Feature Construction

- Calculates differences between fencers for key features:
  - Elo rating, win rates, elimination win rates, rank, and elimination wins.
- Includes indicators for whether each fencer is in the preliminary round.
- Scales features using the provided `scaler`.

### 12.3 Predictions

- **Win Probability**:
  - RandomForest, XGBoost, and Elo models are used.
  - Predictions are combined using a simple average.
  - Result is clipped between 0 and 1.
- **Score Difference**:
  - RandomForest and XGBoost predictions are averaged.
  - Scores are capped appropriately (winner = 15, loser between 0–14).

### 12.4 Output Formatting

- Displays a clear comparison:
  - Win probability bar for each fencer.
  - Predicted scoreline.
  - Textual summary analysis:
    - Strongly favored
    - Slight advantage
    - Close match
- Includes the predicted winner.

### 12.5 Purpose

- Provides an interactive way for users to predict head-to-head outcomes.
- Combines ML models with Elo ratings for more reliable predictions.
- Outputs predictions in a user-friendly and interpretable format.

## 13. User Interaction Loop

This section describes how the program interacts with the user to make head-to-head predictions.

### 13.1 Fencer Selection

- Users are prompted to select Fencer A and Fencer B for the prediction.
- Provides three options:
  - Type part of the fencer's name to search.
  - Type "all" to display the full list of fencers.
  - Type "exit" to quit the prediction loop.
- If multiple matches are found, the user selects the fencer by number.

### 13.2 Prediction Execution

- Once two fencers are selected, the program calls the head-to-head prediction function.
- Displays:
  - Win probabilities for each fencer (graphical bar and percentage).
  - Predicted scoreline.
  - Summary analysis of match outcome.
- Uses combined features, Elo ratings, and trained ML models.

### 13.3 Repeat Predictions

- After a prediction, the user is prompted to predict another match.
- Users can continue predicting as many matches as desired or exit the loop.

### 13.4 Purpose

- Provides an interactive, user-friendly interface for exploring predictions.
- Allows for quick comparison between any two fencers in the dataset.
- Makes the analysis accessible even for non-technical users.

## 14. Model Training and Evaluation

This section explains how machine learning models are trained and evaluated to predict win probabilities and score differences.

### 14.1 Feature Selection

- Uses the head-to-head dataset created from fencer features.
- Input features (`X`) include:
  - `Elo_Diff`: Difference in Elo ratings.
  - `Win_Rate_Diff`: Difference in pool win rates.
  - `Elim_Win_Rate_Diff`: Difference in elimination win rates.
  - `Rank_Diff`: Difference in ranking.
  - `Elim_Wins_Diff`: Difference in elimination wins.
  - `IsPreliminary_A`, `IsPreliminary_B`: Indicators if the fencers are preliminary-stage participants.
- Target variables (`y`) include:
  - `Result`: Binary outcome (1 if Fencer A wins, 0 if Fencer B wins).
  - `Score_Diff`: Numerical score difference.

### 14.2 Train-Test Split

- Splits data into training and testing sets (80%-20%).
- Standardizes features using `StandardScaler` for better model performance.

### 14.3 Random Forest Models

- RandomForestRegressor is used for predicting both:
  - Win probability (`Result`).
  - Score difference (`Score_Diff`).
- Hyperparameter tuning via `GridSearchCV`:
  - `n_estimators`: 100, 200, 300
  - `max_depth`: 10, 20, 30, None
  - `min_samples_split`: 2, 5, 10
- Best models are selected based on cross-validated mean squared error (MSE).

### 14.4 XGBoost Models

- XGBRegressor is also used for predicting:
  - Win probability
  - Score difference
- Hyperparameter tuning via `GridSearchCV`:
  - `n_estimators`: 100, 200, 300
  - `max_depth`: 3, 5, 7
  - `learning_rate`: 0.01, 0.1, 0.3
- Best models are selected based on cross-validated mean squared error.

### 14.5 Evaluation Metrics

- Mean Squared Error (MSE) is calculated for all trained models.
- Allows comparison of model accuracy and selection of the best-performing model for predictions.
- Ensures that the predictions are both statistically and practically reliable.

## 15. Head-to-Head Prediction Function

This section explains how the code predicts the outcome and scoreline of a match between two fencers.

### 15.1 Inputs

- Names of Fencer A and Fencer B.
- Combined fencer features dataset.
- Fencer Elo ratings.
- Trained models:
  - Random Forest and XGBoost for win probability.
  - Random Forest and XGBoost for score difference.
- Scaler used for feature standardization.

### 15.2 Feature Preparation

- Retrieves feature values for both fencers from the combined features dataset.
- Computes differences between the fencers for features such as:
  - Elo rating
  - Win rate
  - Elimination win rate
  - Rank
  - Elimination wins
- Adds binary indicators for whether each fencer is in the preliminary stage.
- Scales the features using the pre-fitted `StandardScaler`.

### 15.3 Predictions

- Predicts win probability using:
  - Random Forest model
  - XGBoost model
  - Elo expected score
- Combines predictions from all three sources into a single average win probability.
- Predicts score difference using Random Forest and XGBoost, averaged together.

### 15.4 Scoreline Derivation

- Winner is assigned 15 points.
- Loser’s score is derived from predicted score difference, capped between 0–14.
- Ensures realistic fencing scoring rules.

### 15.5 Output

- Displays a visual bar representation of win probabilities for both fencers.
- Provides the predicted winner and scoreline.
- Includes a textual analysis summarizing which fencer is favored and how close the match is expected to be.

## 16. Usage Instructions / Examples

This section explains how to interact with the prediction program and get head-to-head outcomes.

### 16.1 Running the Program
- The program loads multiple tournament datasets (pool results, elimination rounds, final rankings, athletes) for the 2023/2024 season.
- Data is cleaned, processed, and features are engineered automatically when the program runs.

### 16.2 Choosing Fencers
- The user is prompted to select **Fencer A** and **Fencer B** for a head-to-head prediction.
- Options for searching:
  - Type part of a fencer's name to filter results.
  - Type `all` to display the full list of fencers.
  - Type `exit` to quit the program.
- If multiple matches are found for a search, the program asks the user to select by number.

### 16.3 Head-to-Head Predictions
- Once both fencers are selected, the program calculates:
  - **Win probability** using RandomForest, XGBoost, and Elo rating (combined for final probability).
  - **Score difference** using RandomForest and XGBoost models (averaged for final prediction).
  - Predicted **scoreline** for the match with proper caps (winner = 15, loser between 0–14).
- The output includes a **visual probability bar**, winner, scoreline, and a short analysis summary.

### 16.4 Example Interaction
1. Prompt: `Fencer A (type part of the name or 'all' to see full list, 'exit' to quit):`
2. User types: `Smith`
3. Program lists matching names with numbers.
4. User selects: `1`
5. Repeat for Fencer B.
6. Program outputs:
```python
Fencer A (type part of the name or 'all' to see full list, 'exit' to quit): mao
1. Kokubo Mao
Select by number: 1

Fencer B (type part of the name or 'all' to see full list, 'exit' to quit): yag
1. Moran Yago
2. Yagodka Andriy
Select by number: 2

Head-to-head: Kokubo Mao vs Yagodka Andriy


===== HEAD-TO-HEAD PREDICTION =====
Fencers: Kokubo Mao vs Yagodka Andriy

Win Probability:
Kokubo Mao: ███████████--------- 56.0%
Yagodka Andriy: ████████------------ 44.0%

Predicted Outcome: Kokubo Mao
Predicted Scoreline: Kokubo Mao 15 : 14 Yagodka Andriy
Analysis: Kokubo Mao has a slight advantage over Yagodka Andriy.
 ==================================
```
 ### 16.5 Notes on Head-to-Head Predictions

- The win probability is calculated using a combination of RandomForest, XGBoost, and Elo ratings.
- Scoreline predictions are derived from the expected win probability and historical match data.
- The visual bar (e.g., `█████-----`) gives a quick intuitive sense of probability.
- `Predicted Outcome` shows which fencer is expected to win.
- `Analysis` provides a textual summary of the prediction, highlighting whether it’s a close match or a clear favorite.

## 16.6 User Interaction

- Users are prompted to select two fencers for prediction.
- Partial names can be typed to search, or `all` to see the full list of fencers.
- The system validates input and allows re-selection if no match is found.
- Users can exit the loop anytime by typing `exit`.

## 16.7 Prediction Loop

- Once fencers are selected, the model predicts:
  - Win probability for each fencer
  - Scoreline prediction
  - Outcome summary
- Users can optionally predict another match without restarting the program.
- The loop continues until the user chooses to exit.

## 16.8 Machine Learning Models

- **RandomForestRegressor** and **XGBoost** are used for predicting:
  - Win probability
  - Score difference
- Models are trained on head-to-head features and fencer statistics.
- Hyperparameter tuning is done using `GridSearchCV` for optimal performance.
- Model predictions are combined with Elo-based predictions for robustness.

## 16.9 Feature Engineering

- Features include:
  - Elo rating difference
  - Win rates and elimination win rates
  - Rank differences
  - Total elimination wins
  - Preliminary round participation
- Features are merged into a single DataFrame for each fencer.
- Missing values are filled with defaults to ensure all fencers have complete features.

## 16.10 Elo Rating System

- Initial Elo is derived from the final ranking points of fencers.
- Elo updates after each match using:
  - Expected score
  - Actual result
  - K-factor adjustment
- Provides an additional baseline for win probability prediction.

## 16.11 Data Cleaning

- Ensures consistent formatting for:
  - Fencer names
  - Countries
  - Scores and rounds
- Handles missing values and converts data types as needed.
- Standardizes pool and elimination data for analysis.

## 16.12 Data Loading

- Users upload tournament CSV files:
  - `pool_results.csv`
  - `elimination_rounds.csv`
  - `final_ranking.csv`
  - `athletes.csv`
- Column names are cleaned to remove extra characters and whitespace.
- Data is organized into consistent DataFrames for processing.

## 16.13 Head-to-Head Features

- For each match:
  - Differences in Elo, win rate, elimination wins, and rank are calculated.
  - Preliminary round indicators are included.
  - Score difference and result are recorded.
- These features are used as input for ML models.

## 16.14 Output Summary

- The final output includes:
  - Win probability bars for each fencer
  - Predicted outcome
  - Predicted scoreline
  - Text analysis summary
- This provides both numerical and visual insight into the match prediction.

## 16.15 System Requirements

- Python 3.8+
- Libraries:
  - pandas, numpy, scikit-learn, xgboost
- Optional: Google Colab for file uploads and runtime environment.

## 16.16 Notes & Considerations

- Predictions are based on historical data and may not account for real-time conditions.
- Top-ranked fencers bypass preliminary rounds and are accounted for in feature engineering.
- The system is modular and can be extended to other fencing seasons or weapons with minimal modification.



