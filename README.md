# Sabre Fencing Win & Scoreline Prediction (2023/2024 Season)

Predict match outcome and scoreline by assigning Elo ratings to fencers and using an ensemble of Random Forest and XGBoost models to predict win probabilities and expected scorelines for head-to-head matches.

The project uses data from the 2023/2024 season, including:

- **Elimination rounds**: Scores and winners of head-to-head matches.
- **Pool results**: Performance in preliminary rounds.
- **Final rankings**: Official FIE rankings for the 2023/2024 season and their final ranking for the competitions used in the datasets.
- **Athlete data**: Personal information like age, nationality, and points.

The goal is to provide:

- Head-to-head win probability
- Predicted scoreline
- Insights into match competitiveness

## 2. Data Structure

The project uses four main datasets for each tournament which are scrapped from FIE's website.

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

## 3. Elo class creations, Data Cleaning & Feature Engineering

This section handles preprocessing the raw datasets, creating an elo class for rating and creating features for ML models.

### **3.1 Cleaning Column Names function**
- Removes extra spaces, Byte Order Mark characters found in CSV, and zero-width spaces.
- Ensures uniform column names across datasets.

### **3.1 loading dataset**
- It loads the 4 datasets mentioned in section 2 for each competition as pandas dataframes and encodes with utf-8-sig so that BOM characters are removed
- Clean column names function is applied to all 4 pandas data frames
- Forces elimination round dataframes to fixed columns and validate the structure
  
### **3.1 dataframe cleaning**
- Loops through the 4 dataframes, ensures that the columns are there and drops any unexpected columns
- Handles missing rows where names, winner or scores by dropping
- Standardise all variables to their supposed data type (round number to integer, names to string)
- created lists to organise top 16 fencers who are fencing in the main tableau and the rest who have to compete for the remaining 48 spots in preliminaries

### **Elo class creation and elo calculation**
 - Elo is used to calculate relative skill level between fencer and all are given a base elo of 1000 points
 - A default value K=5 is set and it controls the sensitivity of rating changes
 - Expected score returns a probability that player A wins and actual score returns the probability of the actual outcome (1 = win, 0 = Lose)
 - An update rating function then uses the difference of these scores multiplied by the K factor to calculate the new elos of both fencers.
 - Elo is calculated as it it loops through every elimination bout in the eliminations df and updates the elo rating of the fencers. An elo_df is created with the fencer's final elo and a dictionary of fencer_elos for easy lookup

## 4. Head-to-Head Feature Engineering

This section creates features for each match between two fencers based on their individual statistics.

### **4.1 Feature Calculation for Each Match**

A dataframe is created with each row being a fencer and each column representing a feature.

The definition of the feature is listed below:
- **V** → Number of victories in pool round.  
- **M** → Total pool matches fenced.  
- **Rk.** → Average pool ranking.  
- **Win_Rate** → Pool win rate (`V / M`).  
- **Elim_Wins** → Number of wins in elimination rounds.  
- **Elim_Matches** → Total elimination matches fenced.  
- **Elo** → Elo rating after elimination rounds.  
- **Rank** → Official FIE ranking.  
- **Elim_Win_Rate** → Elimination win rate (`Elim_Wins / Elim_Matches`).  
- **IsPreliminary** → 1 if fencer came from preliminary rounds, 0 otherwise.


### **4.2 Main execution when the datasets are uploaded**
- 2 lists are created: all_features to store the feature tables for each tournament and all_elimination_dfs to store the cleaned elimination_dfs for each tournament
- Loop is created to prompt user to upload the 4 required files for each tournament
- Once the 4 files are uploaded, the data is cleaned using the clean_data function in section 3
- Lastly, the feature tables and elimination results from each tournament are merged into a big dataset respectively


## 5. Machine Learning Model Training

This section trains RandomForest and XGBoost models to predict win probability and score differences using fencer performance metrics differences as features which are standardized and hyperparameter tuned with 5-fold GridSearchCV (cross-validation).

### **5.1 Dataset Preparation**
Creates a head to head dataset and calculates the feature differences in both fencers and add the win outcome and scoreline columns of both fencers

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

- Train-test split: 80% training, 20% testing and random_state=42 for reproducible splits
- Standardize features using `StandardScaler` to zero mean and unit variance as we will be using gradient-based models like XGBoost

### **5.2 Models**
- Initialise hyperparameter tuning for both models using 5-fold cross validation to find the best parameters.
- The model with the best parameters are fitted onto the training data to predict and compare with test set data
- Mean Squared Error is used for model evaluation
- This setup is used for the first target variable y_win and then repeated with the target variable changed to y_score_diff
  
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
     - Preliminary status
  4. Features are then scaled.
  5. Calculate win probability: 
     - RandomForest prediction
     - XGBoost prediction
     - Elo expected score
     - Average all three predictions
  6. Predict score difference using RandomForest and XGBoost, then average.
  7. Determine predicted scoreline (winner = 15, loser 0–14).
  8. The system generates a user-friendly match prediction output, showing a visual bar chart of win probabilities, the predicted scoreline, and a short textual analysis that highlights how close or one-sided the bout is.

## 7. User Interface for Predictions/Usage

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


### 7.3 Example Interaction
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
 
## 8 System Requirements

- Python 3.8+
- Libraries:
  - pandas, numpy, scikit-learn, xgboost
- Google Colab for file uploads and runtime environment.

## 9 Notes & Considerations
- Predictions are based on historical data and may not account for real-time conditions.
- The system is modular and can be extended to other fencing seasons or weapons with minimal modification.



