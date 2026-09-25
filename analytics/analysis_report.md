# Titanic Analytics Report

Raw shape: (891, 15)

Raw df.info():
<class 'pandas.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    str     
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    str     
 8   class        891 non-null    category
 9   who          891 non-null    str     
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    str     
 13  alive        891 non-null    str     
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), str(5)
memory usage: 80.7 KB


Missing-value percentages before cleaning:
age            19.87
embarked        0.22
deck           77.22
embark_town     0.22

deck was dropped because its measured missingness is above 30%; imputing that much missing cabin information would be unreliable.

age missingness (19.87%) was between 5% and 30%, so it was median-imputed.

embarked missingness (0.22%) was below 5%, so affected rows were dropped.

Shape after cleaning: (889, 14)

Profile:
             count unique          top freq       mean        std   min     25%      50%   75%       max
survived     889.0    NaN          NaN  NaN   0.382452    0.48626   0.0     0.0      0.0   1.0       1.0
pclass       889.0    NaN          NaN  NaN   2.311586     0.8347   1.0     2.0      3.0   3.0       3.0
sex            889      2         male  577        NaN        NaN   NaN     NaN      NaN   NaN       NaN
age          889.0    NaN          NaN  NaN  29.315152  12.984932  0.42    22.0     28.0  35.0      80.0
sibsp        889.0    NaN          NaN  NaN   0.524184   1.103705   0.0     0.0      0.0   1.0       8.0
parch        889.0    NaN          NaN  NaN   0.382452   0.806761   0.0     0.0      0.0   0.0       6.0
fare         889.0    NaN          NaN  NaN  32.096681  49.697504   0.0  7.8958  14.4542  31.0  512.3292
embarked       889      3            S  644        NaN        NaN   NaN     NaN      NaN   NaN       NaN
class          889      3        Third  491        NaN        NaN   NaN     NaN      NaN   NaN       NaN
who            889      3          man  537        NaN        NaN   NaN     NaN      NaN   NaN       NaN
adult_male     889      2         True  537        NaN        NaN   NaN     NaN      NaN   NaN       NaN
embark_town    889      3  Southampton  644        NaN        NaN   NaN     NaN      NaN   NaN       NaN
alive          889      2           no  549        NaN        NaN   NaN     NaN      NaN   NaN       NaN
alone          889      2         True  535        NaN        NaN   NaN     NaN      NaN   NaN       NaN

age IQR outlier count: 65

fare IQR outlier count: 114

Fare mean=32.10, median=14.45, mode=8.05; the distribution is right-skewed by mean/median/mode ordering.

Survival rate by sex:
sex
female    0.740385
male      0.188908

Survival rate by pclass:
pclass
1    0.626168
2    0.472826
3    0.242363

Survival rate by sex and pclass:
sex     pclass
female  1         0.967391
        2         0.921053
        3         0.500000
male    1         0.368852
        2         0.157407
        3         0.135447

Required six-column correlation matrix:
          survived    pclass       age     sibsp     parch      fare
survived  1.000000 -0.335549 -0.069822 -0.034040  0.083151  0.255290
pclass   -0.335549  1.000000 -0.336512  0.081656  0.016824 -0.548193
age      -0.069822 -0.336512  1.000000 -0.232543 -0.171485  0.093707
sibsp    -0.034040  0.081656 -0.232543  1.000000  0.414542  0.160887
parch     0.083151  0.016824 -0.171485  0.414542  1.000000  0.217532
fare      0.255290 -0.548193  0.093707  0.160887  0.217532  1.000000

Two strongest absolute off-diagonal correlations: pclass-fare=-0.548; sibsp-parch=0.415

Chart interpretation: The age histogram and box plot show the central passenger age and identify extreme ages using the IQR rule.

Chart interpretation: The fare charts show a long upper tail; higher fares cluster with first-class travel and visibly different survival outcomes.

Chart interpretation: Survival bars show a strong sex difference and a class gradient, with women and higher classes surviving at higher rates.

Chart interpretation: The correlation heatmap and multivariate plots connect class, fare, age, and survival while making the strongest numeric associations visible.

Standardization check (mean/std):
         age    fare
mean  0.0000  0.0000
std   1.0006  1.0006

Stratified split class balance: train=0.383, test=0.382; stratification preserves the survival ratio in both sets.

Logistic Regression confusion matrix:
[[97 13]
 [21 47]]
              precision    recall  f1-score   support

           0       0.82      0.88      0.85       110
           1       0.78      0.69      0.73        68

    accuracy                           0.81       178
   macro avg       0.80      0.79      0.79       178
weighted avg       0.81      0.81      0.81       178


Decision Tree confusion matrix:
[[98 12]
 [30 38]]
              precision    recall  f1-score   support

           0       0.77      0.89      0.82       110
           1       0.76      0.56      0.64        68

    accuracy                           0.76       178
   macro avg       0.76      0.72      0.73       178
weighted avg       0.76      0.76      0.75       178


Random Forest confusion matrix:
[[96 14]
 [19 49]]
              precision    recall  f1-score   support

           0       0.83      0.87      0.85       110
           1       0.78      0.72      0.75        68

    accuracy                           0.81       178
   macro avg       0.81      0.80      0.80       178
weighted avg       0.81      0.81      0.81       178


Classification comparison table:
              model  accuracy  precision  recall     f1    auc
Logistic Regression    0.8090     0.7833  0.6912 0.7344 0.8610
      Decision Tree    0.7640     0.7600  0.5588 0.6441 0.8374
      Random Forest    0.8146     0.7778  0.7206 0.7481 0.8182

Recommendation: deploy Random Forest because it has the strongest observed F1 score (0.748) on the held-out stratified test set. Its precision is 0.778, recall is 0.721, and AUC is 0.818. These metrics balance identifying survivors with avoiding incorrect positive predictions; the final choice should be revisited if operational costs change.

Class balance: {0: 0.618, 1: 0.382}
Imbalance comparison:
              variant  precision  recall     f1
             baseline     0.7833  0.6912 0.7344
class_weight_balanced     0.7183  0.7500 0.7338
     SMOTE_train_only     0.7353  0.7353 0.7353
The preferred imbalance strategy should be selected by the precision/recall trade-off; SMOTE is confined to the training fold by the imbalanced-learn pipeline.

Random Forest GridSearchCV best parameters: {'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 150}; best CV score=0.7451; OOB score=0.8186.

Reloaded complete pipeline prediction on raw input: [0]

Regression metrics (fare prediction):
{
  "MAE": 21.098604259640446,
  "RMSE": 41.70210467903215,
  "R2": 0.3481625721605429,
  "Adjusted_R2": 0.30913039085279104
}

Residual conclusion: the residual spread is not strongly heteroscedastic in this run based on prediction-bin spread ratios; inspect the saved residual plot for the visual funnel pattern.

Final model comparison table with separate classification and regression metric groups:
 accuracy  precision  recall     f1    auc     model_type               model     MAE    RMSE     R2  Adjusted_R2
   0.8090     0.7833  0.6912 0.7344 0.8610 classification Logistic Regression     NaN     NaN    NaN          NaN
   0.7640     0.7600  0.5588 0.6441 0.8374 classification       Decision Tree     NaN     NaN    NaN          NaN
   0.8146     0.7778  0.7206 0.7481 0.8182 classification       Random Forest     NaN     NaN    NaN          NaN
      NaN        NaN     NaN    NaN    NaN     regression   Linear Regression 21.0986 41.7021 0.3482       0.3091