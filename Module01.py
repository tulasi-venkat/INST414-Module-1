import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# loads the dataset
file_path = 'Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv'
data = pd.read_csv(file_path)

# defines mappings for life satisfaction, mental health, and ethnic identity responses
satisfaction_mapping = {
    "Strongly disagree": 1, "Disagree": 2, "Slightly disagree": 3,
    "Neither agree or disagree": 4, "Slightly agree": 5, "Agree": 6, "Strongly agree": 7
}
mental_health_mapping = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5
}
ethnic_identity_mapping = {
    "Not at all": 1, "Not very close": 2, "Somewhat close": 3, "Very close": 4
}

# applies the mappings to each column
data['Life_Satisfaction'] = data['Satisfied With Life 1'].map(satisfaction_mapping)
data['Mental_Health'] = data['Present Mental Health'].map(mental_health_mapping)
data['Ethnic_Identity'] = data['Identify Ethnically'].map(ethnic_identity_mapping)

# cleaning for relevant columns
analysis_data = data[['Life_Satisfaction', 'Mental_Health', 'Ethnic_Identity', 'Discrimination ']].dropna()
analysis_data.columns = ['Life_Satisfaction', 'Mental_Health', 'Ethnic_Identity', 'Discrimination']

# calculates pearson correlations between variables
correlations = {
    'Discrimination & Mental Health': pearsonr(analysis_data['Discrimination'], analysis_data['Mental_Health'])[0],
    'Discrimination & Life Satisfaction': pearsonr(analysis_data['Discrimination'], analysis_data['Life_Satisfaction'])[0],
    'Ethnic Identity & Life Satisfaction': pearsonr(analysis_data['Ethnic_Identity'], analysis_data['Life_Satisfaction'])[0],
    'Ethnic Identity & Mental Health': pearsonr(analysis_data['Ethnic_Identity'], analysis_data['Mental_Health'])[0]
}

# prints correlation results
print("Correlations:", correlations)

# regression to predict mental health from discrimination and ethnic identity
X = analysis_data[['Discrimination', 'Ethnic_Identity']]
X = sm.add_constant(X)  # Add constant for intercept
y = analysis_data['Mental_Health']
regression_model = sm.OLS(y, X).fit()

# displays regression summary
print(regression_model.summary())

# correlation data as a DataFrame in table format
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# plots correlation heatmap 
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df.T, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Heatmap: Discrimination, Mental Health, Life Satisfaction, and Ethnic Identity")
plt.show()

# visualizes regression coefficients from the  fitted regression model
coefficients = pd.DataFrame(regression_model.params, columns=['Coefficient'])
coefficients.index.name = 'Variable'
coefficients = coefficients.reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients, x='Variable', y='Coefficient')
plt.title("Regression Coefficients for Predicting Mental Health")
plt.show()
