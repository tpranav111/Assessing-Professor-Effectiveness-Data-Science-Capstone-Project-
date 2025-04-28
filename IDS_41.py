import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)

np.random.seed(11931503)

# load the numeric dataset and add column names
df_main = pd.read_csv('rmpCapstoneNum.csv', header=None, names=[
    'average_rating',
    'average_difficulty',
    'number_of_ratings',
    'received_pepper',
    'proportion_take_again',
    'online_ratings',
    'male',
    'female'
])

# load the qualitative dataset and add column names
df_qual = pd.read_csv('rmpCapstoneQual.csv', header=None, names=[
    'major_field',
    'university',
    'us_state'
])

# load the tags dataset and add column names
df_tags = pd.read_csv('rmpCapstoneTags.csv', header=None, names=[
    'tough_grader',
    'good_feedback',
    'respected',
    'lots_to_read',
    'participation_matters',
    'dont_skip_class',
    'lots_of_homework',
    'inspirational',
    'pop_quizzes',
    'accessible',
    'so_many_papers',
    'clear_grading',
    'hilarious',
    'test_heavy',
    'graded_by_few',
    'amazing_lectures',
    'caring',
    'extra_credit',
    'group_projects',
    'lecture_heavy'
])


# # Q1
# Activists have asserted that there is a strong gender bias in student evaluations of professors, with
# male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues,
# skeptics have pointed out that this research is of technically poor quality, either due to a low sample
# size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching
# experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015). We would like you to
# answer the question whether there is evidence of a pro-male gender bias in this dataset.
# Hint: A significance test is probably required.

# In[8]:


relevant_columns = ['average_rating', 'male', 'female']
missing_data_summary = df_main[relevant_columns].isnull().sum()

# no overlapping entries (e.g., a professor marked as both male and female)
overlapping_gender = df_main[(df_main['male'] == 1) & (df_main['female'] == 1)]

df_main_cleaned = df_main.dropna(subset=['average_rating'])

# remove overlapping gender entries
df_main_cleaned = df_main_cleaned[~((df_main_cleaned['male'] == 1) & (df_main_cleaned['female'] == 1))]

male_ratings = df_main_cleaned[df_main_cleaned['male'] == 1]['average_rating']
female_ratings = df_main_cleaned[df_main_cleaned['female'] == 1]['average_rating']


# In[9]:


# Visualize distributions of average ratings for male and female professors
plt.figure(figsize=(12, 6))

# Histogram for male ratings
plt.subplot(1, 2, 1)
plt.hist(male_ratings, bins=30, alpha=0.7, color='blue', label='Male Ratings')
plt.title('Distribution of Male Professors\' Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram for female ratings
plt.subplot(1, 2, 2)
plt.hist(female_ratings, bins=30, alpha=0.7, color='green', label='Female Ratings')
plt.title('Distribution of Female Professors\' Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# In[10]:


import numpy as np

# calculate the observed difference in means
observed_diff = male_ratings.mean() - female_ratings.mean()

# combine the two groups
combined_ratings = np.concatenate([male_ratings, female_ratings])

# number of permutations
n_permutations = 10000

# initialize a counter for the number of times the permuted difference is greater than or equal to the observed difference
count_extreme = 0

# perform permutations
for _ in range(n_permutations):
    # shuffle the combined ratings
    np.random.shuffle(combined_ratings)

    # split the shuffled data into two groups
    perm_male = combined_ratings[:len(male_ratings)]
    perm_female = combined_ratings[len(male_ratings):]

    # calculate the difference in means for the permuted groups
    perm_diff = perm_male.mean() - perm_female.mean()

    # count if the permuted difference is greater than or equal to the observed difference
    if perm_diff >= observed_diff:
        count_extreme += 1

# calculate the p-value
perm_p_value = count_extreme / n_permutations

# results summary
perm_test_results = {
    "Observed Difference in Means": observed_diff,
    "P-Value (Permutation Test)": perm_p_value
}

perm_test_results


# In[11]:


import matplotlib.pyplot as plt

# generate the null distribution from the permutations
null_distribution = []
for _ in range(n_permutations):
    np.random.shuffle(combined_ratings)
    perm_male = combined_ratings[:len(male_ratings)]
    perm_female = combined_ratings[len(male_ratings):]
    null_distribution.append(perm_male.mean() - perm_female.mean())

# plot the null distribution
plt.hist(null_distribution, bins=30, alpha=0.7, edgecolor='black', label='Null Distribution')
plt.axvline(observed_diff, color='red', linestyle='--', label='Observed Difference')
plt.title('Permutation Test: Null Distribution of Difference in Means')
plt.xlabel('Difference in Means')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# To investigate whether there is evidence of a pro-male gender bias in professor ratings, we performed a permutation test comparing the average ratings of male and female professors. A permutation test was chosen because the distribution of ratings was not normal, making it inappropriate to use parametric tests like the t-test. Null and Alternative Hypotheses:
# Null Hypothesis (H0): There is no difference in average ratings between male and female professors (no gender bias).
# Alternative Hypothesis (Ha): Male professors have higher average ratings than female professors (pro-male gender bias).
# We calculated the observed difference in means between male and female professors' ratings as 0.067. By randomly shuffling the gender labels 10,000 times and computing the difference in means for each permutation, we constructed a null distribution. The resulting p-value was 0.0, indicating that the observed difference is highly unlikely under the null hypothesis of no gender bias. This supports the alternative hypothesis that male professors tend to receive higher ratings than female professors.
#
#

# # Q2
# Is there a gender difference in the spread (variance/dispersion) of the ratings distribution? Again, it
# is advisable to consider the statistical significance of any observed gender differences in this spread.


import numpy as np
import matplotlib.pyplot as plt

# Function for permutation test
def permutation_test(data1, data2, num_permutations=10000):
    # calculate observed difference in variance
    observed_diff = np.var(data1, ddof=1) - np.var(data2, ddof=1)

    # combine the data
    combined = np.concatenate([data1, data2])

    # initialize permutation differences
    perm_diffs = []

    for _ in range(num_permutations):
        # shuffle combined data
        np.random.shuffle(combined)

        # split into two groups
        perm_data1 = combined[:len(data1)]
        perm_data2 = combined[len(data1):]

        # calculate variance difference for permutation
        perm_diff = np.var(perm_data1, ddof=1) - np.var(perm_data2, ddof=1)
        perm_diffs.append(perm_diff)

    # calculate p-value (two-tailed test)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return observed_diff, p_value

# Perform permutation test
obs_diff, p_value = permutation_test(male_ratings, female_ratings)

# Results summary
permutation_test_results = {
    "Observed Variance Difference": obs_diff,
    "P-Value (Permutation Test)": p_value
}

# Visualize the distributions using overlaid density plots
plt.figure(figsize=(8, 6))
male_ratings.plot(kind='kde', label='Male Ratings', alpha=0.7)
female_ratings.plot(kind='kde', label='Female Ratings', alpha=0.7)
plt.title("Kernel Density Estimation of Average Ratings by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.show()

# Display results
permutation_test_results


# To determine whether there is a significant gender difference in the spread (variance) of the ratings distribution, we conducted a permutation test (which is appropriate in this case as it doesn't have assumption about the distribution of the data). The null hypothesis posits that there is no difference in the variance of ratings between males and females, meaning any observed difference is due to random chance. Conversely, the alternative hypothesis asserts that there is a meaningful difference in variance between the two groups. The observed variance difference was -0.131, indicating that female ratings had a slightly greater spread than male ratings. However, with a p-value of 0.0 (based on 10,000 permutations), the results are highly significant, leading us to reject the null hypothesis. This suggests that the variance in ratings is statistically different between genders.

# # Q3
# What is the likely size of both of these effects (gender bias in average rating, gender bias in spread of
# average rating), as estimated from this dataset? Please use 95% confidence and make sure to report
# each/both.


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# function to compute cohen's d
def compute_cohens_d(group1, group2, metric='mean'):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))  # pooled std dev

    if metric == 'mean':
        d = (np.mean(group1) - np.mean(group2)) / sp  # cohen's d for mean diff
    elif metric == 'std':
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        d = (var1 - var2) / 2  # cohen's d for variance difference
    else:
        raise ValueError("Metric must be 'mean' or 'std'")

    return d

# bootstrapping function to compute two-sided confidence intervals
def bootstrap_ci(data1, data2, metric, n_bootstraps=10000, ci=95):
    bootstrapped_d = []
    combined = np.concatenate([data1, data2])
    for _ in range(n_bootstraps):
        sample1 = np.random.choice(combined, size=len(data1), replace=True)
        sample2 = np.random.choice(combined, size=len(data2), replace=True)
        bootstrapped_d.append(compute_cohens_d(sample1, sample2, metric=metric))

    # calculate the two-sided confidence interval
    lower_bound = np.percentile(bootstrapped_d, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_d, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound

# assuming male_ratings and female_ratings are numpy arrays
# compute cohen's d for mean difference
cohens_d_mean = compute_cohens_d(male_ratings, female_ratings, metric='mean')
ci_mean_lower, ci_mean_upper = bootstrap_ci(male_ratings, female_ratings, metric='mean')

# compute cohen's d for std (variance) difference
cohens_d_std = compute_cohens_d(male_ratings, female_ratings, metric='std')
ci_std_lower, ci_std_upper = bootstrap_ci(male_ratings, female_ratings, metric='std')

# results summary
effect_sizes = {
    "Mean Difference": {
        "Cohen's d": cohens_d_mean,
        "95% CI Lower Bound": ci_mean_lower,
        "95% CI Upper Bound": ci_mean_upper
    },
    "Variance Difference": {
        "Cohen's d": cohens_d_std,
        "95% CI Lower Bound": ci_std_lower,
        "95% CI Upper Bound": ci_std_upper
    }
}

# visualizing distributions
plt.figure(figsize=(8, 6))
plt.hist(male_ratings, bins=20, alpha=0.7, label='Male Ratings', density=True)
plt.hist(female_ratings, bins=20, alpha=0.7, label='Female Ratings', density=True)
plt.title("Distributions of Ratings by Gender")
plt.xlabel("Average Rating")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.show()

# display results
effect_sizes


import matplotlib.pyplot as plt

# generate a boxplot to visualize the distributions of male and female ratings
plt.figure(figsize=(8, 6))
plt.boxplot([male_ratings, female_ratings], labels=['Male Ratings', 'Female Ratings'], patch_artist=True)
plt.title("Boxplot of Ratings by Gender")
plt.ylabel("Average Rating")
plt.grid(alpha=0.5, linestyle='--')
plt.show()


# After checking in with Prof. during the office hours, he pointed out that this question is meant to directly relate to Q1, and so a one-sided test was conducted. The null hypothesis posited that there is no significant gender bias in student evaluations of professors, while the alternative hypothesis stated that there is a strong gender bias, with male professors enjoying a boost in ratings due to this bias. Cohen's 𝑑 was used to measure effect sizes: for mean differences, it was computed as the difference in means normalized by the pooled standard deviation, and for variance differences, Prof. suggested using (var 1 ​ −var 2 ​ )/2. Bootstrapping was employed to calculate 95% confidence intervals, a legitimate approach under these conditions as it relies on resampling to model the sampling distribution without parametric assumptions. The results showed a Cohen's 𝑑 d of 0.0599 for the mean difference with a CI of (-0.0170, 0.0171), and a Cohen's 𝑑 of -0.0656 for the variance difference with a CI of (-0.0149, 0.0153). These findings suggest even though we found statistical significance in the above 2 questions, the effect size is actually very small as they include 0, which indicate not a significant practical significance.

# # Q4
# Is there a gender difference in the tags awarded by students? Make sure to teach each of the 20 tags
# for a potential gender difference and report which of them exhibit a statistically significant different.
# Comment on the 3 most gendered (lowest p-value) and least gendered (highest p-value) tags.

# In[21]:


# Merge the tags dataset with the main dataset to associate tags with gender
df_combined = pd.concat([df_main_cleaned.reset_index(drop=True), df_tags.reset_index(drop=True)], axis=1)

# Check for missing data in the combined dataset
missing_data_tags = df_combined.isnull().sum()

# Drop rows with missing data in relevant columns (gender and tags)
df_combined_cleaned = df_combined.dropna(subset=['male', 'female'] + list(df_tags.columns))

# Prepare male and female groups for each tag
tags = df_tags.columns
gender_tag_results = {}

# Perform Mann-Whitney U Test for each tag
from scipy.stats import mannwhitneyu

for tag in tags:
    male_tag = df_combined_cleaned[df_combined_cleaned['male'] == 1][tag]
    female_tag = df_combined_cleaned[df_combined_cleaned['female'] == 1][tag]

    u_stat, p_value = mannwhitneyu(male_tag, female_tag, alternative='two-sided')
    gender_tag_results[tag] = {"U-Statistic": u_stat, "P-Value": p_value}

# Sort results by p-value to identify most and least gendered tags
sorted_gender_tag_results = sorted(gender_tag_results.items(), key=lambda x: x[1]['P-Value'])

# Display the 3 most and 3 least gendered tags
most_gendered = sorted_gender_tag_results[:3]
least_gendered = sorted_gender_tag_results[-3:]

most_gendered, least_gendered



# Extract average counts for the most and least gendered tags
most_gendered_tags = ["accessible", "lecture_heavy", "pop_quizzes"]
least_gendered_tags = ["participation_matters", "tough_grader", "clear_grading"]

# Calculate average counts for male and female professors for the selected tags
tag_summary = {
    "Tag": [],
    "Gender": [],
    "Average Count": []
}

for tag in most_gendered_tags + least_gendered_tags:
    tag_summary["Tag"].append(tag)
    tag_summary["Gender"].append("Male")
    tag_summary["Average Count"].append(df_combined_cleaned[df_combined_cleaned['male'] == 1][tag].mean())

    tag_summary["Tag"].append(tag)
    tag_summary["Gender"].append("Female")
    tag_summary["Average Count"].append(df_combined_cleaned[df_combined_cleaned['female'] == 1][tag].mean())

# Convert to DataFrame for visualization
tag_summary_df = pd.DataFrame(tag_summary)

# Plot the data
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(data=tag_summary_df, x="Tag", y="Average Count", hue="Gender")
plt.title("Average Counts of Most and Least Gendered Tags by Gender")
plt.ylabel("Average Count")
plt.xlabel("Tag")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# What We Did:
# To explore gender differences in the 20 tags awarded by students, we conducted a Mann-Whitney U Test for each tag. This test was chosen as the tags are count data and may not follow normal distributions. The data was first cleaned to remove rows with missing values in gender or tag counts. For each tag, the test compared the distributions of tag counts between male and female professors. We ranked the results based on p-values to identify the tags most and least associated with gender differences.
#
# What We Found:
# The analysis revealed that the three most gendered tags (lowest p-values) were "accessible" (p=0.201), "lecture heavy" (p=0.216), and "pop quizzes" (p=0.224), though none showed strong statistical significance. Conversely, the least gendered tags (highest p-values) were "participation matters" (p=0.917), "tough grader" (p=0.929), and "clear grading" (p=0.969), indicating no meaningful gender differences for these tags.

# # Q5
# Is there a gender difference in terms of average difficulty? Again, a significance test is indicated.


import numpy as np
import matplotlib.pyplot as plt

# check for missing values in 'average_difficulty' column
missing_difficulty = df_main_cleaned['average_difficulty'].isnull().sum()

# drop rows with missing 'average_difficulty' for simplicity
df_difficulty_cleaned = df_main_cleaned.dropna(subset=['average_difficulty'])

# subset the data by gender
male_difficulty = df_difficulty_cleaned[df_difficulty_cleaned['male'] == 1]['average_difficulty']
female_difficulty = df_difficulty_cleaned[df_difficulty_cleaned['female'] == 1]['average_difficulty']

# calculate observed difference in means
observed_mean_diff = male_difficulty.mean() - female_difficulty.mean()

# combine both groups into one array
combined_difficulty = np.concatenate([male_difficulty, female_difficulty])

# permutation test setup
n_permutations = 10000
permuted_diffs = []

# perform permutation test
for _ in range(n_permutations):
    shuffled = np.random.permutation(combined_difficulty)
    perm_male = shuffled[:len(male_difficulty)]
    perm_female = shuffled[len(male_difficulty):]
    permuted_diffs.append(perm_male.mean() - perm_female.mean())

# calculate p-value (two-sided)
p_value_permutation = (np.sum(np.abs(permuted_diffs) >= np.abs(observed_mean_diff)) / n_permutations)

# summarize results
difficulty_test_results = {
    "Observed Mean Difference": observed_mean_diff,
    "Permutation P-Value (Two-Sided)": p_value_permutation,
    "Male Mean Difficulty": male_difficulty.mean(),
    "Male Std Dev": male_difficulty.std(),
    "Female Mean Difficulty": female_difficulty.mean(),
    "Female Std Dev": female_difficulty.std()
}

# visualize the null distribution
plt.figure(figsize=(10, 6))
plt.hist(permuted_diffs, bins=30, color='skyblue', alpha=0.7, density=True)
plt.axvline(observed_mean_diff, color='red', linestyle='--', label=f'Observed Diff: {observed_mean_diff:.3f}')
plt.axvline(-np.abs(observed_mean_diff), color='red', linestyle='--', label=f'-Observed Diff: {-np.abs(observed_mean_diff):.3f}')
plt.title("Permutation Test Null Distribution of Mean Differences")
plt.xlabel("Mean Difference")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.show()

# display results
difficulty_test_results


# To evaluate whether there is a significant gender difference in average difficulty ratings of professors, a two-sided permutation test was conducted, which does not assume normality and is appropriate for comparing two independent groups. The null hypothesis stated that there is no difference in difficulty ratings between male and female professors, while the alternative hypothesis posited a difference in either direction. The observed mean difference was -0.0056, with male professors having slightly lower ratings, and the permutation p-value was 0.5329, far above the standard significance threshold (α=0.05). Descriptive statistics showed mean difficulty ratings of 2.84 (SD = 0.99) for male professors and 2.85 (SD = 0.99) for female professors. Since the p-value indicates that the observed difference is not statistically significant, we fail to reject the null hypothesis and conclude that there is no evidence of a meaningful gender difference in average difficulty ratings.

# # Q6
# Please quantify the likely size of this effect at 95% confidence.


import numpy as np

# define a function to calculate cohen's d
def calculate_cohens_d(sample1, sample2):
    mean_diff = np.mean(sample1) - np.mean(sample2)
    pooled_std = np.sqrt(
        ((len(sample1) - 1) * np.var(sample1, ddof=1) +
         (len(sample2) - 1) * np.var(sample2, ddof=1)) /
        (len(sample1) + len(sample2) - 2)
    )
    return abs(mean_diff) / pooled_std

# bootstrapping setup
n_bootstraps = 10000
bootstrap_d = []

# perform bootstrapping
for _ in range(n_bootstraps):
    male_sample = np.random.choice(male_difficulty, size=len(male_difficulty), replace=True)
    female_sample = np.random.choice(female_difficulty, size=len(female_difficulty), replace=True)
    bootstrap_d.append(calculate_cohens_d(male_sample, female_sample))

# calculate 95% confidence interval
ci_lower, ci_upper = np.percentile(bootstrap_d, [2.5, 97.5])

# summarize results
{
    "Cohen's d (bootstrap mean)": np.mean(bootstrap_d),
    "95% CI Lower": ci_lower,
    "95% CI Upper": ci_upper
}


# To quantify the likely size of the gender effect on average difficulty ratings, we calculated Cohen's
# d, a standardized measure of effect size, using bootstrapping to estimate the 95% confidence interval. Bootstrapping was chosen as it resamples the data repeatedly to generate a distribution of effect sizes without making strong parametric assumptions. Cohen's
# d was used because it provides a standardized way to measure the magnitude of the mean difference relative to pooled variability, allowing comparison across studies. The bootstrap mean Cohen's
# d was 0.0085, with a 95% confidence interval ranging from 0.0003 to 0.0232, indicating that the effect size is extremely small and likely negligible. This confirms that any gender difference in average difficulty ratings is both statistically and practically insignificant.

import matplotlib.pyplot as plt

# plot the distribution of bootstrapped cohen's d values
plt.figure(figsize=(8, 6))
plt.hist(bootstrap_d, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(bootstrap_d), color='red', linestyle='--', label="Mean Cohen's d")
plt.axvline(ci_lower, color='green', linestyle='--', label='95% CI Lower')
plt.axvline(ci_upper, color='orange', linestyle='--', label='95% CI Upper')
plt.title("Bootstrapped Distribution of Cohen's d (Gender Differences in Difficulty)")
plt.xlabel("Cohen's d")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.show()

#Q7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


random_seed = 11058720
np.random.seed(random_seed)
df = pd.read_csv('rmpCapstoneNum.csv')
df.columns = ['Average Rating', 'Average Difficulty', 'Number of ratings', 'Received a “pepper”?',
              'Take class again', 'Ratings from online class', 'Male', 'Female']
raw_df = df.copy()
df = df.dropna()

k = 5
df = df[df['Number of ratings'] >= k]


X = df[['Average Difficulty', 'Number of ratings', 'Received a “pepper”?',
        'Take class again', 'Ratings from online class', 'Male', 'Female']]

y = df['Average Rating']

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#we remove columns based on analyzing the heatmap
X = df[[ 'Number of ratings', 'Received a “pepper”?',
        'Take class again', 'Ratings from online class', 'Male']]


def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    return vif_data

vif_data = calculate_vif(X)
print("Variance Inflation Factor (VIF):")
print(vif_data)

plt.figure(figsize=(10, 6))
plt.bar(vif_data['Feature'], vif_data['VIF'], color='skyblue', edgecolor='k')
plt.axhline(5, color='red', linestyle='--', linewidth=1.5, label='VIF Threshold (5)')
plt.xlabel('Feature')
plt.ylabel('VIF')
plt.title('Variance Inflation Factor (VIF) for Features')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
X = X.drop(columns=high_vif_features)
print(f"Features removed due to high VIF: {high_vif_features}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_seed )



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)



y_train_pred_linear = linear_model.predict(X_train_scaled)
y_test_pred_linear = linear_model.predict(X_test_scaled)





coefficients = pd.DataFrame({
    "Tag": X.columns,
    "Coefficient": linear_model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Regression Coefficients:")
print(coefficients)

test_rmse_linear = np.sqrt( mean_squared_error(y_test, y_test_pred_linear))
print(f"Linear Regression - Test RMSE: {test_rmse_linear}")
print(f"Standard Deviation of the data is: {np.std(df['Average Rating'])}")
test_r2_linear = r2_score(y_test_pred_linear , y_test)
print(f"Linear Regression - Test R^2: {test_r2_linear}")





residuals = y_test - y_test_pred_linear

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()


plt.figure(figsize=(8, 6))
coefficients.plot(kind='barh', x='Tag', y='Coefficient', legend=False, color='skyblue', edgecolor='k')
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Coefficient Value')
plt.ylabel('Predictor')
plt.title('Feature Importance (Regression Coefficients)')
plt.gca().invert_yaxis()
plt.show()

# Mean prediction
y_train_mean = y_train.mean()

# Generate predictions using the mean
y_test_pred_mean = np.full_like(y_test, y_train_mean)

# Calculate RMSE and R^2 for the mean prediction
mean_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_mean))


# Print results
print(f"Mean Prediction - Test RMSE: {mean_rmse}")




# Q8
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

random_seed = 11058720
np.random.seed(random_seed)

num_data = pd.read_csv('rmpCapstoneNum.csv', header=None)
qual_data = pd.read_csv('rmpCapstoneQual.csv', header=None)
tags_data = pd.read_csv('rmpCapstoneTags.csv', header=None)

num_data.columns = [
    "Average_Rating", "Average_Difficulty", "Num_Ratings", "Received_Pepper",
    "Proportion_Take_Class_Again", "Online_Class_Ratings", "Male", "Female"
]
qual_data.columns = ["Major_Field", "University", "US_State"]
tags_data.columns = [
    "Tough_Grader", "Good_Feedback", "Respected", "Lots_to_Read",
    "Participation_Matters", "Dont_Skip_Class", "Lots_of_Homework",
    "Inspirational", "Pop_Quizzes", "Accessible", "So_Many_Papers",
    "Clear_Grading", "Hilarious", "Test_Heavy", "Graded_by_Few_Things",
    "Amazing_Lectures", "Caring", "Extra_Credit", "Group_Projects", "Lecture_Heavy"
]

data = pd.concat([num_data, tags_data], axis=1)
data = data[data["Num_Ratings"] >= 5]

data = data.dropna()

for tag in tags_data.columns:
    data[tag] = data[tag] / data["Num_Ratings"]

X = data[tags_data.columns]

def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    return vif_data

vif_data = calculate_vif(X)
print("Variance Inflation Factor (VIF):")
print(vif_data)

high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
X = X.drop(columns=high_vif_features)
print(f"Features removed due to high VIF: {high_vif_features}")

y = data["Average_Rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

coefficients = pd.DataFrame({
    "Tag": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Regression Coefficients:")
print(coefficients)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Average Ratings")
plt.ylabel("Predicted Average Ratings")
plt.title("Predicted vs Actual Ratings")
plt.show()

# 9
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(11058720)

tags = pd.read_csv("rmpCapstoneTags.csv", header=None)
num_data = pd.read_csv("rmpCapstoneNum.csv", header=None)

tags.columns = [
    "Tough grader", "Good feedback", "Respected", "Lots to read",
    "Participation matters", "Don’t skip class", "Lots of homework",
    "Inspirational", "Pop quizzes!", "Accessible", "So many papers",
    "Clear grading", "Hilarious", "Test heavy", "Graded by few things",
    "Amazing lectures", "Caring", "Extra credit", "Group projects",
    "Lecture heavy"
]
num_data.columns = [
    "avg_rating", "avg_difficulty", "num_ratings", "pepper",
    "class_again", "online_ratings", "male", "female"
]

data = pd.concat([num_data, tags], axis=1)

data = data[data["num_ratings"] >= 5]

tag_columns = tags.columns
for col in tag_columns:
    data[col] = data[col] / data["num_ratings"]

data = data.dropna()

X = data[tag_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
vif_data = pd.DataFrame()
vif_data["feature"] = tag_columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

high_vif_features = vif_data[vif_data["VIF"] > 5]["feature"]
X = X.drop(columns=high_vif_features)

tag_columns = X.columns

y = data["avg_difficulty"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11058720)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse_regression = np.sqrt(mean_squared_error(y_test, y_pred))
r2_regression = r2_score(y_test, y_pred)


print("Regression Model Results")
print("R²:", r2_regression)
print("RMSE (Regression):", rmse_regression)

coefficients = pd.DataFrame({
    "Tag": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("Regression Coefficients:")
print(coefficients)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Average Ratings")
plt.ylabel("Predicted Average Ratings")
plt.title("Predicted vs Actual Ratings")
plt.show()


# # Q10
#
# Build a classification model that predicts whether a professor receives a “pepper” from all available
# factors (both tags and numerical). Make sure to include model quality metrics such as AU(RO)C and
# also address class imbalance concerns.


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier

# Load datasets
num_data_path = 'rmpCapstoneNum.csv'
qual_data_path = 'rmpCapstoneQual.csv'
tags_data_path = 'rmpCapstoneTags.csv'

num_data = pd.read_csv(num_data_path)
qual_data = pd.read_csv(qual_data_path)
tags_data = pd.read_csv(tags_data_path)

# Rename columns
num_data.columns = [
    "average_rating", "average_difficulty", "num_ratings", "pepper",
    "proportion_retaking", "online_ratings", "male", "female"
]
qual_data.columns = ["major", "university", "state"]
tags_data.columns = [
    "tough_grader", "good_feedback", "respected", "lots_to_read",
    "participation_matters", "dont_skip_class", "lots_of_homework",
    "inspirational", "pop_quizzes", "accessible", "so_many_papers",
    "clear_grading", "hilarious", "test_heavy", "graded_by_few_things",
    "amazing_lectures", "caring", "extra_credit", "group_projects",
    "lecture_heavy"
]

# Ensure alignment: Drop rows where 'pepper' is missing from all datasets
valid_rows_mask = num_data['pepper'].notna()

num_data = num_data[valid_rows_mask].reset_index(drop=True)
qual_data = qual_data[valid_rows_mask].reset_index(drop=True)
tags_data = tags_data[valid_rows_mask].reset_index(drop=True)

# Remove rows where 'num_ratings' < 5
num_data = num_data[num_data['num_ratings'] >= 5].reset_index(drop=True)

# Handle missing values
num_data.fillna(num_data.mean(), inplace=True)
qual_data.fillna(qual_data.mode().iloc[0], inplace=True)
tags_data.fillna(0, inplace=True)

# Merge datasets
merged_data = pd.concat([num_data, qual_data, tags_data], axis=1)

# One-hot encode categorical variables
merged_data = pd.get_dummies(merged_data, columns=['major', 'university', 'state'], drop_first=True)

# Remove rows where 'pepper' is NaN
merged_data = merged_data[merged_data['pepper'].notna()].reset_index(drop=True)

# Separate features and target
y = merged_data['pepper']
X = merged_data.drop(columns=['pepper'])

# Ensure all features in X are numeric
X = X.select_dtypes(include=[np.number])

# Handle missing and infinite values in X
X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
X = X.dropna(axis=1, how='any')  # Drop columns with NaN values


# Compute Variance Inflation Factor (VIF)
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = data.columns
    vif_data['VIF'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

# Remove features with VIF >= 5
while True:
    vif = calculate_vif(X)
    max_vif = vif['VIF'].max()
    if max_vif > 5:
        feature_to_remove = vif.loc[vif['VIF'] == max_vif, 'Feature'].values[0]
        print(f"Removing feature '{feature_to_remove}' with VIF = {max_vif:.2f}")
        X = X.drop(columns=[feature_to_remove])
    else:
        break


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=11931503, stratify=y
)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=11931503)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=11931503, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]
auc_score_rf = roc_auc_score(y_test, y_pred_prob)
class_report_rf = classification_report(y_test, y_pred, output_dict=True)

# Train Logistic Regression model
logistic_model = LogisticRegression(random_state=11931503, class_weight='balanced')
logistic_model.fit(X_train_resampled, y_train_resampled)
y_pred_logistic = logistic_model.predict(X_test)
y_pred_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]
auc_score_logistic = roc_auc_score(y_test, y_pred_prob_logistic)
class_report_logistic = classification_report(y_test, y_pred_logistic, output_dict=True)

# Train baseline model
baseline_model = DummyClassifier(strategy="uniform")
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
y_pred_prob_baseline = baseline_model.predict_proba(X_test)[:, 1]
auc_score_baseline = roc_auc_score(y_test, y_pred_prob_baseline)
class_report_baseline = classification_report(y_test, y_pred_baseline, output_dict=True)

# Compare results
results = {
    "Model": ["Random Guess Baseline", "Random Forest", "Logistic Regression"],
    "AUC Score": [auc_score_baseline, auc_score_rf, auc_score_logistic],
    "Accuracy": [
        class_report_baseline['accuracy'],
        class_report_rf['accuracy'],
        class_report_logistic['accuracy']
    ],
    "Recall": [
        class_report_baseline['1.0']['recall'],
        class_report_rf['1.0']['recall'],
        class_report_logistic['1.0']['recall']
    ],
    "Precision": [
        class_report_baseline['1.0']['precision'],
        class_report_rf['1.0']['precision'],
        class_report_logistic['1.0']['precision']
    ]
}

results_df = pd.DataFrame(results)

# Display the results
results_df


# In this analysis, we aimed to predict whether a professor receives a "pepper" rating based on a variety of numerical, categorical, and tag-based features. We first performed preprocessing by removing rows where the number of ratings was less than 5 to ensure the reliability of the data. We then handled missing values by filling numerical columns with their means, categorical columns with their mode, and tag-based features with 0, assuming no tag was awarded. We removed multicollinearity by calculating the Variance Inflation Factor (VIF) for each feature and iteratively dropped features with VIF ≥ 5 to improve model stability; this process removed proportion_retaking and average_rating. To address the issue of class imbalance in the target variable (pepper), we applied SMOTE (Synthetic Minority Oversampling Technique), which generates synthetic samples of the minority class to balance the dataset, ensuring the models are not biased towards the majority class.
#
# We trained three models: a random forest classifier, logistic regression, and a random guess baseline. The logistic regression model achieved the highest Area Under the Curve (AUC) score of 0.667, followed by the random forest model with an AUC of 0.642, and the baseline at 0.500. The AUC score measures the model's ability to distinguish between classes, with 0.5 indicating random guessing and 1.0 indicating perfect discrimination. The logistic regression model also achieved better overall accuracy (0.621) and recall (0.619), indicating its ability to correctly identify positive cases more frequently compared to the random forest model. These findings suggest that logistic regression is the most effective model for this task, providing a reasonable trade-off between sensitivity and precision in predicting "pepper" ratings.


from sklearn.metrics import roc_curve, roc_auc_score

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_logistic)
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_prob_baseline)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_score_rf:.3f})")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_score_logistic:.3f})")
plt.plot(fpr_baseline, tpr_baseline, label="Majority Class Baseline (AUC = 0.500)")
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid()
plt.show()



#EC
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

np.random.seed(11058720)

num_data = pd.read_csv('rmpCapstoneNum.csv', header=None)
qual_data = pd.read_csv('rmpCapstoneQual.csv', header=None)
tags_data = pd.read_csv('rmpCapstoneTags.csv', header=None)

num_data.columns = ['avg_rating', 'avg_difficulty', 'num_ratings', 'received_pepper',
                    'would_take_again', 'online_ratings', 'is_male', 'is_female']

num_data = num_data.dropna()

median_rating = num_data['avg_rating'].median()

low_ratings = num_data[num_data['avg_rating'] < median_rating]['num_ratings']
high_ratings = num_data[num_data['avg_rating'] >= median_rating]['num_ratings']
#print(num_data[num_data['avg_rating'] >= median_rating])
stat, p_value = mannwhitneyu(low_ratings, high_ratings, alternative='greater')

plt.figure(figsize=(10, 6))
plt.boxplot([low_ratings, high_ratings], labels=['Low Avg Ratings', 'High Avg Ratings'])
plt.title('Number of Ratings by Avg Rating Groups')
plt.ylabel('Number of Ratings')
plt.grid(axis='y')
plt.show()

print("Median of Average Ratings:", median_rating)
print("Mann-Whitney U Test Statistic:", stat)
print("P-value:", p_value)

if p_value < 0.005:
    print("There is a statistically significant difference in the number of ratings between low and high avg rating groups (alpha=0.005).")
else:
    print("There is no statistically significant difference in the number of ratings between low and high avg rating groups (alpha=0.005).")

plt.figure(figsize=(12, 6))
plt.hist(low_ratings, bins=30, alpha=0.7, label='Low Avg Ratings', color='blue')
plt.hist(high_ratings, bins=30, alpha=0.7, label='High Avg Ratings', color='orange')
plt.title('Distribution of Number of Ratings by Avg Rating Groups')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.show()