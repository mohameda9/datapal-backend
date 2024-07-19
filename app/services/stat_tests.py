from app.services import processing_service as funs
from app.routes.common_router_functions import Data, convert_to_df
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, mannwhitneyu, kruskal, f, norm
from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import pearsonr, spearmanr



def handle_request(df, stat_config):
    test_type = stat_config['test']
    values = stat_config['values']
    confidence_level = float(values.get('confidence_level', 0.95))

    if test_type in ['OneWayANOVA', 'KruskalWallisTest'] and values.get('group_representation') == 'Groups in separate columns':
        df = transform_separate_columns_to_one(df, values)
        
    # Identify numerical columns and convert them to numeric types
    numeric_columns = [values.get('variable1'), values.get('variable2'), values.get('numeric_variable')]
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    print(numeric_columns)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if test_type == 'OneSampleTTest':
        result = one_sample_ttest(df, values, confidence_level)
    elif test_type in ['IndependentTwoSampleTTest', 'PairedSampleTTest', 'MannWhitneyUTest']:
        result = two_sample_test(df, values, confidence_level, test_type)
    elif test_type == 'KruskalWallisTest':
        result = kruskal_wallis_test(df, values, confidence_level)
    elif test_type == 'OneWayANOVA':
        result = one_way_anova(df, values, confidence_level)

    elif test_type == 'Correlation':
        result = correlation_test(df, values)
    
    return handle_nan_in_dict(result)

def transform_separate_columns_to_one(df, values):
    group_columns = values['group_columns']
    melted_df = df.melt(value_vars=group_columns, var_name='Group', value_name='Value')
    values['numeric_variable'] = 'Value'
    values['categorical_variable'] = 'Group'
    return melted_df

def handle_nan_in_dict(data):
    if isinstance(data, dict):
        return {k: handle_nan_in_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [handle_nan_in_dict(v) for v in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    else:
        return data

def one_sample_ttest(df, values, confidence_level):
    column = values['variable']
    population_mean = float(values['population_mean'])
    comparison = values['comparison']

    if comparison == '=':
        alternative = 'two-sided'
    elif comparison == '<':
        alternative = 'less'
    elif comparison == '>':
        alternative = 'greater'

    t_stat, p_value = ttest_1samp(df[column].dropna(), population_mean, alternative=alternative)
    ci = calculate_confidence_interval(df[column].dropna(), confidence_level)
    decision = "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100)

    if (comparison == '='):
        null_hypothesis = f"The mean of {column} is equal to {population_mean}"
        alternative_hypothesis = f"The mean of {column} is not equal to {population_mean}"
    elif (comparison == '<'):
        null_hypothesis = f"The mean of {column} is >= {population_mean}"
        alternative_hypothesis = f"The mean of {column} is < {population_mean}"
    elif (comparison == '>'):
        null_hypothesis = f"The mean of {column} is <= {population_mean}"
        alternative_hypothesis = f"The mean of {column} is > {population_mean}"

    summary_stats = calculate_summary_statistics(df[column].dropna())

    return {
        'summary_statistics': { column: summary_stats },
        'test_results': {
            'test': 'OneSampleTTest',
            't_stat': t_stat,
            'p_value': p_value,
            'confidence_intervals': { column: ci },
            'null_hypothesis': null_hypothesis,
            'alternative_hypothesis': alternative_hypothesis,
            'decision': decision
        }
    }

def two_sample_test(df, values, confidence_level, test_type):
    comparison_type = values['comparison_type']
    equal_var = values.get('equal_variances', True)
    comparison = values['comparison']

    if comparison == '=':
        alternative = 'two-sided'
    elif comparison == '<':
        alternative = 'less'
    elif comparison == '>':
        alternative = 'greater'

    if comparison_type == 'Compare two columns':
        column1 = values['variable1']
        column2 = values['variable2']
        group1, group2 = df[column1].dropna(), df[column2].dropna()
        group_names = [column1, column2]
        if comparison == '=':
            null_hypothesis = f"The mean of {column1} is equal to the mean of {column2}"
            alternative_hypothesis = f"The mean of {column1} is not equal to the mean of {column2}"
        elif comparison == '<':
            null_hypothesis = f"The mean of {column1} is >= {column2}"
            alternative_hypothesis = f"The mean of {column1} is < {column2}"
        elif comparison == '>':
            null_hypothesis = f"The mean of {column1} is <= {column2}"
            alternative_hypothesis = f"The mean of {column1} is > {column2}"
    else:
        numeric_variable = values['numeric_variable']
        categorical_variable = values['categorical_variable']
        categories = list(values['categories'])
        group1 = df[df[categorical_variable] == categories[0]][numeric_variable].dropna()
        group2 = df[df[categorical_variable] == categories[1]][numeric_variable].dropna()
        group_names = categories
        if comparison == '=':
            null_hypothesis = f"The mean of {numeric_variable} is equal across {categorical_variable} categories {categories[0]} and {categories[1]}"
            alternative_hypothesis = f"The mean of {numeric_variable} is not equal across {categorical_variable} categories {categories[0]} and {categories[1]}"
        elif comparison == '<':
            null_hypothesis = f"The mean of {numeric_variable}  for {categories[0]} group >= {categories[1]} group"
            alternative_hypothesis = f"The mean of {numeric_variable} is less than or equal across {categorical_variable} categories {categories[0]} and {categories[1]}"
        elif comparison == '>':
            null_hypothesis = f"The mean of {numeric_variable}  for {categories[0]} group <= {categories[1]} group"
            alternative_hypothesis = f"The mean of {numeric_variable}  for {categories[0]} group > {categories[1]} group"

    if test_type == 'IndependentTwoSampleTTest':
        t_stat, p_value = ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
    elif test_type == 'PairedSampleTTest':
        t_stat, p_value = ttest_rel(group1, group2, alternative=alternative)
    else:  # MannWhitneyUTest
        t_stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)

    reject_null = reject_null_hypothesis(p_value, confidence_level)
    decision = "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100)

    result = {
        'test': test_type,
        't_stat': t_stat,
        'p_value': p_value,
        'null_hypothesis': null_hypothesis,
        'alternative_hypothesis': alternative_hypothesis,
        'decision': decision
    }

    summary_statistics = {
        group_names[0]: calculate_summary_statistics(group1),
        group_names[1]: calculate_summary_statistics(group2)
    }

    return {
        'summary_statistics': summary_statistics,
        'test_results': result
    }

def calculate_summary_statistics(data):
    return {
        'count': int(len(data)),
        'sum': float(np.sum(data)),
        'mean': float(np.mean(data)),
        'variance': float(np.var(data, ddof=1))
    }

def kruskal_wallis_test(df, values, confidence_level):
    group_representation = values.get('group_representation')

    if group_representation == 'Groups in one column':
        numeric_variable = values.get('numeric_variable')
        categorical_variable = values.get('categorical_variable')
        if not numeric_variable or not categorical_variable:
            raise ValueError("Both numeric and categorical variables must be selected.")
        model_data = df[[numeric_variable, categorical_variable]].dropna()
        model_data[numeric_variable] = pd.to_numeric(model_data[numeric_variable], errors='coerce')
        model_data = model_data.dropna()
        kruskal_result = stats.kruskal(*[model_data[model_data[categorical_variable] == category][numeric_variable] for category in model_data[categorical_variable].unique()])
        group_col = categorical_variable
        value_col = numeric_variable
    else:
        group_columns = values.get('group_columns')
        if not group_columns:
            raise ValueError("Group columns must be selected.")
        
        model_data = df.dropna()       
        model_data['Value'] = pd.to_numeric(model_data['Value'], errors='coerce')
        model_data = model_data.dropna()
        kruskal_result = stats.kruskal(*[model_data[model_data['Group'] == group]['Value'] for group in model_data['Group'].unique()])
        group_col = 'Group'
        value_col = 'Value'

    # Calculate sum of ranks for each group
    model_data['rank'] = model_data[value_col].rank()
    summary_statistics = model_data.groupby(group_col).agg(
        count=(value_col, 'count'), 
        sum_of_ranks=('rank', 'sum'),
        sum=(value_col, 'sum'),
        mean=(value_col, 'mean'),
        variance=(value_col, 'var')
    )

    # Post hoc analysis (if applicable)
    post_hoc_results = dunn_posthoc(model_data[value_col], model_data[group_col])
    print(post_hoc_results)
    formatted_post_hoc_results = []
    for comparison in post_hoc_results:
        contrast = comparison['contrast']
        z_value = comparison['z_value']
        p_value = comparison['p_value']
        decision = comparison['decision']
        formatted_post_hoc_results.append({
            'contrast': contrast,
            'z_value': z_value,
            'p_value': p_value,
            'decision': decision
        })

    formatted_summary_stats = {}
    for name, row in summary_statistics.iterrows():
        formatted_summary_stats[name] = {
            'count': int(row['count']),
            'sum_of_ranks': float(row['sum_of_ranks']),
            'mean_of_ranks':(row['sum_of_ranks']/row['count']),
            'sum': float(row['sum']),
            'mean': float(row['mean']),
            'variance': float(row['variance'])
        }

    return {
        'summary_statistics': formatted_summary_stats,
        'test_results': {
            'test': 'KruskalWallisTest',
            'H_statistic': kruskal_result.statistic,
            'p_value': kruskal_result.pvalue,
            'decision': 'Reject the null hypothesis at the {}% confidence level'.format(confidence_level * 100) if kruskal_result.pvalue < (1 - confidence_level) else 'Fail to reject the null hypothesis at the {}% confidence level'.format(confidence_level * 100)
        },
        'post_hoc': {
            'test': 'Dunn-Bonferroni',
            'results': formatted_post_hoc_results
        }
    }









def one_way_anova(df, values, confidence_level):
    numeric_variable = values['numeric_variable']
    categorical_variable = values['categorical_variable']

    formula = f'{numeric_variable} ~ C({categorical_variable})'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    summary_stats = df.groupby(categorical_variable)[numeric_variable].agg(['count', 'sum', 'mean', 'var']).reset_index()
    summary_stats = summary_stats.rename(columns={'var': 'variance', categorical_variable: 'group'})

    # Convert summary statistics to the expected format
    formatted_summary_stats = {}
    for _, row in summary_stats.iterrows():
        group_name = row['group']
        formatted_summary_stats[group_name] = {
            'count': int(row['count']),
            'sum': float(row['sum']),
            'mean': float(row['mean']),
            'variance': float(row['variance'])
        }

    anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

    df_between = anova_table['df'][0]
    df_within = anova_table['df'][1]
    f_critical = f.ppf(1 - (1 - confidence_level), df_between, df_within)

    # Post hoc analysis using Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df[numeric_variable].dropna(), groups=df[categorical_variable].dropna(), alpha=1 - confidence_level)
    tukey_summary = tukey.summary().data[1:]
    post_hoc_results = []
    for row in tukey_summary:
        contrast = f"{row[0]} - {row[1]}"
        diff = row[2]
        p_value = row[3]

        decision = "Significantly different" if p_value < (1 - confidence_level) else "Not significantly different"
        post_hoc_results.append({
            'contrast': contrast,
            'difference': diff,
            'p_value': p_value,
            'decision': decision
        })

    between_group_summary = {
        'SS': anova_table['sum_sq'][0],
        'df': anova_table['df'][0],
        'MS': anova_table['mean_sq'][0],
        'F': anova_table['F'][0],
        'p-value': anova_table['PR(>F)'][0]
    }

    within_group_summary = {
        'SS': anova_table['sum_sq'][1],
        'df': anova_table['df'][1],
        'MS': anova_table['mean_sq'][1],
        'F': None,
        'p-value': None
    }

    result = {
        'summary_statistics': formatted_summary_stats,
        'anova_table': {
            'SS': anova_table['sum_sq'].tolist(),
            'df': anova_table['df'].tolist(),
            'MS': anova_table['mean_sq'].tolist(),
            'F': anova_table['F'].tolist(),
            'p-value': anova_table['PR(>F)'].tolist()
        },
        'f_critical': f_critical,
        'test_results': {
            'test': 'OneWayANOVA',
            'null_hypothesis': f"There is no effect of {categorical_variable} on {numeric_variable}.",
            'alternative_hypothesis': f"There is an effect of {categorical_variable} on {numeric_variable}.",
            'p_value': anova_table['PR(>F)'][0],
            'reject_null': "Reject the null hypothesis" if anova_table['PR(>F)'][0] < (1 - confidence_level) else "Fail to reject the null hypothesis",
            'decision': "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if anova_table['PR(>F)'][0] < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100),
        },
        'group_summary': {
            'between_group': between_group_summary,
            'within_group': within_group_summary
        },
        'post_hoc': {
            'test': 'TukeyHSD',
            'results': post_hoc_results
        }
    }

    return result




def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return [mean - h, mean + h]

def reject_null_hypothesis(p_value, confidence_level):
    return "Reject the null hypothesis" if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level"

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

def dunn_posthoc(data, groups):
    """
    Dunn's test for multiple comparisons post hoc analysis. This version calculates Z-values
    and includes only raw p-values in the output.
    """
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({'data': data, 'groups': groups})

    # Rank all data points together
    df['ranked'] = rankdata(df['data'])

    # Calculate mean rank for each group
    group_ranks = df.groupby('groups')['ranked'].mean()

    # Retrieve unique groups and prepare for comparisons
    unique_groups = df['groups'].unique()
    n_groups = len(unique_groups)
    total_n = len(df)
    comparisons = []

    # Conduct pairwise comparisons
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            R1 = group_ranks[unique_groups[i]]
            R2 = group_ranks[unique_groups[j]]
            n1 = df[df['groups'] == unique_groups[i]].shape[0]
            n2 = df[df['groups'] == unique_groups[j]].shape[0]

            # Calculate the Z value
            SE = np.sqrt(((total_n * (total_n + 1)) / 12) * ((1 / n1) + (1 / n2)))
            Z = (R1 - R2) / SE

            # Compute raw p-value
            p_value = 2 * (1 - norm.cdf(abs(Z)))  # two-tailed test

            # Store the comparison results
            decision = "Significantly different" if p_value < 0.05 else "Not significantly different"
            comparisons.append({
                'contrast': f"{unique_groups[i]} - {unique_groups[j]}",
                'z_value': Z,
                'p_value': p_value,
                'decision': decision
            })

    return comparisons


def correlation_test(df, values):
    columns = values['columns']
    corr_type = values.get('correlation_type', 'pearson')
    nan_handling = values.get('nan_handling', 'drop_rows')

    if nan_handling == 'drop_rows':
        df = df.dropna(subset=columns)
    elif nan_handling == 'pairwise':
        pass  # No need to drop rows globally; handled in correlation calculation

    summary_stats = {col: calculate_summary_statistics(pd.to_numeric(df[col], errors='coerce').dropna()) for col in columns}
    correlation_matrix, p_value_matrix = calculate_correlation(df, columns, corr_type, nan_handling)

    return {
        'summary_statistics': summary_stats,
        'correlation_matrix': correlation_matrix,
        'p_value_matrix': p_value_matrix
    }

def calculate_correlation(df, columns, corr_type, nan_handling):
    correlation_matrix = {}
    p_value_matrix = {}

    for col1 in columns:
        correlation_matrix[col1] = {}
        p_value_matrix[col1] = {}
        for col2 in columns:
            if col1 == col2:
                correlation_matrix[col1][col2] = 1.0
                p_value_matrix[col1][col2] = 0.0
            else:
                if nan_handling == 'pairwise':
                    pairwise_df = df[[col1, col2]].dropna()
                else:
                    pairwise_df = df

                if corr_type == 'pearson':
                    corr, p_value = stats.pearsonr(pd.to_numeric(pairwise_df[col1], errors='coerce').dropna(), 
                                                   pd.to_numeric(pairwise_df[col2], errors='coerce').dropna())
                else:
                    corr, p_value = stats.spearmanr(pd.to_numeric(pairwise_df[col1], errors='coerce').dropna(), 
                                                    pd.to_numeric(pairwise_df[col2], errors='coerce').dropna())

                correlation_matrix[col1][col2] = corr
                p_value_matrix[col1][col2] = p_value

    return correlation_matrix, p_value_matrix

def calculate_summary_statistics(data):
    return {
        'count': int(len(data)),
        'sum': float(np.sum(data)),
        'mean': float(np.mean(data)),
        'variance': float(np.var(data, ddof=1))
    }
