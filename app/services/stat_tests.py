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
from statsmodels.stats.anova import anova_lm
from scipy.stats import f_oneway, chi2_contingency



def handle_request(df, stat_config):
    test_type = stat_config['test']
    values = stat_config['values']
    confidence_level = float(values.get('confidence_level', 0.95))

    if test_type in ['OneWayANOVA', 'KruskalWallisTest'] and values.get('group_representation') == 'Groups in separate columns':
        df = transform_separate_columns_to_one(df, values)
        
    numeric_columns = [values.get('variable'), values.get('variable1'), values.get('variable2'), values.get('numeric_variable'), values.get('dependent_variable')]
    numeric_columns = [col for col in numeric_columns if col in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if test_type != 'IndependentTwoSampleTTest':
        df = df.dropna()

    if test_type == 'OneSampleTTest':
        result = one_sample_ttest(df, values, confidence_level)
    elif test_type in ['IndependentTwoSampleTTest', 'PairedSampleTTest', 'MannWhitneyUTest']:
        result = two_sample_test(df, values, confidence_level, test_type)
    elif test_type == 'KruskalWallisTest':
        result = kruskal_wallis_test(df, values, confidence_level)
    elif test_type == 'OneWayANOVA':
        result = one_way_anova(df, values, confidence_level)
    elif test_type == 'TwoWayANOVA':
        result = two_way_anova(df, values, confidence_level)
    elif test_type == 'Correlation':
        result = correlation_test(df, values)
    elif test_type == 'FTest':
        result = f_test(df, values, confidence_level)
    elif test_type == 'ChiSquareTest':
        print(df)
        result = chi_square_test(df, values, confidence_level)

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

    summary_stats = calculate_summary_statistics(df[column].dropna(), confidence_level)

    return {
        'summary_statistics': { column: summary_stats },
        'test_results': {
            'test': 'OneSampleTTest',
            't_stat': t_stat,
            'p_value': p_value,
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
        group_names[0]: calculate_summary_statistics(group1, confidence_level),
        group_names[1]: calculate_summary_statistics(group2, confidence_level)
    }

    return {
        'summary_statistics': summary_statistics,
        'test_results': result
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
    anova_type = int(values['anova_type'])  # Retrieve ANOVA type from values

    # Create a copy of the DataFrame with just the relevant columns
    df_relevant = df[[numeric_variable, categorical_variable]].copy()

    # Create a mapping for renaming columns
    original_columns = {numeric_variable: 'numeric_variable', categorical_variable: 'categorical_variable'}
    df_relevant = df_relevant.rename(columns=original_columns)

    # Use the temporary names in the formula
    formula = 'numeric_variable ~ C(categorical_variable)'
    model = ols(formula, data=df_relevant).fit()
    anova_table = sm.stats.anova_lm(model, typ=anova_type)

    summary_stats = df_relevant.groupby('categorical_variable')['numeric_variable'].agg(['count', 'sum', 'mean', 'var']).reset_index()
    summary_stats = summary_stats.rename(columns={'var': 'variance', 'categorical_variable': 'group'})

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

    df_between = anova_table['df'].iloc[0]
    df_within = anova_table['df'].iloc[1]
    f_critical = f.ppf(1 - (1 - confidence_level), df_between, df_within)

    # Post hoc analysis using Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df_relevant['numeric_variable'].dropna(), groups=df_relevant['categorical_variable'].dropna(), alpha=1 - confidence_level)
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
        'SS': anova_table['sum_sq'].iloc[0],
        'df': anova_table['df'].iloc[0],
        'MS': anova_table['mean_sq'].iloc[0],
        'F': anova_table['F'].iloc[0],
        'p-value': anova_table['PR(>F)'].iloc[0]
    }

    within_group_summary = {
        'SS': anova_table['sum_sq'].iloc[1],
        'df': anova_table['df'].iloc[1],
        'MS': anova_table['mean_sq'].iloc[1],
        'F': None,
        'p-value': None
    }

    result = {
        'summary_statistics': formatted_summary_stats,

        'f_critical': f_critical,
        'test_results': {
            'test': 'OneWayANOVA',
            'null_hypothesis': f"There is no effect of {categorical_variable} on {numeric_variable}.",
            'alternative_hypothesis': f"There is an effect of {categorical_variable} on {numeric_variable}.",
            'p_value': anova_table['PR(>F)'].iloc[0],
            'reject_null': "Reject the null hypothesis" if anova_table['PR(>F)'].iloc[0] < (1 - confidence_level) else "Fail to reject the null hypothesis",
            'decision': "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if anova_table['PR(>F)'].iloc[0] < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100),
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




def two_way_anova(df, values, confidence_level):
    dependent_variable = values['dependent_variable']
    factor1 = values['factor1']
    factor2 = values['factor2']
    anova_type = int(values['anova_type'])  # Retrieve ANOVA type from values
    
    # Create a copy of the DataFrame with just the relevant columns
    df_relevant = df[[dependent_variable, factor1, factor2]].copy()

    # Create a mapping for renaming columns
    original_columns = {dependent_variable: 'dependent_variable', factor1: 'factor1', factor2: 'factor2'}
    df_relevant = df_relevant.rename(columns=original_columns)

    # Use the temporary names in the formula
    formula = 'dependent_variable ~ C(factor1) + C(factor2) + C(factor1):C(factor2)'
    model = ols(formula, data=df_relevant).fit()
    anova_table = anova_lm(model, typ=anova_type)

    summary_stats = df_relevant.groupby(['factor1', 'factor2'])['dependent_variable'].agg(['count', 'sum', 'mean', 'var']).reset_index()
    summary_stats = summary_stats.rename(columns={'var': 'variance', 'factor1': 'Factor 1', 'factor2': 'Factor 2'})

    # Convert summary statistics to the expected format
    formatted_summary_stats = {}
    for _, row in summary_stats.iterrows():
        group_name = f"{row['Factor 1']} - {row['Factor 2']}"
        formatted_summary_stats[group_name] = {
            'count': int(row['count']),
            'sum': float(row['sum']),
            'mean': float(row['mean']),
            'variance': float(row['variance'])
        }

    anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

    df_between = anova_table['df'].iloc[0]
    df_within = anova_table['df'].iloc[1]
    f_critical = f.ppf(1 - (1 - confidence_level), df_between, df_within)

    # Prepare the formatted ANOVA table
    formatted_anova_table = [
        {
            'source': 'Factor 1',
            'SS': anova_table['sum_sq']['C(factor1)'],
            'df': anova_table['df']['C(factor1)'],
            'MS': anova_table['mean_sq']['C(factor1)'],
            'F': anova_table['F']['C(factor1)'],
            'p': anova_table['PR(>F)']['C(factor1)']
        },
        {
            'source': 'Factor 2',
            'SS': anova_table['sum_sq']['C(factor2)'],
            'df': anova_table['df']['C(factor2)'],
            'MS': anova_table['mean_sq']['C(factor2)'],
            'F': anova_table['F']['C(factor2)'],
            'p': anova_table['PR(>F)']['C(factor2)']
        },
        {
            'source': 'Interaction',
            'SS': anova_table['sum_sq']['C(factor1):C(factor2)'],
            'df': anova_table['df']['C(factor1):C(factor2)'],
            'MS': anova_table['mean_sq']['C(factor1):C(factor2)'],
            'F': anova_table['F']['C(factor1):C(factor2)'],
            'p': anova_table['PR(>F)']['C(factor1):C(factor2)']
        },
        {
            'source': 'Within Group (Error)',
            'SS': anova_table['sum_sq']['Residual'],
            'df': anova_table['df']['Residual'],
            'MS': anova_table['mean_sq']['Residual'],
            'F': None,
            'p': None
        }
    ]

    result = {
        'summary_statistics': formatted_summary_stats,
        'anova_table': formatted_anova_table,
        'f_critical': f_critical,
        'test_results': {
            'test': 'TwoWayANOVA',
            'null_hypothesis': f"There is no interaction effect between {factor1} and {factor2} on {dependent_variable}.",
            'alternative_hypothesis': f"There is an interaction effect between {factor1} and {factor2} on {dependent_variable}.",
            'p_value': anova_table['PR(>F)'].iloc[0],
            'reject_null': "Reject the null hypothesis" if anova_table['PR(>F)'].iloc[0] < (1 - confidence_level) else "Fail to reject the null hypothesis",
            'decision': "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if anova_table['PR(>F)'].iloc[0] < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100),
        }
    }

    return result






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

def calculate_summary_statistics(data, confidence_level=0.95):
    count = int(len(data))
    mean = float(np.mean(data))
    variance = float(np.var(data, ddof=1))
    ci = calculate_confidence_interval(data, confidence_level)
    ci_lower_label = f'Lower {round((1 - confidence_level) / 2 * 100, 4)}%'
    ci_upper_label = f'Upper {round((1 + confidence_level) / 2 * 100,4)}%'
    
    return {
        'count': count,
        'sum': float(np.sum(data)),
        'mean': mean,
        'variance': variance,
        ci_lower_label: ci[0],
        ci_upper_label: ci[1]
    }


def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return [mean - h, mean + h]







def f_test(df, values, confidence_level):
    comparison_type = values['comparison_type']
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
            null_hypothesis = f"The variance of {column1} is equal to the variance of {column2}"
            alternative_hypothesis = f"The variance of {column1} is not equal to the variance of {column2}"
        elif comparison == '<':
            null_hypothesis = f"The variance of {column1} is >= the variance of {column2}"
            alternative_hypothesis = f"The variance of {column1} is < the variance of {column2}"
        elif comparison == '>':
            null_hypothesis = f"The variance of {column1} is <= the variance of {column2}"
            alternative_hypothesis = f"The variance of {column1} is > the variance of {column2}"
    else:
        numeric_variable = values['numeric_variable']
        categorical_variable = values['categorical_variable']
        categories = list(values['categories'])
        group1 = df[df[categorical_variable] == categories[0]][numeric_variable].dropna()
        group2 = df[df[categorical_variable] == categories[1]][numeric_variable].dropna()
        group_names = categories
        if comparison == '=':
            null_hypothesis = f"The variance of {numeric_variable} is equal across {categorical_variable} categories {categories[0]} and {categories[1]}"
            alternative_hypothesis = f"The variance of {numeric_variable} is not equal across {categorical_variable} categories {categories[0]} and {categories[1]}"
        elif comparison == '<':
            null_hypothesis = f"The variance of {numeric_variable} for {categories[0]} group is >= the variance for {categories[1]} group"
            alternative_hypothesis = f"The variance of {numeric_variable} for {categories[0]} group is < the variance for {categories[1]} group"
        elif comparison == '>':
            null_hypothesis = f"The variance of {numeric_variable} for {categories[0]} group is <= the variance for {categories[1]} group"
            alternative_hypothesis = f"The variance of {numeric_variable} for {categories[0]} group is > the variance for {categories[1]} group"

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    f_stat = var1 / var2
    dfn = len(group1) - 1
    dfd = len(group2) - 1

    if comparison == '=':
        p_value = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))
    elif comparison == '<':
        p_value = stats.f.cdf(f_stat, dfn, dfd)
    else:  # comparison == '>'
        p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)

    decision = "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100)

    result = {
        'test': 'FTest',
        'f_stat': f_stat,
        'p_value': p_value,
        'null_hypothesis': null_hypothesis,
        'alternative_hypothesis': alternative_hypothesis,
        'decision': decision
    }

    summary_statistics = {
        group_names[0]: calculate_summary_statistics(group1, confidence_level),
        group_names[1]: calculate_summary_statistics(group2, confidence_level)
    }

    return {
        'summary_statistics': summary_statistics,
        'test_results': result
    }






def chi_square_test(df, values, confidence_level):
    categorical_variable1 = values['categorical_variable1']
    categorical_variable2 = values['categorical_variable2']

    # Drop rows with NaN values in the specified columns
    df = df.dropna(subset=[categorical_variable1, categorical_variable2])

    contingency_table = pd.crosstab(df[categorical_variable1], df[categorical_variable2])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

    decision = "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100)

    null_hypothesis = f"There is no association between {categorical_variable1} and {categorical_variable2}."
    alternative_hypothesis = f"There is an association between {categorical_variable1} and {categorical_variable2}."



    return {
        'test_results': {
            'test': 'ChiSquareTest',
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'dof': dof,
            'null_hypothesis': null_hypothesis,
            'alternative_hypothesis': alternative_hypothesis,
            'decision': decision
        }
    }

