from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel
from app.routes.common_router_functions import Data, convert_to_df
from scipy.stats import kstest, norm, pareto, gamma, uniform, expon, beta
import numpy as np
from matplotlib.pyplot import plot as plt
from app.services.stat_tests import *

router = APIRouter()

@router.post("/stat")
async def stat_analysis(data: Data, statConfig: dict):
    df = convert_to_df(data)
    result = handle_request(df, statConfig)
    print(result)
    return result

@router.post("/goodFit")
async def good_fit_test(data: Data, statConfig: dict):
    df = convert_to_df(data)
    result = handle_good_fit_request(df, statConfig)
    print(result)
    return result

def handle_good_fit_request(df, stat_config):
    test_type = stat_config['test']
    values = stat_config['values']
    confidence_level = float(values.get('confidence_level', 0.95))

    if test_type == 'KolmogorovSmirnovTest':
        result = kolmogorov_smirnov_test(df, values, confidence_level)
    
    return handle_nan_in_dict(result)

def kolmogorov_smirnov_test(df, values, confidence_level):
    column = values['variable']
    distribution = values['distribution']
    data = df[column].dropna()
    
    if distribution == 'normal':
        dist = norm
        params = dist.fit(data)
        param_names = ['mean', 'std']
    elif distribution == 'pareto':
        dist = pareto
        params = dist.fit(data)
        param_names = ['shape', 'loc', 'scale']
    elif distribution == 'gamma':
        dist = gamma
        params = dist.fit(data)
        param_names = ['shape', 'loc', 'scale']
    elif distribution == 'uniform':
        dist = uniform
        params = dist.fit(data)
        param_names = ['loc', 'scale']
    elif distribution == 'exponential':
        dist = expon
        params = dist.fit(data)
        param_names = ['loc', 'scale']
    elif distribution == 'beta':
        dist = beta
        params = dist.fit(data)
        param_names = ['a', 'b', 'loc', 'scale']
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    stat, p_value = kstest(data, dist.cdf, args=params)

    decision = "Reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100) if p_value < (1 - confidence_level) else "Fail to reject the null hypothesis at the {0:.1f}% confidence level".format(confidence_level * 100)

    # Generate the histogram and distribution data
    histogram_data, distribution_data = generate_distribution_data(data, dist, params)
    
    # Calculate summary statistics
    summary_statistics = calculate_summary_statistics(data)

    return {
        'test_results': {
            'test': 'KolmogorovSmirnovTest',
            'distribution': distribution,
            'statistic': stat,
            'p_value': p_value,
            'decision': decision
        },
        'distribution_parameters': {
            name: param for name, param in zip(param_names, params)
        },
        'histogram_data': histogram_data,
        'distribution_data': distribution_data,
    }



def generate_distribution_data(data, dist, params):
    # Generate histogram data
    hist, bin_edges = np.histogram(data, bins=30, density=True)
    histogram_data = {
        'bin_edges': bin_edges.tolist(),
        'hist_values': hist.tolist()
    }
    
    # Generate fitted distribution data
    x = np.linspace(min(data), max(data), 100)
    p = dist.pdf(x, *params)
    distribution_data = {
        'x': x.tolist(),
        'y': p.tolist()
    }
    
    return histogram_data, distribution_data
