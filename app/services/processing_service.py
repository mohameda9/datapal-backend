import pandas as pd
import re, json
from collections import deque
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, FunctionTransformer, PowerTransformer, normalize

def shunting_yard(expression):
    # Define operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    associativity = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    
    output = []
    operators = deque()
    
    # Tokenize the input expression
    tokens = re.findall(r'\s*([\+\-\*/\^\(\)]|\d*\.?\d+|df\.\w+)\s*', expression)
    
    for token in tokens:
        if re.match(r'\d*\.?\d+|df\.\w+', token):
            output.append(token)
        elif token in precedence:
            while (operators and operators[-1] != '(' and
                   (associativity[token] == 'L' and precedence[token] <= precedence[operators[-1]]) or
                   (associativity[token] == 'R' and precedence[token] < precedence[operators[-1]])):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
    
    while operators:
        output.append(operators.pop())
    
    return output

def evaluate_rpn(rpn, df):
    stack = deque()
    
    for token in rpn:
        if re.match(r'\d*\.?\d+', token):
            stack.append(pd.Series([float(token)] * len(df)))
        elif re.match(r'df\.\w+', token):
            column_name = re.findall(r'df\.(\w+)', token)[0]
            stack.append(df[column_name])
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
            elif token == '^':
                stack.append(a ** b)
    
    return stack.pop()

def transform_and_execute(expression, df):
    rpn = shunting_yard(expression)
    result = evaluate_rpn(rpn, df)
    return result

# Sample DataFrame
data = {'age': [25, 30, 35], 'Income': [50000, 60000, 70000]}
df = pd.DataFrame(data)

# Example expression
import pandas as pd
import re
from collections import deque

def shunting_yard(expression):
    # Define operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    associativity = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    
    output = []
    operators = deque()
    
    # Tokenize the input expression
    tokens = re.findall(r'\s*([\+\-\*/\^\(\)]|\d*\.?\d+|df\.\w+)\s*', expression)
    
    for token in tokens:
        if re.match(r'\d*\.?\d+|df\.\w+', token):
            output.append(token)
        elif token in precedence:
            while (operators and operators[-1] != '(' and
                   ((associativity[token] == 'L' and precedence[token] <= precedence[operators[-1]]) or
                   (associativity[token] == 'R' and precedence[token] < precedence[operators[-1]]))):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
    
    while operators:
        output.append(operators.pop())
    
    return output

def evaluate_rpn(rpn, df):
    stack = deque()
    
    for token in rpn:
        if re.match(r'\d*\.?\d+', token):
            stack.append(pd.Series([float(token)] * len(df)))
        elif re.match(r'df\.\w+', token):
            column_name = re.findall(r'df\.(\w+)', token)[0]
            stack.append(df[column_name])
        else:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(a / b)
            elif token == '^':
                stack.append(a ** b)
    
    return stack.pop()

def transform_and_execute(expression, df):
    rpn = shunting_yard(expression)
    result = evaluate_rpn(rpn, df)
    return result




def one_hot_encoding(df, column_name, column_defs=None):
    """
    One-hot encode categorical data in a dataframe based on the column defs.
    Example:
    column_defs = {
        group1_name: [values],
        group2_name: [values],
        group3_name: [values]
    }

    returns a df where column name is one-hot encoded based on the column_defs
    """
    df_copy = df.copy()
    unique_values = df_copy[column_name].unique().tolist()
    if column_defs is None:
        for value in unique_values:
            df_copy[f"{column_name}: {value}"] = df_copy[column_name].apply(lambda x: 1 if x == value else 0)
    else:  # probably useless
        all_values_selected = []
        print(unique_values)

        # Iterate over the column definitions to create one-hot encoded columns
        for group_name, values in column_defs.items():
            # Create a new column for each group
            df_copy[group_name] = df_copy[column_name].apply(lambda x: 1 if x in values else 0)
            all_values_selected += values
        print(all_values_selected)

        remaining_values = list(set(unique_values) - set(all_values_selected))
        print(remaining_values)
        if len(remaining_values) > 0:
            df_copy["other_" + column_name] = df_copy[column_name].apply(lambda x: 1 if x in remaining_values else 0)

    return df_copy




def scaleColumn(train_df, test_df, column_name, method, fit_on_train=True, **kwargs):
    if method == "normalize":
        scaler = MinMaxScaler(feature_range=(kwargs.get('newMin', 0), kwargs.get('newMax', 1)))
    elif method == "standardize":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method in ["box-cox", "yeo-johnson"]:
        scaler = PowerTransformer(method=method)
    elif method in ["l1", "l2"]:
        norm = method
    elif method == "log":
        base = kwargs.get('base', np.e)
        log_transform = lambda x: np.log(x) / np.log(base) if base != np.e else np.log1p(x)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    train_values = train_df[[column_name]].values
    if method in ["l1", "l2"]:
        if fit_on_train:
            train_df[[column_name]] = normalize(train_values, norm=norm, axis=0)
            if test_df is not None:
                test_values = test_df[[column_name]].values
                test_df[[column_name]] = normalize(test_values, norm=norm, axis=0)
        else:
            combined_values = train_values if test_df is None else np.vstack((train_values, test_df[[column_name]].values))
            combined_scaled = normalize(combined_values, norm=norm, axis=0)
            train_df[[column_name]] = combined_scaled[:len(train_df)]
            if test_df is not None:
                test_df[[column_name]] = combined_scaled[len(train_df):]
    elif method == "log":
        train_df[[column_name]] = log_transform(train_values)
        if test_df is not None:
            test_values = test_df[[column_name]].values
            test_df[[column_name]] = log_transform(test_values)
    else:
        if fit_on_train:
            train_df[[column_name]] = scaler.fit_transform(train_values)
            if test_df is not None:
                test_values = test_df[[column_name]].values
                test_df[[column_name]] = scaler.transform(test_values)
        else:
            combined_values = train_values if test_df is None else np.vstack((train_values, test_df[[column_name]].values))
            combined_scaled = scaler.fit_transform(combined_values)
            train_df[[column_name]] = combined_scaled[:len(train_df)]
            if test_df is not None:
                test_df[[column_name]] = combined_scaled[len(train_df):]

    return train_df, test_df












def preprocess_column_creation_input(columnCreationInput):
    def parse_value(value):
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                pass
        return value

    for group in columnCreationInput['conditionGroups']:
        group['result'] = group['result'].replace('model.', 'df.')
        group['result'] = parse_value(group['result'])
        for condition_config in group["conditions"]: 
            condition_config["value"] = parse_value(condition_config["value"])

    columnCreationInput['defaultValue'] = columnCreationInput['defaultValue'].replace('model.', 'df.')
    return columnCreationInput

def newColumn_from_condition(columnCreationInput, df):
    def evaluate_condition(row, condition):

        if condition['operator'] == '>':
            return row[condition['column']] > condition['value']
        elif condition['operator'] == '>=':
            return row[condition['column']] >= condition['value']
        elif condition['operator'] == '<':
            return row[condition['column']] < condition['value']
        elif condition['operator'] == '<=':
            return row[condition['column']] <= condition['value']
        elif condition['operator'] == '=':
            return row[condition['column']] == condition['value']
        elif condition['operator'] == 'isin':
            return row[condition['column']] in condition['value']
        elif condition['operator'] == 'contains':
            return condition['value'] in row[condition['column']]

    def evaluate_conditions(row, conditions):
        return all(evaluate_condition(row, condition) for condition in conditions)

    result_column = []

    for i, row in df.iterrows():
        matched = False
        for group in columnCreationInput['conditionGroups']:
            if evaluate_conditions(row, group['conditions']):
                if group["resultType"] =="value":
                    result_column.append(group["result"])
                else:
                    expression = group['result']
                    try:
                        result = transform_and_execute(expression, df.iloc[[i]].reset_index()).iloc[0]

                    except Exception as e:
                        result = f"Error: {e}"
                    result_column.append(result)
                matched = True
                break
        if not matched:
            else_result = columnCreationInput['defaultValue']
            if columnCreationInput["defaultValueType"] =="value":
                        result_column.append(else_result)
            else:
                try:
                    else_result = transform_and_execute(else_result, df.iloc[[i]].reset_index()).iloc[0]
                except Exception as e:
                    else_result = f"Error: {e}"
                result_column.append(else_result)

    df[columnCreationInput['columnName']] = result_column
    return df






def calculateColumnStats(df, column):
        try:
            column_data = df[column].replace(r'^\s*$', np.nan, regex=True).astype(float)
            print(column_data)
            print("ssssssss")
            column_data = column_data.astype(float)

            print(column_data)
            is_numeric = True
        except ValueError:
            column_data = df[column].astype(str)
            column_data = column_data.replace('nan', pd.NA)

            is_numeric = False

        if is_numeric:
            # Calculate numeric statistics
            stats = {
                "count": int(column_data.count()),
                "mean": float(column_data.mean()),
                "median": float(column_data.median()),
                "std": float(column_data.std()),
                "var": float(column_data.var()),
                "min": float(column_data.min()),
                "25%": float(column_data.quantile(0.25)),
                "50%": float(column_data.quantile(0.50)),
                "75%": float(column_data.quantile(0.75)),
                "max": float(column_data.max()),
                "missing": int(column_data.isnull().sum())
            }
            histogram = np.histogram(column_data.dropna(), bins=20)
            histogram_data = {
                "bins": histogram[1].tolist(),
                "counts": histogram[0].tolist()
            }
        else:
            # Calculate categorical statistics
            mode = column_data.mode().tolist()
            stats = {
                "count": int(column_data.count()),
                "missing": int(column_data.isnull().sum()),
                "mode": mode[:min(len(mode),3)]
            }
            value_counts = column_data.value_counts().nlargest(10)
            histogram_data = {
                "bins": value_counts.index.tolist(),
                "counts": value_counts.tolist()
            }
            print(histogram_data)

        response_data = {
            "stats": stats,
            "histogram": histogram_data,
            "is_numeric": is_numeric
        }
        
        return {"message": "Data received", "data": json.dumps(response_data)}

def handle_missing_values(df, column, method, value=None, group_by=None, interpolate_col=None, consider_nan_as_category=False, fit_to_train=True, test_df=None):
    if group_by and method in ["Mean", "Median", "Most Common"]:
        if consider_nan_as_category:
            df[group_by] = df[group_by].fillna('NaN')
            if test_df is not None:
                test_df[group_by] = test_df[group_by].fillna('NaN')

        # Drop rows where group_by column has NaNs
        df_nonan = df.dropna(subset=[group_by])
        
        if method == "Mean":
            group_means = df_nonan.groupby(group_by)[column].mean()
            df[column] = df.apply(lambda row: row[column] if pd.notna(row[column]) else group_means.get(row[group_by], pd.NA), axis=1)
            if test_df is not None:
                test_df[column] = test_df.apply(lambda row: row[column] if pd.notna(row[column]) else group_means.get(row[group_by], pd.NA), axis=1)

        elif method == "Median":
            group_medians = df_nonan.groupby(group_by)[column].median()
            df[column] = df.apply(lambda row: row[column] if pd.notna(row[column]) else group_medians.get(row[group_by], pd.NA), axis=1)
            if test_df is not None:
                test_df[column] = test_df.apply(lambda row: row[column] if pd.notna(row[column]) else group_medians.get(row[group_by], pd.NA), axis=1)

        elif method == "Most Common":
            group_modes = df_nonan.groupby(group_by)[column].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
            df[column] = df.apply(lambda row: row[column] if pd.notna(row[column]) else group_modes.get(row[group_by], pd.NA), axis=1)
            if test_df is not None:
                test_df[column] = test_df.apply(lambda row: row[column] if pd.notna(row[column]) else group_modes.get(row[group_by], pd.NA), axis=1)

        if consider_nan_as_category:
            df[group_by] = df[group_by].replace('NaN', pd.NA)
            if test_df is not None:
                test_df[group_by] = test_df[group_by].replace('NaN', pd.NA)

    else:
        if method == "Mean":
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
            if test_df is not None:
                test_df[column].fillna(mean_value, inplace=True)

        elif method == "Median":
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
            if test_df is not None:
                test_df[column].fillna(median_value, inplace=True)

        elif method == "Most Common":
            mode_value = df[column].mode()[0] if not df[column].mode().empty else pd.NA
            df[column].fillna(mode_value, inplace=True)
            if test_df is not None:
                test_df[column].fillna(mode_value, inplace=True)

        elif method == "Assign Value":
            df[column].fillna(value, inplace=True)
            if test_df is not None:
                test_df[column].fillna(value, inplace=True)

        elif method == "Remove Row":
            df.dropna(subset=[column], inplace=True)
            if test_df is not None:
                test_df.dropna(subset=[column], inplace=True)

        elif method == "Interpolate":
            if interpolate_col:
                original_index = df.index
                df = df.sort_values(by=interpolate_col)
                df[column] = df[column].interpolate()
                df = df.loc[original_index]
                if test_df is not None:
                    original_index_test = test_df.index
                    test_df = test_df.sort_values(by=interpolate_col)
                    test_df[column] = test_df[column].interpolate()
                    test_df = test_df.loc[original_index_test]
            else:
                df[column] = df[column].interpolate()
                if test_df is not None:
                    test_df[column] = test_df[column].interpolate()

        elif method == "Forward Fill":
            df[column].fillna(method='ffill', inplace=True)
            if test_df is not None:
                test_df[column].fillna(method='ffill', inplace=True)

        elif method == "Back Fill":
            df[column].fillna(method='bfill', inplace=True)
            if test_df is not None:
                test_df[column].fillna(method='bfill', inplace=True)

    return df, test_df


