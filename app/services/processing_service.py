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

def normalize_column(df, column_name, new_min=0, new_max=0):
    """Normalizes a column in a dataframe to a new range. Default is 0 to 1."""
    df[column_name] = pd.to_numeric(df[column_name])
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name] = (df[column_name] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return df

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

