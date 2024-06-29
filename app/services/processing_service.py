"""Feature engineering functions"""

import pandas as pd

# df = pd.read_csv("Loan_default 2.csv")


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

    # Drop the original column
    df_copy.drop(columns=[column_name], inplace=True)
    return df_copy


def normalize_column(df, column_name, new_min=0, new_max=0):
    """Normalizes a column in a dataframe to a new range. Default is 0 to 1."""
    df[column_name] = pd.to_numeric(df[column_name])
    print(type(new_min))

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
        group['result'] = group['result'].replace('model.', '')

    columnCreationInput['defaultValue'] = columnCreationInput['defaultValue'].replace('model.', '')

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

    # Preprocess the columnCreationInput to remove "model." prefix
    if len(columnCreationInput['conditionGroups']) > 0:
        columnCreationInput = preprocess_column_creation_input(columnCreationInput)

    result_column = []

    for _, row in df.iterrows():
        if not columnCreationInput['conditionGroups']:
            else_expression = columnCreationInput['defaultValue']
            try:
                else_result = pd.eval(else_expression, local_dict=row.to_dict())
            except Exception as e:
                else_result = f"Error: {e}"
            result_column.append(else_result)

        else:
            matched = False
            for group in columnCreationInput['conditionGroups']:
                if evaluate_conditions(row, group['conditions']):
                    expression = group['result']
                    try:
                        result = pd.eval(expression, local_dict=row.to_dict())
                    except Exception as e:
                        result = f"Error: {e}"
                    result_column.append(result)
                matched = True
                break
            if not matched:
                else_expression = columnCreationInput['defaultValue']
                try:
                    else_result = pd.eval(else_expression, local_dict=row.to_dict())
                except Exception as e:
                    else_result = f"Error: {e}"
                result_column.append(else_result)

    df[columnCreationInput['columnName']] = result_column
    return df
