import pandas as pd

#df =pd.read_csv("Loan_default 2.csv")

def onehotEncoding(df, column_name, column_defs = None):
    '''
    example column defs
    column_name
    column_defs = {
    group1_name:[values],
    group2_name:[values]},
    group3_name:[values]
    }

    returns a df where column name is one hot encoded based on the column_defs
    '''
    df_copy = df.copy()
    unique_values = df_copy[column_name].unique().tolist()
    if column_defs is None:
        for value in unique_values:
            df_copy[f'{column_name}: {value}'] =df_copy[column_name].apply(lambda x: 1 if x == value else 0)

    else: #### probably useless

        all_values_selected = []
        print(unique_values)
        # Iterate over the column definitions to create one-hot encoded columns
        for group_name, values in column_defs.items():
            # Create a new column for each group
            df_copy[group_name] = df_copy[column_name].apply(lambda x: 1 if x in values else 0)
            all_values_selected+=values
        print(all_values_selected)

        remaining_values= list(set(unique_values) - set(all_values_selected))
        print(remaining_values)
        if len(remaining_values)>0:
            df_copy["other_"+column_name] = df_copy[column_name].apply(lambda x: 1 if x in remaining_values else 0)

    # Drop the original column
    df_copy.drop(columns=[column_name], inplace=True)
    return df_copy


#----------

def normalize_column(df, column_name, new_min = 0, new_max =0):
    df[column_name] = pd.to_numeric(df[column_name])
    print(type(new_min))
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name] = (df[column_name] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return df