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
    df_copy[:40].to_csv("test.csv")
    return df_copy