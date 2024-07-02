"""Example data with correct format for testing the API"""

Data = {
  "data": [
    {
      "columns": ["Header1", "Header2", "Header3"]
    },
    {
      "columns": ["Row1Col1", "Row1Col2", "Row1Col3"]
    },
    {
      "columns": ["Row2Col1", "Row2Col2", "Row2Col3"]
    },
    {
      "columns": ["Row3Col1", "Row3Col2", "Row3Col3"]
    }
  ]
}

OneHotDefs = {
  "OneHotDefs": {"test": [1]}
}

{
  "data": {
    "data": [
      {
        "columns": ["Header1", "Header2", "Header3"]
      },
      {
        "columns": ["Row1Col1", "Row1Col2", "Row1Col3"]
      },
      {
        "columns": ["Row2Col1", "Row2Col2", "Row2Col3"]
      },
      {
        "columns": ["Row3Col1", "Row3Col2", "Row3Col3"]
      }
    ]
  },
  "column_defs": {
    "OneHotDefs": {}
  }
}

columnCreation = {
  "columnName": "New Column",
  "conditionGroups": [
    {
      "conditions": [
        {
          "column": "age",
          "operator": ">",
          "value": "30"
        },
        {
          "column": "income",
          "operator": "=",
          "value": "50000"
        }
      ],
      "result": "High Income",
      "resultType": "value"
    },
    {
      "conditions": [
        {
          "column": "age",
          "operator": "<=",
          "value": "30"
        },
        {
          "column": "status",
          "operator": "isin",
          "value": ["single", "divorced"]
        }
      ],
      "result": "Low Income",
      "resultType": "expression"
    }
  ],
  "elseValue": "Unknown",
  "elseValueType": "expression"
}


# linear_regression
{
  "data": {
    "data": [
      {"columns": ["age", "salary", "children"]},
      {"columns": ["17", "2000", "0"]},
      {"columns": ["30", "50000", "2"]},
      {"columns": ["45", "80000", "3"]},
      {"columns": ["25", "30000", "1"]},
      {"columns": ["35", "60000", "2"]},
      {"columns": ["50", "90000", "2"]},
      {"columns": ["28", "40000", "0"]},
      {"columns": ["40", "70000", "1"]},
      {"columns": ["22", "25000", "0"]},
      {"columns": ["32", "55000", "1"]},
      {"columns": ["55", "100000", "3"]},
      {"columns": ["38", "65000", "2"]},
      {"columns": ["42", "75000", "2"]},
      {"columns": ["20", "28000", "0"]},
      {"columns": ["33", "60000", "1"]},
      {"columns": ["48", "85000", "3"]},
      {"columns": ["27", "38000", "1"]},
      {"columns": ["37", "68000", "2"]},
      {"columns": ["52", "95000", "2"]}
    ]
  },
  "indep_var": "age",
  "dep_vars": [
    "salary", "children"
  ],
  "train_size": 0.8,
  "split_seed": -1
}
