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
