import openai

# Define your standard column names
standard_columns = ['Date', 'Amount', 'Description', 'Category', 'Account']

def get_column_mapping(column_name):
    # Use OpenAI API to suggest a mapping for a given column name
    response = openai.Completion.create(
        model="gpt-4",  # Make sure to use a model that supports completions and contextual understanding
        prompt=f"Map the column name '{column_name}' to the most relevant standard column from {standard_columns}. "
               f"Suggest the best match based on typical financial data context.",
        max_tokens=20,
        temperature=0
    )
    # Extract suggested column mapping from the response
    suggestion = response.choices[0].text.strip()
    return suggestion