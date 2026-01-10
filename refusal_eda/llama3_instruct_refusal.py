import pandas as pd

refusal_phrase = [
    "I'm not sure about the accuracy of the summary."
]

df = pd.read_csv("refusal_eda/meta-llama_Meta-Llama-3-70B-Instruct_outputs_20260108_085044.csv")
print(df.columns)
print(df['response'].isna().sum())
print(df['content'].str.split().apply(len).describe())

# groupby age_rating_grouped and content, get the na in each gorup
na_df = df[df['response'].isna()]
print(na_df['age_rating_grouped'].value_counts())
print(na_df['content'].str.split().apply(len).describe())