import pandas as pd

# Read the datasets
movie = pd.read_csv('TMDB_with_ME.csv')
tv_s = pd.read_csv('short_TMDB_with_ME.csv')
tv_m = pd.read_csv('mid_wiki_with_ME.csv')
tv_l = pd.read_csv('long_IMDB_with_ME.csv')

#print(movie['Release year'].describe())

# Renaming columns to maintain consistency
movie_select = movie[['Title', 'rating', 'plots', 'OpenAI_ME_bool', 'OpenAI_ME_responses']].rename(columns={
    'Title': 'name',
    'rating': 'age_rating',
    'plots': 'content'
})

tv_s_select = tv_s[['show_name', 'age_rating', 'episode-overview', 'OpenAI_ME_bool', 'OpenAI_ME_responses']].rename(columns={
    'show_name': 'name',
    'episode-overview': 'content'
})

tv_m_select = tv_m[['show_name', 'age_rating', 'wiki_descs', 'OpenAI_ME_bool', 'OpenAI_ME_responses']].rename(columns={
    'show_name': 'name',
    'wiki_descs': 'content'
})

tv_l_select = tv_l[['show_name', 'age_rating', 'synopsis_with_character_names', 'OpenAI_ME_bool', 'OpenAI_ME_responses']].rename(columns={
    'show_name': 'name',
    'synopsis_with_character_names': 'content'
})

# Concatenating all datasets into one
combined_df = pd.concat([movie_select, tv_s_select, tv_m_select, tv_l_select], ignore_index=True)

# Define the mapping dictionary
# citing https://www.filmratings.com/ and https://www.tvguidelines.org/resources/TheRatings.pdf
age_rating_map = {
    # General Audience/Children
    'G': 'G', 'TV-G': 'G', 'TV-Y': 'G', 'TV-Y7': 'G', 'TV-Y7-FV': 'G',

    # Parental Guidance (Mild Content)
    'PG': 'PG', 'TV-PG': 'PG',

    # Teens/Stronger Content
    'PG-13': 'PG-13', 'TV-14': 'PG-13',

    # Adults Only
    'R': 'R', 'NC-17': 'R', 'TV-MA': 'R', 'M': 'R', '16+': 'R',# Group mature content here
}

# Apply the mapping to the 'age_rating' column
combined_df['age_rating_grouped'] = combined_df['age_rating'].map(age_rating_map)

# deduplicate the dataset on content 
combined_df = combined_df.drop_duplicates(subset=['content'], keep='first')

# Print value counts for the grouped age ratings
print(combined_df['age_rating_grouped'].value_counts())
print(combined_df.shape)

# save the dataset
# combined_df.to_csv('movie_TV_raw_data.csv', index=False)