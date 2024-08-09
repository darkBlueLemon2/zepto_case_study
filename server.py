import streamlit as st
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from googletrans import Translator

# Load pre-trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
    return tokenizer, model

# Load embeddings and product data
@st.cache_data
def load_embeddings_and_data(file_path):
    with open(file_path, 'rb') as f:
        embeddings, df = pickle.load(f)
    return embeddings, df

# Function to translate text to English
def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, dest='en')
    return translated.text

# Function to search for products based on a query description
def search_products(query, embeddings, df, tokenizer, model, top_n=10000):
    # Translate query to English
    english_query = translate_to_english(query)
    
    # Tokenize and encode the query
    encoded_query = tokenizer([english_query], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = mean_pooling(query_output, encoded_query['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    
    # Get top N most similar products
    top_indices = similarities.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices]
    results['cosine_similarity'] = similarities[top_indices]
    
    return results, english_query

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    st.title("Product Search App")
    
    # Load model, tokenizer, embeddings, and product data
    tokenizer, model = load_model_and_tokenizer()
    embeddings, df = load_embeddings_and_data('product_embeddings.pkl')
    
    # Convert 'product_rating' to numeric, replacing non-numeric values with NaN
    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    
    # Convert 'retail_price' to numeric, replacing non-numeric values with NaN
    df['retail_price'] = pd.to_numeric(df['retail_price'], errors='coerce')
    
    # Convert 'brand' to string, handling NaNs
    df['brand'] = df['brand'].astype(str)
    
    # Input query
    query = st.text_input("Enter your search query:")
    
    if query:
        # Search for products
        results, english_query = search_products(query, embeddings, df, tokenizer, model)
        
        # Display the translated query
        # st.write(f"Translated query: {english_query}")
        
        if not results.empty:
            results = results.dropna(subset=['description'])
            
            # Filters
            st.sidebar.header("Filters")
            
            # Convert 'brand' to string in results, handling NaNs
            results['brand'] = results['brand'].astype(str)
            
            # Brand filter
            brands = sorted(results['brand'].unique())
            selected_brands = st.sidebar.multiselect("Select brands", brands)
            
            # Rating filter
            rating_options = ["All", "Above 4 stars", "Above 3 stars", "Above 2 stars", "Above 1 star"]
            selected_rating = st.sidebar.selectbox("Select rating", rating_options)
            
            # Price range filter
            min_price = float(results['retail_price'].min())
            max_price = float(results['retail_price'].max())
            price_range = st.sidebar.slider("Price range", min_price, max_price, (min_price, max_price))
            
            # Cosine similarity cutoff filter
            similarity_cutoff = st.sidebar.slider("Cosine similarity cutoff", 0.0, 1.0, 0.2)
            
            # Apply filters
            filtered_results = results.copy()
            
            if selected_brands:
                filtered_results = filtered_results[filtered_results['brand'].isin(selected_brands)]
            
            if selected_rating == "Above 4 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 4]
            elif selected_rating == "Above 3 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 3]
            elif selected_rating == "Above 2 stars":
                filtered_results = filtered_results[filtered_results['product_rating'] > 2]
            elif selected_rating == "Above 1 star":
                filtered_results = filtered_results[filtered_results['product_rating'] > 1]
            
            filtered_results = filtered_results[
                (filtered_results['retail_price'] >= price_range[0]) & 
                (filtered_results['retail_price'] <= price_range[1]) &
                (filtered_results['cosine_similarity'] >= similarity_cutoff)
            ]
            
            # Display filtered results
            st.write(f"Showing {len(filtered_results)} filtered products:")
            
            # Select columns to display in the table
            columns_to_display = ['product_name', 'description', 'retail_price', 'discounted_price', 'brand', 'product_rating', 'cosine_similarity']
            results_table = filtered_results[columns_to_display]
            
            # Format the 'product_rating' and 'cosine_similarity' columns to display up to 2 decimal places
            results_table['product_rating'] = results_table['product_rating'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
            results_table['cosine_similarity'] = results_table['cosine_similarity'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(results_table, height=400)  # Adjust height as needed
        else:
            st.write("No matching products found.")

if __name__ == "__main__":
    main()