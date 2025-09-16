import os
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import random
import argparse
import pandas as pd

from src.indexer import initialize_vectorstore, count_tokens_and_cost

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..",))

load_dotenv(dotenv_path=os.path.join(get_project_root(), '.env'))

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY environment variable not found. "
        "Please set this variable in your environment or .env file."
    )

BANNED_SET = {
    # Sape (Music)
    ("Indonesia", "music", "sape"): [("Malaysia", "music", "sape"), ("Philippines", "music", "kudyapi")],
    ("Malaysia", "music", "sape"): [("Indonesia", "music", "sape"), ("Philippines", "music", "kudyapi")],
    ("Philippines", "music", "kudyapi"): [("Indonesia", "music", "sape"), ("Malaysia", "music", "sape")],
    
    # Angklung (Music)
    ("Indonesia", "music", "angklung"): [("Thailand", "music", "อังกะลุง")],
    ("Thailand", "music", "อังกะลุง"): [("Indonesia", "music", "angklung")],
    
    # Suling/Sáo/ပလွေ (Music)
    ("Indonesia", "music", "suling"): [("Vietnam", "music", "sáo"), ("Myanmar", "music", "ပလွေ")],
    ("Vietnam", "music", "sáo"): [("Indonesia", "music", "suling"), ("Myanmar", "music", "ပလွေ")],
    ("Myanmar", "music", "ပလွေ"): [("Indonesia", "music", "suling"), ("Vietnam", "music", "sáo")],
    
    # Gambang/ระนาด (Music)
    ("Indonesia", "music", "gambang"): [("Thailand", "music", "ระนาด")],
    ("Thailand", "music", "ระนาด"): [("Indonesia", "music", "gambang")],
    
    # Kecapi/đàn tranh (Music)
    ("Indonesia", "music", "kecapi"): [("Vietnam", "music", "đàn tranh")],
    ("Vietnam", "music", "đàn tranh"): [("Indonesia", "music", "kecapi")],
    
    # Rebab (Music)
    ("Indonesia", "music", "rebab"): [("Malaysia", "music", "rebab")],
    ("Malaysia", "music", "rebab"): [("Indonesia", "music", "rebab")],
    
    # Talempong/Kulintang (Music)
    ("Indonesia", "music", "talempong"): [("Philippines", "music", "kulintang")],
    ("Philippines", "music", "kulintang"): [("Indonesia", "music", "talempong")],
    
    # Engklek/nhảy lò cò (Game)
    ("Indonesia", "game", "engklek"): [("Vietnam", "game", "nhảy lò cò")],
    ("Vietnam", "game", "nhảy lò cò"): [("Indonesia", "game", "engklek")],
    
    # Tug-of-war/ชักเย่อ (Game)
    ("Thailand", "game", "ชักเย่อ"): [("Myanmar", "game", "tug-of-war")],
    ("Myanmar", "game", "tug-of-war"): [("Thailand", "game", "ชักเย่อ")],
    
    # Selendang/Belo (Wedding)
    ("Indonesia", "wedding", "selendang"): [("Philippines", "wedding", "belo")],
    ("Philippines", "wedding", "belo"): [("Indonesia", "wedding", "selendang")],
    
    # Congklak/หมากข่าง (Game)
    ("Indonesia", "game", "permainan congklak"): [("Thailand", "game", "หมากข่าง")],
    ("Thailand", "game", "หมากข่าง"): [("Indonesia", "game", "permainan congklak")],
    
    # Thingyan/Songkran (Celebration)
    ("Myanmar", "celebration", "thingyan festival"): [("Thailand", "celebration", "เทศกาลสงกรานต์")],
    ("Thailand", "celebration", "เทศกาลสงกรานต์"): [("Myanmar", "celebration", "thingyan festival")],
    
    # Dowry (Wedding)
    ("Indonesia", "wedding", "uang panai"): [("Thailand", "wedding", "สินสอด")],
    ("Thailand", "wedding", "สินสอด"): [("Indonesia", "wedding", "uang panai")],
    
    # Wedding Bouquet
    ("Indonesia", "wedding", "bunga nikah"): [("Philippines", "wedding", "bouquet")],
    ("Philippines", "wedding", "bouquet"): [("Indonesia", "wedding", "bunga nikah")]
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

total_tokens = 0
total_cost = 0

def get_banned_items(country, category, concept):
    key = (country, category.lower(), concept.lower())
    if key in BANNED_SET:
        return [(c, cat, n) for (c, cat, n) in BANNED_SET[key]]
    return []

def get_similar_items(query_text, vectorstore, k=3, filter_dict=None, current_item=None):
    # Get banned items if current_item is provided
    banned_items = []
    if current_item:
        country, category, concept = current_item
        banned_items = get_banned_items(country, category, concept)

    # Add banned items to filter criteria
    if filter_dict is None:
        filter_dict = {}
    
    if banned_items:
        # Create a list of banned combinations
        banned_combinations = []
        for country, category, concept in banned_items:
            banned_combinations.append({
                "Country": country,
                "Category": category,
                "Concept": concept
            })
        
        # Add to filter_dict to exclude these combinations
        if "$and" not in filter_dict:
            filter_dict["$and"] = []
        filter_dict["$and"].append({
            "$not": {"$or": banned_combinations}
        })
    
    tokens, cost = count_tokens_and_cost(query_text)
    global total_tokens, total_cost
    total_tokens += tokens
    total_cost += cost

    results = vectorstore.similarity_search_with_score(
        query_text, k=k, filter=filter_dict
    )
    return results

def write_question_to_jsonl(question_dict, filename):
    with open(filename, "a") as f:
        json.dump(question_dict, f)
        f.write("\n")

def filter_similar_items(similar_items):
    filtered_items = {}
    seen_concepts = set()  # Track seen concepts to avoid duplicates
    
    for item, score in similar_items:
        concept = item.metadata['Concept']
        country = item.metadata['Country']

        # Create a unique key combining concept and country
        key = (concept, country)

        if key not in seen_concepts:
            filtered_items[key] = (item, score)
            seen_concepts.add(key)

    return list(filtered_items.values())

def check_existing_file(filename, generated_mcq_dir):
    file_path = os.path.join(generated_mcq_dir, filename)
    if os.path.exists(file_path):
        response = input(f"File {filename} already exists. Do you want to overwrite it? (y/n): ")
        return response.lower() == 'y'
    return True

def has_too_many_overlaps(new_choices, existing_choices_set, overlap_threshold=2):
    new_choices_set = frozenset(choice['id'] for choice in new_choices)
    
    for existing_choices in existing_choices_set:
        overlap_count = len(new_choices_set.intersection(existing_choices))
        if overlap_count >= overlap_threshold:
            return True
    
    return False

def get_random_choices_from_pool(df, filter_criteria, choice_usage, max_choice_usage, num_choices=3, current_item=None, mcq_type=None):
    """Get random choices from the valid pool based on filter criteria"""
    # Get banned items if current_item is provided
    banned_items = []
    if current_item:
        country, category, concept = current_item
        banned_items = get_banned_items(country, category, concept)

    # Filter the dataframe based on the criteria
    filtered_df = df.copy()
    for col, val in filter_criteria.items():
        if isinstance(val, dict):
            if "$eq" in val:
                filtered_df = filtered_df[filtered_df[col] == val["$eq"]]
            elif "$neq" in val:
                filtered_df = filtered_df[filtered_df[col] != val["$neq"]]
            elif "$nin" in val:
                filtered_df = filtered_df[~filtered_df[col].isin(val["$nin"])]
        else:
            filtered_df = filtered_df[filtered_df[col] == val]
    
    # Filter out banned items
    if banned_items:
        for country, category, concept in banned_items:
            filtered_df = filtered_df[
                ~((filtered_df['Country'] == country) & 
                  (filtered_df['Category'].str.lower() == category) & 
                  (filtered_df['Concept'].str.lower() == concept))
            ]
    
    # Filter out items that have reached max usage
    filtered_df = filtered_df[~filtered_df["Image Path"].isin([id for id, count in choice_usage.items() 
                                                     if count >= max_choice_usage])]
    
    # Special handling for Type 2 MCQs (different countries)
    if mcq_type == 2:
        random_choices = []
        used_countries = set()  # Track countries we've used
        used_concepts = set()      # Track concepts we've used
        
        # Add the current item's country to used countries
        if current_item:
            used_countries.add(current_item[0])
            used_concepts.add(current_item[2])
        
        # First, get all available countries (excluding used ones)
        available_countries = set(filtered_df['Country'].unique()) - used_countries
        
        # If we don't have enough unique countries, return empty list
        if len(available_countries) < num_choices:
            return []
        
        # Randomly select countries we'll use
        selected_countries = random.sample(list(available_countries), num_choices)
        
        # For each selected country, randomly select one item
        for selected_country in selected_countries:
            # Get all items from this country
            country_df = filtered_df[
                (filtered_df['Country'] == selected_country) &
                (~filtered_df['Concept'].isin(used_concepts))
            ]
            
            if len(country_df) == 0:
                continue
                
            # Randomly select one item from this country
            choice = country_df.sample(n=1).iloc[0]
            
            # Add the choice
            random_choices.append({
                'id': choice['Image Path'],
                'score': 0.0,
                'metadata': {
                    'Country': choice['Country'],
                    'Category': choice['Category'],
                    'Concept': choice['Concept'],
                    'Image Path': choice['Image Path'],
                    'Object': choice['Object']
                }
            })
            
            # Update tracking sets
            used_countries.add(choice['Country'])
            used_concepts.add(choice['Concept'])
        
        if len(random_choices) < num_choices:
            return []
            
        return random_choices
    
    # Original logic for Type 1 MCQs
    else:
        # Group by Concept and take only one row per concept to avoid duplicates
        filtered_df = filtered_df.groupby('Concept').first().reset_index()

        if len(filtered_df) < num_choices:
            return []
        
        # Sample random choices
        random_choices = filtered_df.sample(n=num_choices)
        
        # Format the choices
        formatted_choices = []
        for _, choice in random_choices.iterrows():
            formatted_choices.append({
                'id': choice['Image Path'],
                'score': 0.0,
                'metadata': {
                    'Country': choice['Country'],
                    'Category': choice['Category'],
                    'Concept': choice['Concept'],
                    'Image Path': choice['Image Path'],
                    'Object': choice['Object']
                }
            })
        
        return formatted_choices

def has_duplicate_choices(choices):
    """Check if there are any duplicate choices in the options"""
    seen = set()
    for choice in choices:
        if 'metadata' in choice:
            choice_key = (
                choice['metadata']['Country'],
                choice['metadata']['Category'],
                choice['metadata']['Concept']
            )
        else:
            # For the correct answer that doesn't have metadata
            continue
            
        if choice_key in seen:
            return True
        seen.add(choice_key)
    return False

def generate_mcq_questions_v1(
    df,
    vectorstore,
    max_question_per_concept=3,
    max_choice_usage=5,
    top_k=30,
    output_filename="questions.jsonl",
):
    """Generate MCQ questions - different country, same category (Type 2)"""
    NOT_ENOUGH_CHOICES = 0
    RANDOM_FALLBACK_COUNT = 0
    
    # Ensure required columns exist
    required_columns = ["Concept", "Category", "Country", "Image Path", "Question", "Rationale", "text"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    # TYPE 2 GENERATION - different country, same category
    choice_usage = defaultdict(int)
    banned_IDs = set()

    # Order concepts by frequency (from least to most common)
    concept_counts = df["Concept"].value_counts().sort_values(ascending=True)
    ordered_concepts = concept_counts.index.tolist()

    existing_choice_sets = set()  # Store all generated choice sets

    for concept in tqdm(ordered_concepts, desc="Generating MCQs (different country)", total=len(ordered_concepts)):
        # establish the country and category
        country = df[df["Concept"] == concept]["Country"].unique()[0]
        category = df[df["Concept"] == concept]["Category"].unique()[0]

        # get relevant columns for this concept
        concept_df = df[df["Concept"] == concept][
            ["Image Path", "Question", "Rationale", "text"]
        ].sample(n=min(max_question_per_concept, len(df[df["Concept"] == concept])))

        # for each question, generate the mcq
        for _, row in concept_df.iterrows():
            question = row["Question"]
            rationale = row["Rationale"]
            context = row["text"]
            incorrect_choices_candidate = get_similar_items(
                context,
                vectorstore,
                k=top_k,
                filter_dict={
                    "Country": {"$neq": country},
                    "Category": category,
                    "Image Path": {"$nin": list(banned_IDs)}
                },
                current_item=(country, category, concept)
            )

            # filter the results of incorrect choices to only include top N of each Concept
            incorrect_choices_candidate = filter_similar_items(incorrect_choices_candidate)
            
            # Get the incorrect choices with their Image Paths and scores
            found_diverse_choices = False
            final_choices = None
            
            # Try different combinations from top_k candidates
            for k in range(0, len(incorrect_choices_candidate) - 2):
                incorrect_choices = []
                need_random_fallback = False
                used_countries = set()  # Track used countries
                
                # Add the current item's country to used countries
                used_countries.add(country)
                
                # Try to find 3 items from different countries
                candidate_index = k
                while len(incorrect_choices) < 3 and candidate_index < len(incorrect_choices_candidate):
                    choice, score = incorrect_choices_candidate[candidate_index]
                    candidate_index += 1
                    
                    # Skip if country already used or max usage reached
                    if choice.metadata['Country'] in used_countries or choice_usage[choice.metadata['Image Path']] >= max_choice_usage:
                        continue
                        
                    choice_usage[choice.metadata['Image Path']] += 1
                    if choice_usage[choice.metadata['Image Path']] >= max_choice_usage:
                        banned_IDs.add(choice.metadata['Image Path'])
                        
                    incorrect_choices.append({
                        'id': choice.metadata['Image Path'],
                        'score': float(score),
                        'metadata': choice.metadata
                    })
                    
                    # Track this country
                    used_countries.add(choice.metadata['Country'])
                
                # If we couldn't find 3 choices from different countries, try next starting point
                if len(incorrect_choices) < 3:
                    # Revert usage counts for choices we couldn't use
                    for choice in incorrect_choices:
                        choice_usage[choice['id']] -= 1
                    continue
                
                # Add correct answer
                choices = incorrect_choices.copy()
                choices.append({
                    'id': row['Image Path'],
                    'score': -1.0
                })
                
                # Check if this combination has too many overlaps
                if not has_too_many_overlaps(choices, existing_choice_sets):
                    found_diverse_choices = True
                    final_choices = choices
                    break
            
            # If we couldn't find a diverse set or not enough choices, use random fallback
            if not found_diverse_choices or len(incorrect_choices_candidate) < 3:
                # Define filter criteria for random selection
                filter_criteria = {
                    "Country": {"$neq": country},
                    "Category": category,
                    "Concept": {"$neq": concept}
                }
                
                random_choices = get_random_choices_from_pool(
                    df, filter_criteria, choice_usage, max_choice_usage, 
                    num_choices=3, 
                    current_item=(country, category, concept),
                    mcq_type=2  # Specify this is for Type 2 MCQs
                )
                
                if len(random_choices) < 3:
                    NOT_ENOUGH_CHOICES += 1
                    continue
                
                # Update choice usage for random choices
                for choice in random_choices:
                    choice_usage[choice['id']] += 1
                    if choice_usage[choice['id']] >= max_choice_usage:
                        banned_IDs.add(choice['id'])
                
                RANDOM_FALLBACK_COUNT += 1
                
                final_choices = random_choices.copy()
                final_choices.append({
                    'id': row['Image Path'],
                    'score': -1.0
                })
            
            # Add this choice set to our tracking set
            choice_ids = frozenset(choice['id'] for choice in final_choices)
            existing_choice_sets.add(choice_ids)
            
            # After creating final_choices but before shuffling:
            if has_duplicate_choices(final_choices):
                # Try getting new random choices
                continue

            # Only proceed if no duplicates found
            random.shuffle(final_choices)
            choice_letters = ['a', 'b', 'c', 'd']

            # Build the MCQ
            mcq = {
                "country": country,
                "category": category,
                "concept": concept,
                "question": question,
                "rationale": rationale,
            }

            # Store scores in a list according to choice order
            scores = []

            for _, (choice, letter) in enumerate(zip(final_choices, choice_letters)):
                if choice['score'] == -1.0:  # Correct answer
                    mcq[f"choice_{letter}"] = f"{choice['id']}"
                else:  # Incorrect choices
                    # Get the metadata for this choice 
                    if 'metadata' in choice:
                        choice_metadata = choice['metadata']
                    else:
                        # For random choices
                        choice_metadata = next((c.metadata for c, s in incorrect_choices_candidate 
                                              if c.metadata['Image Path'] == choice['id']), None)
                        if choice_metadata is None:
                            # Get metadata from dataframe for random choices
                            choice_df = df[df['Image Path'] == choice['id']]
                            if not choice_df.empty:
                                choice_metadata = {
                                    'Country': choice_df['Country'].iloc[0],
                                    'Category': choice_df['Category'].iloc[0],
                                    'Concept': choice_df['Concept'].iloc[0],
                                    'Object': choice_df['Object'].iloc[0],
                                    'Image Path': choice['id']
                                }
                    
                    mcq[f"choice_{letter}"] = f"{choice['id']}"
                
                scores.append(choice['score'])
                
                if choice['score'] == -1.0:
                    mcq["correct_answer"] = f"{choice['id']}"

            mcq["scores"] = scores
            mcq["type"] = 2  # Type 2 for different country, same category

            write_question_to_jsonl(mcq, output_filename)

    print(f"Type 2 MCQs - Skipped {NOT_ENOUGH_CHOICES} questions due to insufficient choices")
    print(f"Type 2 MCQs - Used random fallback for {RANDOM_FALLBACK_COUNT} questions")

def generate_mcq_questions_v2(
    df,
    vectorstore,
    max_question_per_concept=3,
    max_choice_usage=5,
    top_k=3,
    output_filename="questions.jsonl",
):
    """Generate MCQ questions - same country, same category (Type 1)"""
    NOT_ENOUGH_CHOICES = 0
    RANDOM_FALLBACK_COUNT = 0

    # TYPE 1 GENERATION - same country, same category
    choice_usage = defaultdict(int)
    banned_IDs = set()

    # Order concepts by frequency (from least to most common)
    concept_counts = df["Concept"].value_counts().sort_values(ascending=True)
    ordered_concepts = concept_counts.index.tolist()

    existing_choice_sets = set()  # Store all generated choice sets

    for concept in tqdm(ordered_concepts, desc="Generating MCQs (same country)", total=len(ordered_concepts)):
        # establish the country and category
        country = df[df["Concept"] == concept]["Country"].unique()[0]
        category = df[df["Concept"] == concept]["Category"].unique()[0]

        # get relevant columns for this concept
        concept_df = df[df["Concept"] == concept][
            ["Image Path", "Question", "Rationale", "text"]
        ].sample(n=min(max_question_per_concept, len(df[df["Concept"] == concept])))

        # for each question, generate the mcq
        for _, row in concept_df.iterrows():
            question = row["Question"]
            rationale = row["Rationale"]
            context = row["text"]
            incorrect_choices_candidate = get_similar_items(
                context,
                vectorstore,
                k=top_k,
                filter_dict={
                    "Country": {"$eq": country},  # Same country
                    "Category": category,
                    "Concept": {"$neq": concept},  # Different concept
                    "Image Path": {"$nin": list(banned_IDs)}
                },
                current_item=(country, category, concept)
            )

            # filter the results of incorrect choices to only include top N of each Concept
            incorrect_choices_candidate = filter_similar_items(incorrect_choices_candidate)
            
            # Get the incorrect choices with their IDs and scores
            found_diverse_choices = False
            final_choices = None
            
            # Try different combinations from top_k candidates
            for k in range(0, len(incorrect_choices_candidate) - 2):  # Need at least 3 choices
                if k + 3 > len(incorrect_choices_candidate):
                    break
                    
                incorrect_choices = []
                need_random_fallback = False
                
                for i in range(k, k + 3):  # Take 3 consecutive choices starting from k
                    choice, score = incorrect_choices_candidate[i]
                    # Check if this choice has reached max usage
                    if choice_usage[choice.metadata['Image Path']] >= max_choice_usage:
                        need_random_fallback = True
                        break
                        
                    choice_usage[choice.metadata['Image Path']] += 1
                    if choice_usage[choice.metadata['Image Path']] >= max_choice_usage:
                        banned_IDs.add(choice.metadata['Image Path'])
                        
                    incorrect_choices.append({
                        'id': choice.metadata['Image Path'],
                        'score': float(score),
                        'metadata': choice.metadata
                    })
                
                if need_random_fallback:
                    continue
                
                # Add correct answer
                choices = incorrect_choices.copy()
                choices.append({
                    'id': row['Image Path'],
                    'score': -1.0
                })
                
                # Check if this combination has too many overlaps
                if not has_too_many_overlaps(choices, existing_choice_sets):
                    found_diverse_choices = True
                    final_choices = choices
                    break
            
            # If we couldn't find a diverse set or not enough choices, use random fallback
            if not found_diverse_choices or len(incorrect_choices_candidate) < 3:
                # Define filter criteria for random selection - SAME COUNTRY, SAME CATEGORY, DIFFERENT CONCEPT
                filter_criteria = {
                    "Country": {"$eq": country},
                    "Category": category,
                    "Concept": {"$neq": concept}
                }
                
                random_choices = get_random_choices_from_pool(
                    df, filter_criteria, choice_usage, max_choice_usage, num_choices=3, current_item=(country, category, concept)
                )
                
                if len(random_choices) < 3:
                    NOT_ENOUGH_CHOICES += 1
                    continue
                
                # Update choice usage for random choices
                for choice in random_choices:
                    choice_usage[choice['id']] += 1
                    if choice_usage[choice['id']] >= max_choice_usage:
                        banned_IDs.add(choice['id'])
                
                RANDOM_FALLBACK_COUNT += 1
                
                final_choices = random_choices.copy()
                final_choices.append({
                    'id': row['Image Path'],
                    'score': -1.0
                })
            
            # Add this choice set to our tracking set
            choice_ids = frozenset(choice['id'] for choice in final_choices)
            existing_choice_sets.add(choice_ids)
            
            # After creating final_choices but before shuffling:
            if has_duplicate_choices(final_choices):
                # Try getting new random choices
                continue

            # Only proceed if no duplicates found
            random.shuffle(final_choices)
            choice_letters = ['a', 'b', 'c', 'd']

            mcq = {
                "country": country,
                "category": category,
                "concept": concept,
                "question": question,
                "rationale": rationale,
            }

            # Store scores in a list according to choice order
            scores = []

            for _, (choice, letter) in enumerate(zip(final_choices, choice_letters)):
                if choice['score'] == -1.0:  # Correct answer
                    mcq[f"choice_{letter}"] = f"{choice['id']}"
                else:  # Incorrect choices
                    # Get the metadata for this choice
                    if 'metadata' in choice:
                        choice_metadata = choice['metadata']
                    else:
                        # For random choices
                        choice_metadata = next((c.metadata for c, s in incorrect_choices_candidate 
                                               if c.metadata['Image Path'] == choice['id']), None)
                        if choice_metadata is None:
                            # Get metadata from dataframe for random choices
                            choice_df = df[df['Image Path'] == choice['id']]
                            if not choice_df.empty:
                                choice_metadata = {
                                    'Country': choice_df['Country'].iloc[0],
                                    'Category': choice_df['Category'].iloc[0],
                                    'Concept': choice_df['Concept'].iloc[0],
                                    'Object': choice_df['Object'].iloc[0],
                                    'Image Path': choice['id']
                                }
                    
                    mcq[f"choice_{letter}"] = f"{choice['id']}"
                
                scores.append(choice['score'])
                
                if choice['score'] == -1.0:
                    mcq["correct_answer"] = f"{choice['id']}"

            mcq["scores"] = scores
            mcq["type"] = 1  # Type 1 for same country, same category

            write_question_to_jsonl(mcq, output_filename)

    print(f"Type 1 MCQs - Skipped {NOT_ENOUGH_CHOICES} questions due to insufficient choices")
    print(f"Type 1 MCQs - Used random fallback for {RANDOM_FALLBACK_COUNT} questions")

def main():
    parser = argparse.ArgumentParser(description='Generate MCQ questions using embeddings')
    parser.add_argument('--data_path', type=str, default='data.csv',
                      help='Path to the CSV data file')
    parser.add_argument('--max_questions', type=int, default=3,
                      help='Maximum questions per concept')
    parser.add_argument('--max_choice_usage', type=int, default=4,
                      help='Maximum times a choice can be used')
    parser.add_argument('--top_k', type=int, default=3,
                      help='Number of similar items to retrieve')
    parser.add_argument('--testname', type=str, default='test',
                      help='Name for the test to be included in the output filename')
    parser.add_argument('--index_path', type=str, default='faiss_index',
                      help='Path to store or load the FAISS index')
    parser.add_argument('--force', action='store_true',
                      help='Force overwrite existing output file')

    args = parser.parse_args()
    
    # Use absolute paths
    project_root = get_project_root()
    data_path = os.path.join(project_root, "data", args.data_path)
    index_path = os.path.join(project_root, "index", args.index_path)
    
    # Generate output path
    generated_mcq_dir = os.path.join(project_root, "generated_mcq")
    os.makedirs(generated_mcq_dir, exist_ok=True)
    
    output_filename = f"questions-{args.testname}.jsonl"
    output_path = os.path.join(generated_mcq_dir, output_filename)
    
    # Check if file exists, prompt for confirmation if needed
    if not args.force and os.path.exists(output_path):
        if not check_existing_file(output_filename, generated_mcq_dir):
            print("Operation cancelled by user.")
            return
    
    # Initialize or create the index
    df, vectorstore = initialize_vectorstore(data_path, index_path)
    
    # Create text column for MCQ generation if it doesn't exist
    if "text" not in df.columns:
        print("Creating 'text' column by combining Question and Rationale...")
        df["text"] = df.apply(
            lambda x: str(x["Question"])
            + " "
            + str(x["Rationale"] if pd.notna(x["Rationale"]) else ""),
            axis=1,
        )
    
    # Create empty output file
    with open(output_path, "w") as _:
        pass
    
    # Generate both types of questions
    generate_mcq_questions_v1(
        df,
        vectorstore,
        max_question_per_concept=args.max_questions,
        max_choice_usage=args.max_choice_usage,
        top_k=args.top_k,
        output_filename=output_path
    )
    
    generate_mcq_questions_v2(
        df,
        vectorstore,
        max_question_per_concept=args.max_questions,
        max_choice_usage=args.max_choice_usage,
        top_k=args.top_k,
        output_filename=output_path
    )

    print(f"MCQ generation complete. Total tokens: {total_tokens}, Total cost: ${total_cost:.4f}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    main()
