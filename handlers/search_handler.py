from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from pandas import DataFrame
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

from utils.paginator import get_paginated_keyboard
from utils.categories import CATEGORIES

router = Router()

class SearchStates(StatesGroup):
    waiting_for_name = State()

ITEMS_PER_PAGE = 5

SYNONYMS = {
    "кофейни": ["кафе", "coffee", "кофе", "бары"],
    "музей": ["история", "галерея", "культура"],
    "стрит-арт": ["граффити", "арт", "инсталляция", "современное искусство"],
    "история": ["музей", "памятник", "исторический", "архитектура"],
    "еда": ["ресторан", "кафе", "столовая", "закусочная"],
    "парк": ["сквер", "сад", "природа", "отдых"],
    "театр": ["кино", "концерт", "представление"],
    "концерт": ["театр", "представление", "сцена"],
    "магазин": ["покупки", "супермаркет", "бутик"]
}

@router.callback_query(F.data == "search_by_name")
async def search_by_name_callback(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите название объекта (например, \"парк\", \"Панорама\"):")
    await state.set_state(SearchStates.waiting_for_name)
    await callback_query.answer()

@router.message(SearchStates.waiting_for_name)
async def process_name(message: Message, state: FSMContext, df: DataFrame, embeddings: np.ndarray, sbert_model: SentenceTransformer):
    await state.clear()
    query = message.text.lower()
    
    found_results = []

    # 1. Keyword Search (Exact Match) - highest priority
    keyword_results = df[df['title'].str.lower().str.contains(query) | df['category_id'].astype(str).str.lower().str.contains(query)]
    for _, row in keyword_results.iterrows():
        found_results.append({'item': row, 'score': 1.0, 'type': 'keyword'})

    # 2. Synonym Expansion and Search - high priority
    expanded_queries = [query]
    for key, synonyms_list in SYNONYMS.items():
        if key in query:
            expanded_queries.extend(synonyms_list)
        for synonym in synonyms_list:
            if synonym in query and synonym not in expanded_queries:
                expanded_queries.append(synonym)
    
    for exp_query in expanded_queries:
        synonym_keyword_results = df[df['title'].str.lower().str.contains(exp_query) | df['category_id'].astype(str).str.lower().str.contains(exp_query)]
        for _, row in synonym_keyword_results.iterrows():
            if not any(res['item']['id'] == row['id'] for res in found_results):
                found_results.append({'item': row, 'score': 0.95, 'type': 'synonym_keyword'}) # Slightly lower than exact keyword

    # 3. Semantic Search - medium priority
    if not found_results or len(query.split()) > 1: # Perform semantic search if no exact matches or query is multi-word
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
        
        semantic_threshold = 0.5 # Adjustable threshold
        semantic_matches_indices = np.where(cosine_scores > semantic_threshold)[0]
        
        for idx in semantic_matches_indices:
            row = df.iloc[idx]
            if not any(res['item']['id'] == row['id'] for res in found_results):
                found_results.append({'item': row, 'score': cosine_scores[idx] * 0.8, 'type': 'semantic'})

    # 4. Fuzzy Search - lower priority
    if len(found_results) < ITEMS_PER_PAGE * 2: # Only do fuzzy search if few results
        for index, row in df.iterrows():
            title_ratio = fuzz.partial_ratio(query, row['title'].lower())
            description_ratio = fuzz.partial_ratio(query, row['description'].lower() if pd.notna(row['description']) else '')
            
            fuzzy_score = max(title_ratio, description_ratio) / 100.0 # Normalize to 0-1
            fuzzy_threshold = 0.6 # Adjustable threshold

            if fuzzy_score > fuzzy_threshold:
                if not any(res['item']['id'] == row['id'] for res in found_results):
                    found_results.append({'item': row, 'score': fuzzy_score * 0.7, 'type': 'fuzzy'})

    # Sort results: exact > synonym > semantic > fuzzy, then by score descending
    found_results.sort(key=lambda x: (x['type'] == 'keyword', x['type'] == 'synonym_keyword', x['type'] == 'semantic', x['score']), reverse=True)
    
    # Remove duplicates based on 'id'
    unique_results = []
    seen_ids = set()
    for res in found_results:
        item_id = res['item']['id']
        if item_id not in seen_ids:
            unique_results.append(res['item'])
            seen_ids.add(item_id)

    if not unique_results:
        await message.answer("❌ Объекты не найдены")
        return

    await display_results(message, pd.DataFrame(unique_results), 1, query)

async def display_results(message: Message, results: DataFrame, page: int, query: str, message_to_edit: types.Message = None):
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_results = results.iloc[start_index:end_index]

    response = ""
    for _, row in paginated_results.iterrows():
        response += (
            f"🏷 <b>Название:</b> {row['title']}\n"
            f"📍 <b>Адрес:</b> {row['address']}\n"
            f"🗂 <b>Категория:</b> {CATEGORIES.get(row['category_id'], row['category_id'])}\n\n"
        )

    total_pages = (len(results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    keyboard = get_paginated_keyboard(page, total_pages, "search_page", query)
    if keyboard.inline_keyboard:
        keyboard.inline_keyboard.append([InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")])
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")]])

    if message_to_edit:
        await message_to_edit.edit_text(response, reply_markup=keyboard)
    else:
        await message.answer(response, reply_markup=keyboard)

@router.callback_query(F.data.startswith("search_page:"))
async def search_page_callback(callback_query: types.CallbackQuery, df: DataFrame, embeddings: np.ndarray, sbert_model: SentenceTransformer):
    _, page, query = callback_query.data.split(":", 2)
    page = int(page)
    
    found_results = []

    # 1. Keyword Search (Exact Match)
    keyword_results = df[df['title'].str.lower().str.contains(query) | df['category_id'].astype(str).str.lower().str.contains(query)]
    for _, row in keyword_results.iterrows():
        found_results.append({'item': row, 'score': 1.0, 'type': 'keyword'})

    # 2. Synonym Expansion and Search
    expanded_queries = [query]
    for key, synonyms_list in SYNONYMS.items():
        if key in query:
            expanded_queries.extend(synonyms_list)
        for synonym in synonyms_list:
            if synonym in query and synonym not in expanded_queries:
                expanded_queries.append(synonym)
    
    for exp_query in expanded_queries:
        synonym_keyword_results = df[df['title'].str.lower().str.contains(exp_query) | df['category_id'].astype(str).str.lower().str.contains(exp_query)]
        for _, row in synonym_keyword_results.iterrows():
            if not any(res['item']['id'] == row['id'] for res in found_results):
                found_results.append({'item': row, 'score': 0.9, 'type': 'synonym_keyword'})

    # 3. Semantic Search
    if not found_results or len(query.split()) > 1: # Perform semantic search if no exact matches or query is multi-word
        query_embedding = sbert_model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
        
        semantic_threshold = 0.5 # Adjustable threshold
        semantic_matches_indices = np.where(cosine_scores > semantic_threshold)[0]
        
        for idx in semantic_matches_indices:
            row = df.iloc[idx]
            if not any(res['item']['id'] == row['id'] for res in found_results):
                found_results.append({'item': row, 'score': cosine_scores[idx] * 0.8, 'type': 'semantic'})

    # 4. Fuzzy Search (if still not enough results or for broader matching)
    if len(found_results) < ITEMS_PER_PAGE * 2: # Only do fuzzy search if few results
        for index, row in df.iterrows():
            title_ratio = fuzz.partial_ratio(query, row['title'].lower())
            description_ratio = fuzz.partial_ratio(query, row['description'].lower() if pd.notna(row['description']) else '')
            
            fuzzy_score = max(title_ratio, description_ratio) / 100.0 # Normalize to 0-1
            fuzzy_threshold = 0.6 # Adjustable threshold

            if fuzzy_score > fuzzy_threshold:
                if not any(res['item']['id'] == row['id'] for res in found_results):
                    found_results.append({'item': row, 'score': fuzzy_score * 0.7, 'type': 'fuzzy'})

    # Sort results: exact > synonym > semantic > fuzzy, then by score descending
    found_results.sort(key=lambda x: (x['type'] == 'keyword', x['type'] == 'synonym_keyword', x['type'] == 'semantic', x['score']), reverse=True)
    
    # Remove duplicates based on 'id'
    unique_results = []
    seen_ids = set()
    for res in found_results:
        item_id = res['item']['id']
        if item_id not in seen_ids:
            unique_results.append(res['item'])
            seen_ids.add(item_id)

    results_df = pd.DataFrame(unique_results)

    await callback_query.answer() # Dismiss the loading state of the button
    await display_results(callback_query.message, results_df, page, query, message_to_edit=callback_query.message)
