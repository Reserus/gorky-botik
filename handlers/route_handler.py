import random
import logging
import io
import os
from aiogram import F, Router, types
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
from pandas import DataFrame
import numpy as np
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer, util
import folium
# from PIL import Image # PIL is not directly used for HTML to PNG conversion
import imgkit

from utils.visiting_times import VISITING_TIMES, DEFAULT_VISITING_TIME
from utils.paginator import get_paginated_keyboard
from utils.categories import CATEGORIES
from utils.geo_utils import calculate_distance

router = Router()

INTEREST_MAPPING = {
    "история": [1, 5, 7, 10],  # Памятники, архитектурные/исторические объекты, музеи, советское искусство
    "панорамы": [4],  # Набережные (могут предлагать панорамные виды)
    "панорама": [4], # Добавляем единственное число
    "музеи": [7],
    "парки": [2],
    "архитектура": [5, 10],
    "искусство": [7, 10],
    "кофейни": [], # Будет обрабатываться как ключевое слово
    "кофейня": [],  # Будет обрабатываться как ключевое слово
    "стрит-арт": [], # Будет обрабатываться как ключевое слово
    "стритарт": []   # Будет обрабатываться как ключевое слово
}

class RouteStates(StatesGroup):
    waiting_for_interests = State()
    waiting_for_time = State()
    waiting_for_location = State()

@router.callback_query(F.data == "build_route")
async def build_route_callback(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите ваши интересы через запятую (например: история, набережные, кофейня, стрит-арт). Или напишите 'все', чтобы искать по всем категориям.")
    await state.set_state(RouteStates.waiting_for_interests)
    await callback_query.answer()

@router.message(RouteStates.waiting_for_interests)
async def process_interests(message: Message, state: FSMContext, df: DataFrame, embeddings: np.ndarray, sbert_model: SentenceTransformer):
    user_input = message.text.lower().strip()
    
    if user_input == "все":
        await state.update_data(interests=set())
    else:
        input_interests = [interest.strip() for interest in user_input.split(',')]
        matched_category_ids = set()
        keyword_interests = set()
        
        for input_interest in input_interests:
            # 1. Check in INTEREST_MAPPING
            if input_interest in INTEREST_MAPPING:
                if INTEREST_MAPPING[input_interest]:
                    matched_category_ids.update(INTEREST_MAPPING[input_interest])
                else:
                    keyword_interests.add(input_interest)
            # 2. Check in CATEGORIES (exact match)
            else:
                found_in_categories = False
                for category_id, category_name in CATEGORIES.items():
                    if input_interest in category_name.lower():
                        matched_category_ids.add(category_id)
                        found_in_categories = True
                
                # 3. Semantic Search if not found in direct mappings or categories
                if not found_in_categories:
                    query_embedding = sbert_model.encode(input_interest, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
                    
                    semantic_threshold = 0.5 # Adjustable threshold
                    semantic_matches_indices = np.where(cosine_scores > semantic_threshold)[0]
                    
                    if len(semantic_matches_indices) > 0:
                        # Get category_ids from semantically matched objects
                        semantically_matched_categories = df.iloc[semantic_matches_indices]['category_id'].unique()
                        matched_category_ids.update(semantically_matched_categories)
                    else:
                        # If still not found, treat as a keyword for later filtering
                        keyword_interests.add(input_interest)
        
        if not matched_category_ids and not keyword_interests:
            await message.answer("Не удалось найти категории или ключевые слова по вашим интересам. Попробуйте еще раз или введите 'все'.")
            return
        await state.update_data(interests=matched_category_ids, keyword_interests=keyword_interests)

    await message.answer("Сколько у вас времени на прогулку (в часах)?")
    await state.set_state(RouteStates.waiting_for_time)

@router.message(RouteStates.waiting_for_time)
async def process_time(message: Message, state: FSMContext):
    try:
        time = float(message.text)
        if time <= 0:
            await message.answer("Время должно быть положительным числом. Попробуйте еще раз.")
            return
        await state.update_data(time=time)
        await message.answer("Отправьте свою геолокацию или введите адрес для начала маршрута.")
        await state.set_state(RouteStates.waiting_for_location)
    except ValueError:
        await message.answer("Пожалуйста, введите время в часах (числом).")

@router.message(RouteStates.waiting_for_location, F.location)
async def process_route_location_message(message: Message, state: FSMContext, df: DataFrame):
    await process_route_location(message, state, df, message.location.latitude, message.location.longitude)

@router.message(RouteStates.waiting_for_location, F.text)
async def process_route_address(message: Message, state: FSMContext, df: DataFrame):
    geolocator = Nominatim(user_agent="telegram-tourist-bot")
    try:
        address = message.text
        if "нижний новгород" not in address.lower():
            address += ", Нижний Новгород"
        logging.info(f"Geocoding address: {address}")
        location = geolocator.geocode(address)
        if location:
            await process_route_location(message, state, df, location.latitude, location.longitude)
        else:
            await message.answer("❌ Не удалось найти координаты для данного адреса.")
    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке адреса: {e}")

async def generate_route_map(route: list, user_lat: float, user_lon: float, map_filename: str = "route_map.png") -> str:
    if not route:
        return ""

    # Create a map centered around the starting point
    m = folium.Map(location=[user_lat, user_lon], zoom_start=13)

    # Add starting point marker (green)
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Начало маршрута",
        icon=folium.Icon(color="green", icon="play", prefix='fa')
    ).add_to(m)

    # Add route points and lines
    points = [[user_lat, user_lon]]
    for i, obj in enumerate(route):
        lat, lon = obj['latitude'], obj['longitude']
        points.append([lat, lon])
        folium.Marker(
            location=[lat, lon],
            popup=f"{i+1}. {obj['title']}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    
    folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(m)

    # Save as HTML first
    html_path = "route_map.html"
    m.save(html_path)

    # Convert HTML to PNG using imgkit
    # NOTE: imgkit requires wkhtmltopdf to be installed and accessible in your system's PATH.
    # Download from: https://wkhtmltopdf.org/downloads.html
    try:
        imgkit.from_file(html_path, map_filename)
        logging.info(f"Map saved as {map_filename}")
        os.remove(html_path) # Clean up HTML file
        return map_filename
    except Exception as e:
        logging.error(f"Error saving map as PNG using imgkit: {e}. Sending HTML instead. Please ensure wkhtmltopdf is installed and in PATH for PNG export.")
        return html_path # Return HTML path as fallback

async def display_route_page(message: Message, state: FSMContext, page: int = 1, message_to_edit: types.Message = None):
    data = await state.get_data()
    route = data.get("route", [])
    user_lat = data.get("user_lat")
    user_lon = data.get("user_lon")

    if not route:
        if message_to_edit:
            await message_to_edit.edit_text("Не удалось построить маршрут.")
        else:
            await message.answer("Не удалось построить маршрут.")
        return

    ITEMS_PER_PAGE = 5
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_route = route[start_index:end_index]

    response = "<b>Ваш маршрут:</b>\n\n"
    total_time = 0
    for i, obj in enumerate(paginated_route, start=start_index):
        # Recalculate walk_time and visit_time for display from stored values (which are in hours)
        walk_time_hours = obj['calculated_walk_time']
        visit_time_hours = obj['calculated_visit_time']
        
        total_time += walk_time_hours + visit_time_hours

        response += f"{i+1}. <b>{obj['title']} ({obj['address']})</b>\n"
        response += f"   - 🚶‍♂️ Время в пути: {walk_time_hours * 60:.0f} мин.\n"
        response += f"   - 🏛️ Время на осмотр: {visit_time_hours * 60:.0f} мин.\n"

    total_pages = (len(route) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    keyboard = get_paginated_keyboard(page, total_pages, "route_page")
    if keyboard.inline_keyboard:
        keyboard.inline_keyboard.append([InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")])
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")]])

    if message_to_edit:
        await message_to_edit.edit_text(response, reply_markup=keyboard)
    else:
        await message.answer(response, reply_markup=keyboard)

    # Generate and send map
    map_filepath_or_error = await generate_route_map(route, user_lat, user_lon)
    if map_filepath_or_error and map_filepath_or_error.endswith(".png") and os.path.exists(map_filepath_or_error):
        await message.answer_photo(photo=FSInputFile(map_filepath_or_error), caption="Карта вашего маршрута:")
        os.remove(map_filepath_or_error) # Clean up the generated map file
    elif map_filepath_or_error and map_filepath_or_error.endswith(".html") and os.path.exists(map_filepath_or_error):
        await message.answer_document(document=FSInputFile(map_filepath_or_error), caption="Не удалось сгенерировать карту маршрута в формате PNG. Отправляю HTML-файл. Для экспорта в PNG, пожалуйста, установите wkhtmltopdf и добавьте его в PATH вашей системы: https://wkhtmltopdf.org/downloads.html")
        os.remove(map_filepath_or_error) # Clean up the generated map file
    else:
        await message.answer("Не удалось сгенерировать карту маршрута по неизвестной причине.")

@router.callback_query(F.data.startswith("route_page:"))
async def route_page_callback(callback_query: types.CallbackQuery, state: FSMContext):
    page = int(callback_query.data.split(":")[1])
    await callback_query.answer()
    await display_route_page(callback_query.message, state, page, message_to_edit=callback_query.message)

async def process_route_location(message: Message, state: FSMContext, df: DataFrame, user_lat: float, user_lon: float):
    data = await state.get_data()
    interests = data.get("interests", set())
    keyword_interests = data.get("keyword_interests", set())
    time_limit = data.get("time")
    await state.update_data(user_lat=user_lat, user_lon=user_lon)

    # Filter objects by interest
    if interests:
        filtered_df = df[df['category_id'].isin(interests)].copy()
    else:
        filtered_df = df.copy()

    # Filter by keywords if present
    if keyword_interests:
        # Create a boolean mask for keyword matching
        keyword_mask = False
        for keyword in keyword_interests:
            keyword_mask |= filtered_df['title'].str.lower().str.contains(keyword, na=False)
            # Assuming 'description' column might exist and be relevant
            if 'description' in filtered_df.columns:
                keyword_mask |= filtered_df['description'].str.lower().str.contains(keyword, na=False)
        filtered_df = filtered_df[keyword_mask]

    if filtered_df.empty:
        await message.answer("Не найдено объектов по вашим интересам.")
        return

    # Build the route
    route = []
    remaining_objects = filtered_df.copy()
    current_lat, current_lon = user_lat, user_lon
    remaining_time = time_limit

    while not remaining_objects.empty:
        remaining_objects['distance'] = remaining_objects.apply(
            lambda row: calculate_distance(current_lat, current_lon, row['latitude'], row['longitude']),
            axis=1
        )
        closest_object = remaining_objects.sort_values(by="distance").iloc[0]
        
        distance_to_object = closest_object['distance']
        # Assuming average walking speed between 4 and 5 km/h
        walking_speed_ms = random.uniform(66.7, 83.3) # meters per minute
        time_to_walk = ((distance_to_object * 1000) / walking_speed_ms) / 60 # convert minutes to hours

        visit_time = VISITING_TIMES.get(closest_object['category_id'], DEFAULT_VISITING_TIME) / 60 # convert minutes to hours
        total_time_for_object = time_to_walk + visit_time

        if remaining_time - total_time_for_object >= 0:
            # Convert the pandas Series to a dictionary before appending
            obj_dict = closest_object.to_dict()
            obj_dict['calculated_walk_time'] = time_to_walk
            obj_dict['calculated_visit_time'] = visit_time
            route.append(obj_dict)
            remaining_time -= total_time_for_object
            current_lat, current_lon = closest_object['latitude'], closest_object['longitude']
            remaining_objects = remaining_objects[remaining_objects['id'] != closest_object['id']]
        else:
            break

    if not route:
        await message.answer("Не удалось построить маршрут с учетом вашего времени.")
        return

    await state.update_data(route=route)
    await display_route_page(message, state, 1)

@router.message(RouteStates.waiting_for_time)
async def process_time(message: Message, state: FSMContext):
    try:
        time = float(message.text)
        if time <= 0:
            await message.answer("Время должно быть положительным числом. Попробуйте еще раз.")
            return
        await state.update_data(time=time)
        await message.answer("Отправьте свою геолокацию или введите адрес для начала маршрута.")
        await state.set_state(RouteStates.waiting_for_location)
    except ValueError:
        await message.answer("Пожалуйста, введите время в часах (числом).")

@router.message(RouteStates.waiting_for_location, F.location)
async def process_route_location_message(message: Message, state: FSMContext, df: DataFrame):
    await process_route_location(message, state, df, message.location.latitude, message.location.longitude)

@router.message(RouteStates.waiting_for_location, F.text)
async def process_route_address(message: Message, state: FSMContext, df: DataFrame):
    geolocator = Nominatim(user_agent="telegram-tourist-bot")
    try:
        address = message.text
        if "нижний новгород" not in address.lower():
            address += ", Нижний Новгород"
        logging.info(f"Geocoding address: {address}")
        location = geolocator.geocode(address)
        if location:
            await process_route_location(message, state, df, location.latitude, location.longitude)
        else:
            await message.answer("❌ Не удалось найти координаты для данного адреса.")
    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке адреса: {e}")

async def display_route_page(message: Message, state: FSMContext, page: int = 1, message_to_edit: types.Message = None):
    data = await state.get_data()
    route = data.get("route", [])
    user_lat = data.get("user_lat")
    user_lon = data.get("user_lon")

    if not route:
        if message_to_edit:
            await message_to_edit.edit_text("Не удалось построить маршрут.")
        else:
            await message.answer("Не удалось построить маршрут.")
        return

    ITEMS_PER_PAGE = 5
    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_route = route[start_index:end_index]

    response = "<b>Ваш маршрут:</b>\n\n"
    total_time = 0
    for i, obj in enumerate(paginated_route, start=start_index):
        if i == 0:
            dist = calculate_distance(user_lat, user_lon, obj['latitude'], obj['longitude'])
        else:
            dist = calculate_distance(route[i-1]['latitude'], route[i-1]['longitude'], obj['latitude'], obj['longitude'])
        
        walking_speed_ms = random.uniform(66.7, 83.3)
        walk_time = (dist * 1000) / walking_speed_ms
        visit_time = VISITING_TIMES.get(obj['category_id'], DEFAULT_VISITING_TIME)
        total_time += walk_time + visit_time

        response += f"{i+1}. <b>{obj['title']} ({obj['address']})</b>\n"
        response += f"   - 🚶‍♂️ Время в пути: {obj['calculated_walk_time'] * 60:.0f} мин.\n"
        response += f"   - 🏛️ Время на осмотр: {obj['calculated_visit_time'] * 60:.0f} мин.\n"

    total_pages = (len(route) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    keyboard = get_paginated_keyboard(page, total_pages, "route_page")
    if keyboard.inline_keyboard:
        keyboard.inline_keyboard.append([InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")])
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")]])



    if message_to_edit:
        await message_to_edit.edit_text(response, reply_markup=keyboard)
    else:
        await message.answer(response, reply_markup=keyboard)

@router.callback_query(F.data.startswith("route_page:"))
async def route_page_callback(callback_query: types.CallbackQuery, state: FSMContext):
    page = int(callback_query.data.split(":")[1])
    await callback_query.answer()
    await display_route_page(callback_query.message, state, page, message_to_edit=callback_query.message)

async def process_route_location(message: Message, state: FSMContext, df: DataFrame, user_lat: float, user_lon: float):
    data = await state.get_data()
    interests = data.get("interests", set())
    keyword_interests = data.get("keyword_interests", set())
    time_limit = data.get("time")
    await state.update_data(user_lat=user_lat, user_lon=user_lon)

    # Filter objects by interest
    if interests:
        filtered_df = df[df['category_id'].isin(interests)].copy()
    else:
        filtered_df = df.copy()

    # Filter by keywords if present
    if keyword_interests:
        # Create a boolean mask for keyword matching
        keyword_mask = False
        for keyword in keyword_interests:
            keyword_mask |= filtered_df['title'].str.lower().str.contains(keyword, na=False)
            # Assuming 'description' column might exist and be relevant
            if 'description' in filtered_df.columns:
                keyword_mask |= filtered_df['description'].str.lower().str.contains(keyword, na=False)
        filtered_df = filtered_df[keyword_mask]

    if filtered_df.empty:
        await message.answer("Не найдено объектов по вашим интересам.")
        return

    # Build the route
    route = []
    remaining_objects = filtered_df.copy()
    current_lat, current_lon = user_lat, user_lon
    remaining_time = time_limit

    while not remaining_objects.empty:
        remaining_objects['distance'] = remaining_objects.apply(
            lambda row: calculate_distance(current_lat, current_lon, row['latitude'], row['longitude']),
            axis=1
        )
        closest_object = remaining_objects.sort_values(by="distance").iloc[0]
        
        distance_to_object = closest_object['distance']
        # Assuming average walking speed between 4 and 5 km/h
        walking_speed_ms = random.uniform(66.7, 83.3) # meters per minute
        time_to_walk = ((distance_to_object * 1000) / walking_speed_ms) / 60 # convert minutes to hours

        visit_time = VISITING_TIMES.get(closest_object['category_id'], DEFAULT_VISITING_TIME) / 60 # convert minutes to hours
        total_time_for_object = time_to_walk + visit_time

        if remaining_time - total_time_for_object >= 0:
            # Convert the pandas Series to a dictionary before appending
            obj_dict = closest_object.to_dict()
            obj_dict['calculated_walk_time'] = time_to_walk
            obj_dict['calculated_visit_time'] = visit_time
            route.append(obj_dict)
            remaining_time -= total_time_for_object
            current_lat, current_lon = closest_object['latitude'], closest_object['longitude']
            remaining_objects = remaining_objects[remaining_objects['id'] != closest_object['id']]
        else:
            break

    if not route:
        await message.answer("Не удалось построить маршрут с учетом вашего времени.")
        return

    await state.update_data(route=route)
    await display_route_page(message, state, 1)