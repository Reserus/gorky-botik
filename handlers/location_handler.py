import logging
from aiogram import F, Router, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from pandas import DataFrame
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from geopy.geocoders import Nominatim

from utils.categories import CATEGORIES
from utils.geo_utils import calculate_distance
from utils.paginator import get_paginated_keyboard

router = Router()

RADIUS_KM = 2 # Default, will be overwritten by user input

ITEMS_PER_PAGE = 5

class LocationStates(StatesGroup):
    waiting_for_address = State()
    waiting_for_radius = State()
    waiting_for_location_input = State() # New state to differentiate between radius and location input

@router.callback_query(F.data == "search_by_location")
async def search_by_location_callback(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите желаемый радиус поиска в километрах (например, 5 для 5 км).")
    await state.set_state(LocationStates.waiting_for_radius)
    await callback_query.answer()

@router.message(LocationStates.waiting_for_radius)
async def process_radius(message: Message, state: FSMContext):
    try:
        radius = float(message.text)
        if radius <= 0:
            await message.answer("Радиус должен быть положительным числом. Попробуйте еще раз.")
            return
        await state.update_data(radius=radius)
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Отправить геолокацию", callback_data="send_geolocation_with_radius")],
                [InlineKeyboardButton(text="Ввести адрес вручную", callback_data="enter_address_with_radius")]
            ]
        )
        await message.answer("Выберите способ поиска:", reply_markup=keyboard)
        await state.set_state(LocationStates.waiting_for_location_input)
    except ValueError:
        await message.answer("Пожалуйста, введите радиус числом.")

@router.callback_query(F.data == "send_geolocation_with_radius")
async def send_geolocation_with_radius_callback(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Отправьте свою геолокацию, чтобы найти объекты поблизости.")
    await callback_query.answer()

@router.callback_query(F.data == "enter_address_with_radius")
async def enter_address_with_radius_callback(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите адрес:")
    await state.set_state(LocationStates.waiting_for_address)
    await callback_query.answer()

@router.callback_query(F.data == "send_geolocation")
async def send_geolocation_callback(callback_query: types.CallbackQuery):
    await callback_query.message.answer("Отправьте свою геолокацию, чтобы найти объекты поблизости.")
    await callback_query.answer()

@router.callback_query(F.data == "enter_address")
async def enter_address_callback(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Введите адрес:")
    await state.set_state(LocationStates.waiting_for_address)
    await callback_query.answer()

import logging

# ... (the rest of the imports)

@router.message(LocationStates.waiting_for_address)
async def process_address(message: Message, state: FSMContext, df: DataFrame):
    await state.clear()
    logging.info(f"User entered address: {message.text}")
    geolocator = Nominatim(user_agent="telegram-tourist-bot")
    try:
        address = message.text
        if "нижний новгород" not in address.lower():
            address += ", Нижний Новгород"
        logging.info(f"Geocoding address: {address}")
        location = geolocator.geocode(address)
        logging.info(f"Geocoding result: {location}")
        if location:
            await process_location(message, state, df, location.latitude, location.longitude, 1)
        else:
            await message.answer("❌ Не удалось найти координаты для данного адреса.")
    except Exception as e:
        logging.error(f"Error processing address: {e}")
        await message.answer(f"Произошла ошибка при обработке адреса: {e}")

@router.message(F.location)
async def process_location_message(message: Message, state: FSMContext, df: DataFrame):
    await process_location(message, state, df, message.location.latitude, message.location.longitude, 1)

async def process_location(message: Message, state: FSMContext, df: DataFrame, user_lat: float, user_lon: float, page: int = 1, message_to_edit: types.Message = None):
    data = await state.get_data()
    radius_km = data.get("radius", RADIUS_KM) # Use user-defined radius or default

    df["distance"] = df.apply(
        lambda row: calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']),
        axis=1
    )

    results = df[df["distance"] <= radius_km].sort_values(by="distance")

    if results.empty:
        if message_to_edit:
            await message_to_edit.edit_text(f"❌ Объекты в радиусе {radius_km} км не найдены")
        else:
            await message.answer(f"❌ Объекты в радиусе {radius_km} км не найдены")
        return

    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    paginated_results = results.iloc[start_index:end_index]

    response = ""
    for _, row in paginated_results.iterrows():
        response += (
            f"🏷 <b>Название:</b> {row['title']}\n"
            f"📍 <b>Адрес:</b> {row['address']}\n"
            f"🗂 <b>Категория:</b> {CATEGORIES.get(row['category_id'], row['category_id'])}\n"
            f"📏 <b>Расстояние:</b> {row['distance']:.2f} км\n\n"
        )

    total_pages = (len(results) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    payload = f"{user_lat}:{user_lon}:{radius_km}"
    keyboard = get_paginated_keyboard(page, total_pages, "location_page", payload)
    if keyboard.inline_keyboard:
        keyboard.inline_keyboard.append([InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")])
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")]])

    if message_to_edit:
        await message_to_edit.edit_text(response, reply_markup=keyboard)
    else:
        await message.answer(response, reply_markup=keyboard)

@router.callback_query(F.data.startswith("location_page:"))
async def location_page_callback(callback_query: types.CallbackQuery, state: FSMContext, df: DataFrame):
    _, page, lat_lon_radius = callback_query.data.split(":", 2)
    page = int(page)
    lat, lon, radius = lat_lon_radius.split(":")
    lat = float(lat)
    lon = float(lon)
    radius = float(radius)

    await state.update_data(radius=radius) # Restore radius to state for process_location
    await callback_query.answer()
    await process_location(callback_query.message, state, df, lat, lon, page, message_to_edit=callback_query.message)
