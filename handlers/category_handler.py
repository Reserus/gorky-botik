from aiogram import F, Router, types
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from pandas import DataFrame

from utils.paginator import get_paginated_keyboard
from utils.categories import CATEGORIES

router = Router()

ITEMS_PER_PAGE = 5

@router.callback_query(F.data == "show_categories")
async def show_categories_callback(callback_query: types.CallbackQuery, df: DataFrame):
    categories = df['category_id'].unique()
    buttons = []
    for category in sorted(categories):
        buttons.append([InlineKeyboardButton(text=CATEGORIES.get(category, f"Категория {category}"), callback_data=f"category:{category}")])
    
    buttons.append([InlineKeyboardButton(text="🔙 Назад", callback_data="start")])
    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await callback_query.message.edit_text("Выберите категорию:", reply_markup=keyboard)
    await callback_query.answer()

@router.callback_query(F.data.startswith("category:"))
async def category_callback(callback_query: types.CallbackQuery, df: DataFrame):
    category_id = int(callback_query.data.split(":")[1])
    results = df[df['category_id'] == category_id]

    if results.empty:
        await callback_query.answer("❌ Объекты в этой категории не найдены", show_alert=True)
        return

    await display_category_results(callback_query.message, results, 1, category_id)
    await callback_query.answer()

async def display_category_results(message: Message, results: DataFrame, page: int, category_id: int):
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
    keyboard = get_paginated_keyboard(page, total_pages, f"category_page", str(category_id))
    if keyboard.inline_keyboard:
        keyboard.inline_keyboard.append([InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")])
    else:
        keyboard = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="🔙 В главное меню", callback_data="start")]])

    await message.edit_text(response, reply_markup=keyboard)

@router.callback_query(F.data.startswith("category_page:"))
async def category_page_callback(callback_query: types.CallbackQuery, df: DataFrame):
    _, page, category_id = callback_query.data.split(":", 2)
    page = int(page)
    category_id = int(category_id)

    results = df[df['category_id'] == category_id]

    await display_category_results(callback_query.message, results, page, category_id)
    await callback_query.answer()
