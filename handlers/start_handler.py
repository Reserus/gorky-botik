from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.exceptions import TelegramBadRequest

router = Router()

def get_start_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔍 Найти объект по названию", callback_data="search_by_name")],
            [InlineKeyboardButton(text="📍 Найти объекты рядом", callback_data="search_by_location")],
            [InlineKeyboardButton(text="🧭 Показать все категории", callback_data="show_categories")],
            [InlineKeyboardButton(text="🗺️ Построить маршрут", callback_data="build_route")]
        ]
    )

@router.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Привет! 👋 Я помогу найти объекты поблизости или по названию.\nВыберите действие:",
        reply_markup=get_start_keyboard()
    )

@router.callback_query(F.data == "start")
async def start_callback(callback_query: CallbackQuery):
    try:
        await callback_query.message.edit_text(
            "Привет! 👋 Я помогу найти объекты поблизости или по названию.\nВыберите действие:",
            reply_markup=get_start_keyboard()
        )
    except TelegramBadRequest as e:
        # Ignore the error if the message is not modified
        if "message is not modified" not in str(e):
            raise
    await callback_query.answer()
