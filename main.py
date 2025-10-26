import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from sentence_transformers import SentenceTransformer

from config import BOT_TOKEN
from handlers import start_handler, search_handler, category_handler, location_handler, route_handler
from utils.data_loader import load_data, MODEL_NAME

async def main():
    logging.basicConfig(level=logging.INFO)
    from aiogram.client.default import DefaultBotProperties

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)

    # Загрузка данных и модели
    df, embeddings = load_data("data/cultural_objects_mnn.xlsx")
    sbert_model = SentenceTransformer(MODEL_NAME)

    dp["df"] = df
    dp["embeddings"] = embeddings
    dp["sbert_model"] = sbert_model

    # Подключение роутеров
    dp.include_router(start_handler.router)
    dp.include_router(search_handler.router)
    dp.include_router(category_handler.router)
    dp.include_router(location_handler.router)
    dp.include_router(route_handler.router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
