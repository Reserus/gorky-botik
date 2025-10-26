from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

def get_paginated_keyboard(current_page: int, total_pages: int, callback_prefix: str, payload: str = ""):
    keyboard = []
    row = []

    MAX_BUTTONS = 5

    if total_pages > 1:
        if current_page > 1:
            row.append(InlineKeyboardButton(text="<<", callback_data=f"{callback_prefix}:1:{payload}"))
            row.append(InlineKeyboardButton(text="<", callback_data=f"{callback_prefix}:{current_page - 1}:{payload}"))

        start_page = max(1, current_page - MAX_BUTTONS // 2)
        end_page = min(total_pages, start_page + MAX_BUTTONS - 1)
        if end_page - start_page + 1 < MAX_BUTTONS:
            start_page = max(1, end_page - MAX_BUTTONS + 1)

        for page in range(start_page, end_page + 1):
            if page == current_page:
                row.append(InlineKeyboardButton(text=f"[{page}]", callback_data="noop"))
            else:
                row.append(InlineKeyboardButton(text=str(page), callback_data=f"{callback_prefix}:{page}:{payload}"))

        if current_page < total_pages:
            row.append(InlineKeyboardButton(text=">", callback_data=f"{callback_prefix}:{current_page + 1}:{payload}"))
            row.append(InlineKeyboardButton(text=">>", callback_data=f"{callback_prefix}:{total_pages}:{payload}"))
        
        keyboard.append(row)


    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
