from gpytranslate import Translator

from os import getenv
from tiktoken import encoding_for_model

from db import DataBase
from openaitools import OpenAiTools
from stablediffusion import StableDiffusion
from cryptopay import CryptoPay

from dotenv import load_dotenv

import asyncio
import os

from fastapi import FastAPI, Request
import uvicorn
from aiohttp import web

from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.client.bot import DefaultBotProperties
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, Update
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.types.input_file import BufferedInputFile
from aiogram import F
from typing import List, Tuple

class States(StatesGroup):
    ENTRY_STATE = State()
    CHATGPT_STATE = State()
    DALL_E_STATE = State()
    STABLE_STATE = State()
    INFO_STATE = State()
    PURCHASE_STATE = State()
    PURCHASE_CHATGPT_STATE = State()
    PURCHASE_DALL_E_STATE = State()
    PURCHASE_STABLE_STATE = State()

dp = Dispatcher()
app = FastAPI()

# Starts a conversation
@dp.message(Command('start'))
@dp.message(States.ENTRY_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.CHATGPT_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.DALL_E_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.STABLE_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.INFO_STATE, F.text.regexp(r'^🔙Back$'))
async def start(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    result = await DataBase.is_user(user_id)
    button = [[KeyboardButton(text="💭Chatting — ChatGPT-4o")],
              [KeyboardButton(text="🌄Image generation — DALL·E 3")],
              [KeyboardButton(text="🌅Image generation — Stable Diffusion 3")],
              [KeyboardButton(text="👤My account | 💰Buy")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )
    await DataBase.delete_messages(user_id)
    if not result:
        await DataBase.insert_user(user_id)
        await message.answer(
            text = "👋You have: \n💭3000 ChatGPT tokens \n🌄3 DALL·E Image generations \n🌅3 Stable Diffusion Image generations\n Choose an option: 👇 \n If buttons don't work, enter /start command",
            reply_markup=reply_markup,
        )
    else:
        await message.answer(
            text = "Choose an option: 👇🏻 \n If buttons don't work, enter /start command",
            reply_markup=reply_markup,
        )
    await state.set_state(States.ENTRY_STATE)

# Question Handling
@dp.message(States.ENTRY_STATE, F.text.regexp(r'^💭Chatting — ChatGPT-4o$'))
@dp.message(States.ENTRY_STATE, F.text.regexp(r'^🌄Image generation — DALL·E 3$'))
@dp.message(States.ENTRY_STATE, F.text.regexp(r'^🌅Image generation — Stable Diffusion 3$'))
async def question_handler(message: types.Message, state: FSMContext):
    button = [[KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )
    await message.answer(
        text = "Enter your text: 👇🏻",
        reply_markup=reply_markup,
    )
    option = message.text
    if option == "💭Chatting — ChatGPT-4o":
        await state.set_state(States.CHATGPT_STATE)
    elif option == "🌄Image generation — DALL·E 3":
        await state.set_state(States.DALL_E_STATE)
    elif option == "🌅Image generation — Stable Diffusion 3":
        await state.set_state(States.STABLE_STATE)

async def reduce_messages(messages: List[Tuple[int, str, str, int]]) -> Tuple[int, int]:
    question_tokens = 0
    i = len(messages) - 1
    while i >= 0:
        if question_tokens + messages[i][3] < 128000:
            question_tokens += messages[i][3]
        else:
            break
        i -= 1
    for j in range(i+1):
        await DataBase.delete_message(messages[i][0])
    return i+1, question_tokens

# Answer Handling
@dp.message(States.CHATGPT_STATE, F.text)
async def chatgpt_answer_handler(message: types.Message, state: FSMContext):
    button = [[KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )

    user_id = message.from_user.id
    result = await DataBase.get_chatgpt(user_id)

    if result > 0:
        await DataBase.save_message(user_id, "user", message.text, len(await asyncio.get_running_loop().run_in_executor(None, encoding.encode, message.text)))

        messages = await DataBase.get_messages(user_id)

        start, question_tokens = await reduce_messages(messages)

        answer = await OpenAiTools.get_chatgpt(start, messages)

        if answer:
            answer_tokens = len(await asyncio.get_running_loop().run_in_executor(None, encoding.encode,answer))
            await DataBase.save_message(user_id, "assistant", answer, answer_tokens)

            result -= int(question_tokens*0.25 + answer_tokens)

            if result > 0:
                await DataBase.set_chatgpt(user_id, result)
            else:
                await DataBase.set_chatgpt(user_id, 0)

            await message.answer(
                text = answer,
                reply_markup=reply_markup,
            )
        else:
            await DataBase.delete_message(messages[-1][0])
            await message.answer(
                text = "❌Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.",
                reply_markup=reply_markup,
            )

    else:
        await message.answer(
            text = "❎You have 0 ChatGPT tokens. You need to buy them to use ChatGPT.",
            reply_markup=reply_markup,
        )
    await state.set_state(States.CHATGPT_STATE)


# Answer Handling
@dp.message(States.DALL_E_STATE, F.text)
async def dall_e_answer_handler(message: types.Message, state: FSMContext):
    button = [[KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )

    user_id = message.from_user.id
    result = await DataBase.get_dalle(user_id)

    if result > 0:
        question = message.text

        prompt = await translator.translate(question, targetlang='en')

        answer = await OpenAiTools.get_dalle(prompt.text)

        if answer:
            result -= 1
            await DataBase.set_dalle(user_id, result)
            await message.answer_photo(
                photo=answer,
                reply_markup=reply_markup,
                caption=question,
            )
        else:
            await message.answer(
                text = "❌Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.",
                reply_markup=reply_markup,
            )
    else:
        await message.answer(
            text = "❎You have 0 DALL·E image generations. You need to buy them to use DALL·E.",
            reply_markup=reply_markup,
        )
    await state.set_state(States.DALL_E_STATE)


# Answer Handling
@dp.message(States.STABLE_STATE, F.text)
async def stable_answer_handler(message: types, state: FSMContext):
    button = [[KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )

    user_id = message.from_user.id
    result = await DataBase.get_stable(user_id)

    if result > 0:

        question = message.text

        prompt = await translator.translate(question, targetlang='en')

        photo = await StableDiffusion.get_stable(prompt.text)

        if photo:
            result -= 1
            await DataBase.set_stable(user_id, result)
            await message.answer_photo(
                photo=BufferedInputFile(photo, 'image.jpeg'),
                reply_markup=reply_markup,
                caption=question,
            )
        else:
            await message.answer(
                text = "❌Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.",
                reply_markup=reply_markup,
            )
    else:
        await message.answer(
            text = "❎You have 0 Stable Diffusion image generations. You need to buy them to use Stable Diffusion.",
            reply_markup=reply_markup,
        )
    await state.set_state(States.STABLE_STATE)


# Displays information about user
@dp.message(States.ENTRY_STATE, F.text.regexp(r'^👤My account | 💰Buy$'))
@dp.message(States.PURCHASE_STATE, F.text.regexp(r'^🔙Back$'))
async def display_info(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    result = await DataBase.get_userinfo(user_id)

    button = [[KeyboardButton(text="💰Buy tokens and generations")], [KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )
    await message.answer(
        text = f"You have: \n 💭{result[0]} ChatGPT tokens \n 🌄{result[1]} DALL·E image generations \n 🌅{result[2]} Stable Diffusion image generations \n 💸 You can buy more with crypto",
        reply_markup=reply_markup,
    )
    await state.set_state(States.INFO_STATE)


# Displays goods
@dp.message(States.INFO_STATE, F.text.regexp(r'^💰Buy tokens and generations$'))
@dp.message(States.PURCHASE_CHATGPT_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.PURCHASE_DALL_E_STATE, F.text.regexp(r'^🔙Back$'))
@dp.message(States.PURCHASE_STABLE_STATE, F.text.regexp(r'^🔙Back$'))
async def purchase(message: types.Message, state: FSMContext):
    button = [[KeyboardButton(text="100K ChatGPT tokens - 5 USD💵")],
              [KeyboardButton(text="50 DALL·E image generations - 5 USD💵")],
              [KeyboardButton(text="50 Stable Diffusion image generations - 5 USD💵")],
              [KeyboardButton(text="🔙Back")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard = button, resize_keyboard=True
    )
    await message.answer(
        text = "Choose product: 👇",
        reply_markup=reply_markup,
    )
    await state.set_state(States.PURCHASE_STATE)


# Displays cryptocurrencies
@dp.message(States.PURCHASE_STATE, F.text.regexp(r'^100K ChatGPT tokens - 5 USD💵$'))
@dp.message(States.PURCHASE_STATE, F.text.regexp(r'^50 DALL·E image generations - 5 USD💵$'))
@dp.message(States.PURCHASE_STATE, F.text.regexp(r'^50 Stable Diffusion image generations - 5 USD💵$'))
async def currencies(message: types.Message, state: FSMContext):
    buttons = [
        [KeyboardButton(text="💲USDT"),
        KeyboardButton(text="💲TON")],
        [KeyboardButton(text="💲BTC"),
        KeyboardButton(text="💲ETH")],
        [KeyboardButton(text="🔙Back")]
    ]
    keyboard = ReplyKeyboardMarkup(
        keyboard = buttons,
        resize_keyboard=True
    )
    await message.answer(
        text = "Choose currency: 👇",
        reply_markup=keyboard,
    )
    product = message.text
    if product == "100K ChatGPT tokens - 5 USD💵":
        await state.set_state(States.PURCHASE_CHATGPT_STATE)
    elif product == "50 DALL·E image generations - 5 USD💵":
        await state.set_state(States.PURCHASE_DALL_E_STATE)
    elif product == "50 Stable Diffusion image generations - 5 USD💵":
        await state.set_state(States.PURCHASE_STABLE_STATE)

# Makes invoice and displays it
@dp.message(States.PURCHASE_CHATGPT_STATE, F.text.regexp(r'^💲USDT$'))
@dp.message(States.PURCHASE_CHATGPT_STATE, F.text.regexp(r'^💲TON$'))
@dp.message(States.PURCHASE_CHATGPT_STATE, F.text.regexp(r'^💲BTC$'))
@dp.message(States.PURCHASE_CHATGPT_STATE, F.text.regexp(r'^💲ETH$'))
@dp.message(States.PURCHASE_DALL_E_STATE, F.text.regexp(r'^💲USDT$'))
@dp.message(States.PURCHASE_DALL_E_STATE, F.text.regexp(r'^💲TON$'))
@dp.message(States.PURCHASE_DALL_E_STATE, F.text.regexp(r'^💲BTC$'))
@dp.message(States.PURCHASE_DALL_E_STATE, F.text.regexp(r'^💲ETH$'))
@dp.message(States.PURCHASE_STABLE_STATE, F.text.regexp(r'^💲USDT$'))
@dp.message(States.PURCHASE_STABLE_STATE, F.text.regexp(r'^💲TON$'))
@dp.message(States.PURCHASE_STABLE_STATE, F.text.regexp(r'^💲BTC$'))
@dp.message(States.PURCHASE_STABLE_STATE, F.text.regexp(r'^💲ETH$'))
async def buy(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    currency = message.text
    invoice_url, invoice_id = await CryptoPay.create_invoice(5, currency[1:])
    current_state = await state.get_state()
    product = ''
    if current_state == States.PURCHASE_CHATGPT_STATE:
        product = '100K ChatGPT tokens - 5 USD💵'
        await DataBase.new_order(invoice_id, user_id, 'chatgpt')
    elif current_state == States.PURCHASE_DALL_E_STATE:
        product = '50 DALL·E image generations - 5 USD💵'
        await DataBase.new_order(invoice_id, user_id, 'dall_e')
    elif current_state == States.PURCHASE_STABLE_STATE:
        product = '50 Stable Diffusion image generations - 5 USD💵'
        await DataBase.new_order(invoice_id, user_id, 'stable')
    keyboard = InlineKeyboardMarkup(
        inline_keyboard = [[InlineKeyboardButton(text="💰Buy", url=invoice_url)]]
    )
    await message.answer(
        text = f"🪙Product: {product} \n 💳If you want to pay click the button 'Buy', click button 'Start' in Crypto Bot and follow the instructions \n ❗Consider the network commission",
        reply_markup=keyboard,
    )

# Processes message
@app.post("/webhook/{token}")
async def bot_webhook(token: str, request: Request):
    if token != getenv("TELEGRAM_BOT_TOKEN"):  # Проверяем токен в URL
        return {"status": "unauthorized"}, 403

    update = types.Update(**await request.json())
    await dp.feed_webhook_update(bot, update)
    return {"status": "ok"}


# Checks payment
@app.post("/payment/webhook")
async def payments_webhook(request: Request):
    # Получаем ключ из переменной окружения
    cryptopay_key = getenv("CRYPTOPAY_KEY")

    # Проверяем наличие и валидность ключа, если нужно
    if not cryptopay_key:
        return {"status": "error", "message": "Missing CRYPTOPAY_KEY"}, 400

    try:
        # Получаем данные из запроса
        data = await request.json()
        update_type = data.get('update_type')

        # Обрабатываем оплату, если статус - invoice_paid
        if update_type == "invoice_paid":
            invoice = data['payload']
            invoice_id = invoice.get('invoice_id')

            # Получаем информацию о заказе
            result = await DataBase.get_orderdata(invoice_id)

            if not result:
                return {"status": "error", "message": "Order not found"}, 404

            user_id, product = result

            # Проверяем продукт и обновляем соответствующий баланс
            if product == 'chatgpt':
                await DataBase.update_chatgpt(user_id, invoice_id)
                await bot.send_message(user_id, "✅You have received 100000 ChatGPT tokens!")
            elif product == 'dall_e':
                await DataBase.update_dalle(user_id, invoice_id)
                await bot.send_message(user_id, "✅You have received 50 DALL·E image generations!")
            elif product == 'stable':
                await DataBase.update_stable(user_id, invoice_id)
                await bot.send_message(user_id, "✅You have received 50 Stable Diffusion image generations!")
            else:
                return {"status": "error", "message": "Unknown product"}, 400

        return {"status": "ok"}, 200
    except Exception as e:
        # Логируем ошибку для дебага
        print(f"Error handling webhook: {e}")
        return {"status": "error", "message": str(e)}, 500


async def on_startup() -> None:
    await DataBase.open_pool()
    url_webhook = f"{getenv('BASE_WEBHOOK_URL')}/webhook/{getenv('TELEGRAM_BOT_TOKEN')}"
    await bot.set_webhook(url=url_webhook)


if __name__ == '__main__':
    load_dotenv()
    translator = Translator()
    encoding = encoding_for_model("gpt-4o")

    bot = Bot(token=getenv("TELEGRAM_BOT_TOKEN"), default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    app.add_event_handler("startup", on_startup)

    uvicorn.run(app, host=getenv("0.0.0.0"), port=int(os.environ.get("PORT", 5000)))
