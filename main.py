import sys
import os
from PIL import Image
from tensorflow.keras.models import load_model
import aiohttp
import numpy as np
import asyncio
import json
from io import BytesIO
import keep_alive

import discord
from discord.ext import commands

version = 'custom'

TKN = os.environ['token']

#bot setup
intents = discord.Intents.all()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix='on.', intents=intents)
bot.remove_command('help')

loaded_model = load_model('model.h5', compile=False)

with open('classes.json', 'r') as f:
  classes = json.load(f)

@bot.event
async def on_ready():
  print(f"{'='*40}")
  print(f"{'The OutN Project':^40}")
  print(f"{'='*40}")
  print(f"{'Version:':<10} {version}")
  print(f"{'GitHub:':<10} {'https://github.com/Pranjal-SB/OutN'}")
  print()
  print(f"{'Logged in as':<10} {bot.user.name}#{bot.user.discriminator}")
  print(f"{'Bot User ID:':<10} {bot.user.id}")
  print(f"{'='*40}")
  await bot.change_presence(status=discord.Status.online, activity=discord.Game("Pokémon"))



@bot.event
async def on_message(message):
  if message.author.id == 716390085896962058 and len(message.embeds) > 0:
    embed = message.embeds[0]
    if "appeared!" in embed.title and embed.image:
      url = embed.image.url
      async with aiohttp.ClientSession() as session:
        async with session.get(url=url) as resp:
          if resp.status == 200:
            content = await resp.read()
            image_data = BytesIO(content)
            image = Image.open(image_data)
      preprocessed_image = await preprocess_image(image)
      predictions = loaded_model.predict(preprocessed_image)
      classes_x = np.argmax(predictions, axis=1)
      name = list(classes.keys())[classes_x[0]]
      await message.channel.send(f"__**{name}**__ Spawned | catch using:")
      await message.channel.send(f"<@716390085896962058> c {name}")


async def preprocess_image(image):
  image = image.resize((64, 64))
  image = np.array(image)
  image = image / 255.0
  image = np.expand_dims(image, axis=0)
  return image

bot.run(TKN)
