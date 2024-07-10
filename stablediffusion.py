from os import getenv
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()
key = getenv("STABLE_DIFFUSION_API_KEY")

class StableDiffusion:
    def get_stable(prompt: str):
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "authorization": f"Bearer {key}",
                "accept": "image/*"
            },
            files={"none": ''},
            data={
                "prompt": prompt,
                "output_format": "jpeg",
            },
        )

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img_path = str(response.seed) + ".png"
            img.save(img_path)

            return img_path
        else:
            return