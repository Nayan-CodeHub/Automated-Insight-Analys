import asyncio
import io
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

OUTPUT_PATH = Path(__file__).parent / "demo_video.mp4"


async def capture_demo_frames():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        await page.goto("http://localhost:8501", wait_until="networkidle", timeout=120000)
        await page.wait_for_timeout(1500)

        actions = [
            ("Dashboard", None),
            ("ML Insights", "text=🤖 ML Insights"),
            ("Data Quality", "text=🧹 Data Quality"),
            ("Data Preview", "text=🔎 Data Preview"),
            ("Clean Data", "text=🧽 Clean Data"),
            ("Dashboard Final", "text=📊 Dashboard"),
        ]

        frames = []
        for _, selector in actions:
            if selector:
                await page.click(selector)
                await page.wait_for_timeout(1200)
            screenshot = await page.screenshot(type="png")
            image = Image.open(io.BytesIO(screenshot)).convert("RGB")
            frames.append(np.asarray(image))

        await browser.close()
        return frames


async def main():
    frames = await capture_demo_frames()
    if not frames:
        raise RuntimeError("No frames were captured for the demo video.")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    writer = imageio.get_writer(str(OUTPUT_PATH), fps=1.5, codec="libx264")
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Created demo video: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size} bytes")


asyncio.run(main())
