"""Read this table's visible replay state from the already connected Edge."""
import asyncio
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.async_api import async_playwright
from splendor.bga_live import SNAPSHOT_JS
from splendor.bga_watch import cdp_endpoint

OUT = Path(__file__).resolve().parent


async def main():
    found = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(
            cdp_endpoint(profile=Path.home() / '.config/microsoft-edge'),
            no_defaults=True, timeout=5000)
        for context in browser.contexts:
            for page in context.pages:
                for frame in page.frames:
                    url = urlparse(frame.url)
                    if not (url.hostname or '').endswith('boardgamearena.com'):
                        continue
                    if parse_qs(url.query).get('table') != ['934713322']:
                        continue
                    snapshot = await frame.evaluate(SNAPSHOT_JS)
                    if snapshot:
                        found.append(snapshot)
    assert len(found) == 1, len(found)
    path = OUT / 'captured_snapshot.json'
    assert not path.exists(), 'Preserve the original capture; choose a separate output.'
    path.write_text(json.dumps(found[0], ensure_ascii=False, indent=2) + '\n')
    snapshot = found[0]
    print(json.dumps({'move': snapshot['move_text'],
                      'actor': snapshot['gamedatas']['gamestate']['active_player'],
                      'round': snapshot['gamedatas']['roundnumber'],
                      'path': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
