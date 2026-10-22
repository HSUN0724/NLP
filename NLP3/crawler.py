"""步驟 1：爬取健康醫療網「新聞分類」的新聞，存入 SQLite。

用法：
    python crawler.py              # 每個分類抓 2 頁
    python crawler.py --pages 3    # 每個分類抓 3 頁
"""
import argparse
import re
import time

import requests
from bs4 import BeautifulSoup

from config import NEWS_CHANNELS, SITE
from db import get_conn

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NLP-homework-crawler"}
DELAY = 0.5  # 每次請求間隔秒數，避免造成網站負擔


def fetch(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def list_article_urls(channel_id, page):
    url = f"{SITE}/channel/{channel_id}/" + (str(page) if page > 1 else "")
    soup = fetch(url)
    urls = []
    for a in soup.select('a[href*="/article/"]'):
        m = re.search(r"/article/(\d+)", a["href"])
        if m:
            full = f"{SITE}/article/{m.group(1)}"
            if full not in urls:
                urls.append(full)
    return urls


def parse_article(url):
    soup = fetch(url)
    h1 = soup.select_one("#article-title h1")
    body = soup.select_one("#article-content")
    if not h1 or not body:
        return None

    # 全文：取段落與小標，去掉圖說以外的空白
    parts = [t.get_text(" ", strip=True) for t in body.find_all(["p", "h2", "h3", "li"])]
    content = "\n".join(p for p in parts if p)

    published = soup.select_one('meta[property="article:published_time"]')
    published_at = published["content"].replace("T", " ")[:19] if published else None

    author = None
    info = soup.select_one(".article-info")
    if info:
        info_text = info.get_text(" ", strip=True)
        m = re.search(r"健康醫療網／(.+)", info_text)
        author = m.group(1).strip() if m else None
        # 部分文章沒有 meta 發布時間，改用頁面上的日期
        d = re.search(r"\d{4}-\d{2}-\d{2}", info_text)
        if not published_at and d:
            published_at = d.group(0)

    return {
        "title": h1.get_text(strip=True),
        "url": url,
        "author": author,
        "content": content,
        "published_at": published_at,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2, help="每個分類要抓幾頁")
    args = parser.parse_args()

    conn = get_conn()
    existing = {r[0] for r in conn.execute("SELECT url FROM news")}
    added = 0

    for category, channel_id in NEWS_CHANNELS.items():
        for page in range(1, args.pages + 1):
            try:
                urls = list_article_urls(channel_id, page)
            except requests.RequestException as e:
                print(f"[略過] {category} 第 {page} 頁：{e}")
                continue
            print(f"{category} 第 {page} 頁：{len(urls)} 篇")
            time.sleep(DELAY)

            for url in urls:
                if url in existing:
                    continue
                try:
                    art = parse_article(url)
                except requests.RequestException as e:
                    print(f"  [失敗] {url}：{e}")
                    continue
                time.sleep(DELAY)
                if not art or not art["content"]:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO news (title, url, category, author, content, published_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (art["title"], url, category, art["author"], art["content"], art["published_at"]),
                )
                conn.commit()
                existing.add(url)
                added += 1
                print(f"  + {art['published_at']} {art['title']}")

    total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    print(f"\n本次新增 {added} 篇，資料庫共 {total} 篇。")


if __name__ == "__main__":
    main()
