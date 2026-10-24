"""
NewsSpider — scrapes ~500 recent news articles from BBC, The Guardian, NPR, and Ars Technica via RSS.

Output goes to articles.jsonl (configured in custom_settings["FEEDS"]). Run with:
    python run_spider.py
or:
    scrapy runspider news_spider.py
"""

import scrapy
from datetime import datetime, timezone


# RSS feeds to seed article discovery across multiple topics.
# BBC and Guardian: ~20-30 items/feed. NPR: 10 items/feed. Ars Technica: 20 items/feed.
# Together these supply well over 500 candidate URLs so the spider can hit TARGET_COUNT.
RSS_FEEDS = {
    # BBC News (10 feeds)
    "bbc_top":       "https://feeds.bbci.co.uk/news/rss.xml",
    "bbc_world":     "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_tech":      "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "bbc_business":  "https://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_science":   "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "bbc_health":    "https://feeds.bbci.co.uk/news/health/rss.xml",
    "bbc_politics":  "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "bbc_us_canada": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "bbc_europe":    "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "bbc_asia":      "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    # The Guardian (8 feeds)
    "guardian_world":       "https://www.theguardian.com/world/rss",
    "guardian_us":          "https://www.theguardian.com/us-news/rss",
    "guardian_tech":        "https://www.theguardian.com/technology/rss",
    "guardian_science":     "https://www.theguardian.com/science/rss",
    "guardian_business":    "https://www.theguardian.com/business/rss",
    "guardian_environment": "https://www.theguardian.com/environment/rss",
    "guardian_politics":    "https://www.theguardian.com/politics/rss",
    "guardian_health":      "https://www.theguardian.com/society/rss",
    # NPR News (11 feeds, 10 items each — verified working)
    "npr_top":      "https://feeds.npr.org/1001/rss.xml",
    "npr_science":  "https://feeds.npr.org/1007/rss.xml",
    "npr_health":   "https://feeds.npr.org/1128/rss.xml",
    "npr_politics": "https://feeds.npr.org/1014/rss.xml",
    "npr_1002":     "https://feeds.npr.org/1002/rss.xml",
    "npr_1003":     "https://feeds.npr.org/1003/rss.xml",
    "npr_1013":     "https://feeds.npr.org/1013/rss.xml",
    "npr_1019":     "https://feeds.npr.org/1019/rss.xml",
    "npr_1039":     "https://feeds.npr.org/1039/rss.xml",
    "npr_1045":     "https://feeds.npr.org/1045/rss.xml",
    "npr_1057":     "https://feeds.npr.org/1057/rss.xml",
    # Ars Technica (5 feeds, 20 items each — verified working)
    "ars_index":    "https://feeds.arstechnica.com/arstechnica/index",
    "ars_tech_lab": "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "ars_science":  "https://feeds.arstechnica.com/arstechnica/science",
    "ars_policy":   "https://feeds.arstechnica.com/arstechnica/tech-policy",
    "ars_gaming":   "https://feeds.arstechnica.com/arstechnica/gaming",
}

TARGET_COUNT = 500


class NewsSpider(scrapy.Spider):
    name = "news_spider"

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "ROBOTSTXT_OBEY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 1.0,
        "AUTOTHROTTLE_MAX_DELAY": 5.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 0.5,
        "FEEDS": {
            "articles.jsonl": {"format": "jsonlines", "overwrite": True},
        },
        "LOG_LEVEL": "INFO",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.article_count = 0

    async def start(self):
        for feed_name, url in RSS_FEEDS.items():
            yield scrapy.Request(
                url,
                callback=self.parse_rss,
                cb_kwargs={"feed_name": feed_name},
                headers={"Accept": "application/rss+xml, application/xml, text/xml"},
            )

    def parse_rss(self, response, feed_name):
        """Parse RSS feed and yield requests for each article URL."""
        items = response.xpath("//item")
        self.logger.info(f"[{feed_name}] Found {len(items)} items in RSS")

        for item in items:
            if self.article_count >= TARGET_COUNT:
                return

            url = item.xpath("link/text()").get()
            if not url:
                url = item.xpath("guid/text()").get()
            if not url or not url.startswith("http"):
                continue

            title       = item.xpath("title/text()").get("").strip()
            pub_date    = item.xpath("pubDate/text()").get("").strip()
            description = item.xpath("description/text()").get("").strip()

            # Route to the correct parser based on feed name prefix
            if feed_name.startswith("bbc"):
                source   = "BBC News"
                callback = self.parse_bbc_article
            elif feed_name.startswith("npr"):
                source   = "NPR News"
                callback = self.parse_npr_article
            elif feed_name.startswith("ars"):
                source   = "Ars Technica"
                callback = self.parse_ars_article
            else:
                source   = "The Guardian"
                callback = self.parse_guardian_article

            yield scrapy.Request(
                url,
                callback=callback,
                cb_kwargs={
                    "rss_title":       title,
                    "rss_pub_date":    pub_date,
                    "rss_description": description,
                    "source":          source,
                    "feed_name":       feed_name,
                },
                dont_filter=False,
            )

    # ------------------------------------------------------------------
    # BBC article parser
    # ------------------------------------------------------------------
    def parse_bbc_article(self, response, rss_title, rss_pub_date, rss_description, source, feed_name):
        if self.article_count >= TARGET_COUNT:
            return

        paragraphs = response.css(
            "article [data-component='text-block'] p::text, "
            "article [data-component='text-block'] p *::text"
        ).getall()

        if not paragraphs:
            paragraphs = response.css("article p::text, article p *::text").getall()

        body = " ".join(p.strip() for p in paragraphs if p.strip())

        if len(body.split()) < 80:
            return

        title = (
            response.css("h1::text").get()
            or response.css("title::text").get("").replace(" - BBC News", "")
            or rss_title
        ).strip()

        pub_date = (
            response.css("time[datetime]::attr(datetime)").get()
            or rss_pub_date
        )

        topic = self._topic_from_url(response.url)
        self.article_count += 1
        self.logger.info(f"[{self.article_count}/{TARGET_COUNT}] {title[:60]}")

        yield {
            "article_id": self.article_count,
            "title": title,
            "url": response.url,
            "source": source,
            "topic": topic,
            "pub_date": pub_date,
            "description": rss_description,
            "body": body,
            "word_count": len(body.split()),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Guardian article parser
    # ------------------------------------------------------------------
    def parse_guardian_article(self, response, rss_title, rss_pub_date, rss_description, source, feed_name):
        if self.article_count >= TARGET_COUNT:
            return

        paragraphs = response.css(
            "div[data-gu-name='body'] p::text, "
            "div[data-gu-name='body'] p *::text, "
            "[itemprop='articleBody'] p::text, "
            "[itemprop='articleBody'] p *::text"
        ).getall()

        if not paragraphs:
            paragraphs = response.css("div.content__article-body p::text").getall()

        body = " ".join(p.strip() for p in paragraphs if p.strip())

        if len(body.split()) < 80:
            return

        title = (
            response.css("h1::text").get()
            or response.css("title::text").get("").split("|")[0]
            or rss_title
        ).strip()

        pub_date = (
            response.css("time[datetime]::attr(datetime)").get()
            or response.css("meta[property='article:published_time']::attr(content)").get()
            or rss_pub_date
        )

        topic = self._topic_from_url(response.url)
        self.article_count += 1
        self.logger.info(f"[{self.article_count}/{TARGET_COUNT}] {title[:60]}")

        yield {
            "article_id": self.article_count,
            "title": title,
            "url": response.url,
            "source": source,
            "topic": topic,
            "pub_date": pub_date,
            "description": rss_description,
            "body": body,
            "word_count": len(body.split()),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # NPR article parser
    # ------------------------------------------------------------------
    def parse_npr_article(self, response, rss_title, rss_pub_date, rss_description, source, feed_name):
        if self.article_count >= TARGET_COUNT:
            return

        # NPR body lives in div.storytext (class="storytext storylocation linkLocation")
        paragraphs = response.css(
            "div.storytext p::text, div.storytext p *::text"
        ).getall()

        if not paragraphs:
            paragraphs = response.css(
                "#storytext p::text, #storytext p *::text"
            ).getall()

        if not paragraphs:
            paragraphs = response.css("article p::text, article p *::text").getall()

        body = " ".join(p.strip() for p in paragraphs if p.strip())

        if len(body.split()) < 80:
            return

        title = (
            response.css("h1::text").get()
            or response.css("title::text").get("").split("|")[0].split("-")[0]
            or rss_title
        ).strip()

        pub_date = (
            response.css("time[datetime]::attr(datetime)").get()
            or response.css("meta[property='article:published_time']::attr(content)").get()
            or rss_pub_date
        )

        topic = self._topic_from_url(response.url)
        self.article_count += 1
        self.logger.info(f"[{self.article_count}/{TARGET_COUNT}] {title[:60]}")

        yield {
            "article_id": self.article_count,
            "title": title,
            "url": response.url,
            "source": source,
            "topic": topic,
            "pub_date": pub_date,
            "description": rss_description,
            "body": body,
            "word_count": len(body.split()),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Ars Technica article parser
    # ------------------------------------------------------------------
    def parse_ars_article(self, response, rss_title, rss_pub_date, rss_description, source, feed_name):
        if self.article_count >= TARGET_COUNT:
            return

        # Ars Technica body is in div.article-content
        paragraphs = response.css(
            "div.article-content p::text, div.article-content p *::text"
        ).getall()

        if not paragraphs:
            paragraphs = response.css(
                "section.article-content p::text, section.article-content p *::text"
            ).getall()

        if not paragraphs:
            paragraphs = response.css("article p::text, article p *::text").getall()

        body = " ".join(p.strip() for p in paragraphs if p.strip())

        if len(body.split()) < 80:
            return

        title = (
            response.css("h1::text").get()
            or response.css("title::text").get("").split("|")[0]
            or rss_title
        ).strip()

        pub_date = (
            response.css("meta[property='article:published_time']::attr(content)").get()
            or response.css("time[datetime]::attr(datetime)").get()
            or rss_pub_date
        )

        topic = self._topic_from_url(response.url)
        self.article_count += 1
        self.logger.info(f"[{self.article_count}/{TARGET_COUNT}] {title[:60]}")

        yield {
            "article_id": self.article_count,
            "title": title,
            "url": response.url,
            "source": source,
            "topic": topic,
            "pub_date": pub_date,
            "description": rss_description,
            "body": body,
            "word_count": len(body.split()),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _topic_from_url(url: str) -> str:
        """Infer a broad topic label from the article URL path."""
        segments = url.lower().split("/")
        topic_keywords = {
            "technology": "technology",
            "tech": "technology",
            "science": "science",
            "environment": "environment",
            "health": "health",
            "business": "business",
            "politics": "politics",
            "world": "world",
            "us-news": "us news",
            "us_and_canada": "us news",
            "europe": "europe",
            "asia": "asia",
            "sport": "sports",
            "gaming": "entertainment",
            "entertainment": "entertainment",
            "culture": "culture",
            "society": "society",
        }
        for seg in segments:
            for kw, label in topic_keywords.items():
                if kw in seg:
                    return label
        return "general"
