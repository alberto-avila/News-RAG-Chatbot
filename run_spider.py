"""
Convenience runner: executes NewsSpider without needing a Scrapy project.

Usage:
    python run_spider.py
"""

from scrapy.crawler import CrawlerProcess
from news_spider import NewsSpider

process = CrawlerProcess()
process.crawl(NewsSpider)
process.start()
