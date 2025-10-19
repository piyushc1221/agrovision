from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(storage={'root_dir': 'dataset/train/wheat_blight'})
google_crawler.crawl(keyword='wheat blight leaf', max_num=100)
