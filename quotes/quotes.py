import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['toscrape.com']
    # start_urls = ['http://quotes.toscrape.com/random']
    start_urls = ['http://quotes.toscrape.com']

    def parse(self, response):
        self.log('I just visited ' + response.url)
        # yield {
        #     'author_name': response.css('small.author::text').extract_first(),
        #     'text': response.css('span::text').extract_first(),
        #     'tags': response.css('a.tag::text').extract(),
        # }
        for quote in response.css('div.quote'):
            item = {
                'author_name': quote.css('small.author::text').extract_first(),
                'text': quote.css('span.text::text').extract_first(),
                'tags': quote.css('a.tag::text').extract()
            }
            yield item