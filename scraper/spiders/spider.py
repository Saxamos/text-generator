import scrapy


class Spider(scrapy.Spider):
    name = 'spider'
    start_urls = [
        'https://www.contrepoints.org/category/cuturen'
    ]

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
