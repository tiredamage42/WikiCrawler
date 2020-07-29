import wikipedia_utils

# get text from the articles
page_texts = wikipedia_utils.get_random_pages(page_count=4)
    
for t in page_texts:
   print ('\n\n{0}\nLength: {1}\n{2}\n\n'.format(t['summary'], len(t['text']), t['text']))
    