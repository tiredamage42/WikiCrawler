'''
wrappers for wikipedia methods
'''
import wikipedia
import random


def _get_random_pages(num_pages):
    '''
    Given the wanted number of articles returned, 
    get random wikipedia article objects
    '''

    valid_pages = []

    def validate_title (title):
        return not (('disambiguation' in title) or ('it may refer to' in title))

    def add_page_if_valid(title):
        try:
            page = wikipedia.page(title)
            if validate_title(page.title.lower()):
                valid_pages.append(page)

        except wikipedia.exceptions.DisambiguationError as e:
            add_page_if_valid(random.choice(e.options))    

        except wikipedia.exceptions.PageError:
            pass

    while len(valid_pages) != num_pages:

        n = num_pages - len(valid_pages)
        article_titles = wikipedia.random(n)

        if n == 1:
            add_page_if_valid(article_titles)
        else:
            for title_candidate in article_titles:
                add_page_if_valid(title_candidate)

            
    return valid_pages

def _get_text_from_page(page):
    '''
    returns an object with the summary of the page
    and teh formatted text from the body (excluding external links, 
    and other sections that don't have natural language type text)
    '''
    #print page.sections
    #print page.url
    #print page.images
    #print page.links

    # the section headers to filter out (we just want normal text)
    sections_not_needed = [
        'external links', 'see also', 'references', 'cast', 'side 2', 'side 1'
    ]

    article_body = []
    skip_current_section = False
    for line in page.content.split('\n'):

        # if line is not empty
        if line.strip():

            # if it starts with '==' then it's a section header
            if line.startswith('=='):
                # skip this section if it's any of our sections we don't need
                skip_current_section = any((section in line.lower()) for section in sections_not_needed) 
            else:
                if not skip_current_section:
                    article_body.append(line)


    return {
        'summary': page.summary, 
        'text': ' '.join(article_body) 
    }


# get the formatted summary / text objects
# for a number of random pages
def get_random_pages(page_count):
    return [ _get_text_from_page(page) for page in _get_random_pages(page_count) ]

