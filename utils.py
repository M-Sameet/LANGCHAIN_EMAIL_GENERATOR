import re


def clean_text(text):
    text=re.sub(r'<[^>]*?>', '' ,text)
    text=re.sub(r'\s{2,}', '' ,text)
    text=text.strip()
    text = ' '.join(text.split())
    return text