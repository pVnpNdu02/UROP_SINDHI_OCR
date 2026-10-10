
import pytesseract
from PIL import Image

# Path to the Tesseract executable
tesseract_cmd = "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Tesseract-OCR"

# Read the Sindhi text image
sindhi_image = Image.open('sindhi_text.png')

# Perform OCR to extract Sindhi text
sindhi_text = pytesseract.image_to_string(sindhi_image, lang='snd')

# Transliterate Sindhi text to Devanagari
def sindhi_to_devanagari(sindhi_text):
    # Define a dictionary for Sindhi to Devanagari transliteration
    transliteration_dict = {
        'آ': 'आ',
        'ا': 'अ',
        'ب': 'ब',
        'پ': 'प',
        'ت': 'त',
        'ٽ': 'ट',
        'ٹ': 'ट',
        'ث': 'स',
        'ج': 'ज',
        'ڄ': 'ड़',
        'ڊ': 'ड',
        'ڈ': 'ड',
        'ح': 'ह',
        'خ': 'ख',
        'د': 'द',
        'ڍ': 'द',
        'ڏ': 'द',
        'ڌ': 'द',
        'ذ': 'ज़',
        'ر': 'र',
        'ڙ': 'ड़',
        'ز': 'ज़',
        'ژ': 'ज़',
        'س': 'स',
        'ش': 'श',
        'ص': 'स',
        'ض': 'ज़',
        'ع': 'अ',
        'غ': 'ग',
        'ف': 'फ',
        'ق': 'क़',
        'ڪ': 'क',
        'ک': 'क',
        'ڪ': 'क',
        'گ': 'ग',
        'ڳ': 'ग़',
        'ڱ': 'ड़',
        'ل': 'ल',
        'م': 'म',
        'ن': 'न',
        'ڻ': 'ण',
        'ڼ': 'ण',
        'ڽ': 'ण',
        'ه': 'ह',
        'و': 'व',
        'ء': 'अ',
        'ي': 'य',
        '۽': 'य',
        '۾': 'य',
        'ئ': 'अ',
        'ى': 'य',
        'ۏ': 'व',
        'ٻ': 'ब',
        'ٺ': 'ठ',
        'ٿ': 'ट',
        'ڀ': 'ब',
        'ؤ': 'व',
        '٭': 'ऱ',
        'ڃ': 'च',
        'ڻ': 'ण',
        'ڨ': 'फ़',
        'ﻁ': 'च',
        'ڪ': 'क',
        '٪': '%',
        '،': ',',
        '؛': ';',
        '؟': '?',
    }

    devanagari_text = ''.join([transliteration_dict.get(char, char) for char in sindhi_text])

    return devanagari_text

    # You need to implement your own Sindhi to Devanagari transliteration logic here
    # There are various libraries and methods available to perform this transliteration.
    # You can use language-specific libraries or write custom rules based on your needs.

    


devanagari_text = sindhi_to_devanagari(sindhi_text)

# Print the transliterated Devanagari text
print(devanagari_text)
