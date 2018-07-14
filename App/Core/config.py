from InformationExtraction.settings import TESSERACT_PATH
import pytesseract

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH  # '/usr/local/bin/tesseract'
LANG = 'rus'
