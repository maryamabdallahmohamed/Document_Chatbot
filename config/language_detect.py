from langdetect import detect, DetectorFactory
import logging

# Set a fixed seed for consistent results
DetectorFactory.seed = 0

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def returnlang(text):
    """Detect language of the input text."""
    try:
        detected = detect(text)
        # Map language codes to full names for better model understanding
        lang_mapping = {
            'ar': 'Arabic',
            'en': 'English',
            'fr': 'French'
        }
        return lang_mapping.get(detected, 'English')
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to English.")
        return 'English'
