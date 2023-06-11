import streamlit as st
import openai
import os
from gtts import gTTS
import streamlit as st
from translate import Translator
from dotenv import load_dotenv

load_dotenv()
my_id = os.getenv("APIKEY")

openai.api_key = my_id

lang_dict = {
    'Afrikaans': 'af', 'Albanian': 'sq','Arabic': 'ar','Armenian': 'hy',
    'Bengali': 'bn','Bosnian': 'bs',
    'Catalan': 'ca','Croatian': 'hr','Czech': 'cs',
    'Danish': 'da','Dutch': 'nl',
    'English': 'en','Esperanto': 'eo','Estonian': 'et',
    'Filipino': 'tl','Finnish': 'fi','French': 'fr',
    'German': 'de','Greek': 'el','Gujarati': 'gu',
    'Haitian Creole': 'ht','Hebrew': 'he','Hindi': 'hi','Hungarian': 'hu',
    'Icelandic': 'is','Indonesian': 'id','Italian': 'it',
    'Japanese': 'ja','Javanese': 'jv',
    'Kannada': 'kn','Kazakh': 'kk',
    'Khmer': 'km','Korean': 'ko','Kurdish': 'ku','Kyrgyz': 'ky',
    'Lao': 'lo','Latvian': 'lv','Lithuanian': 'lt','Luxembourgish': 'lb',
    'Macedonian': 'mk','Malayalam': 'ml','Marathi': 'mr','Mongolian': 'mn',
    'Nepali': 'ne','Norwegian': 'no',
    'Persian': 'fa','Polish': 'pl','Portuguese': 'pt','Punjabi': 'pa',
    'Romanian': 'ro','Russian': 'ru',
    'Serbian': 'sr','Sinhalese': 'si','Slovak': 'sk','Slovenian': 'sl','Somali': 'so','Spanish': 'es','Sundanese': 'su','Swahili': 'sw','Swedish': 'sv',
    'Tamil': 'ta','Telugu': 'te','Thai': 'th','Turkish': 'tr',
    'Ukrainian': 'uk','Urdu': 'ur','Uyghur': 'ug',
    'Vietnamese': 'vi','Welsh': 'cy','Xhosa': 'xh','Yiddish': 'yi','Zulu': 'zu'
}

def language_translator(text_explanation, lang='ar', justText=False):
    translator = Translator(to_lang=lang)
    original_text = text_explanation
    translated_text = translator.translate(original_text)
    
    if justText == True:
        return translated_text
    else:
        myobj = gTTS(text=translated_text, lang=lang, slow=False)
        myobj.save("welcome.mp3")

        audio_file = open('welcome.mp3', 'rb')
        audio_bytes = audio_file.read()
        return st.audio(audio_bytes, format='audio/mp3')
    

def get_chat_response(message):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=message,
      temperature=0.5,
      max_tokens=100
    )
    return response.choices[0].text.strip()

def app():
    st.header('Healthcare Query')
    
    st.subheader('Ask anything about your health.')


    user_input = st.text_input("Ask a health-related question:")
    if user_input:
        response = get_chat_response(user_input)
        st.text(response)
    if st.checkbox("Get Translation"):
        lang = st.text_input('Please provide desired language')
        if lang != '':
            with st.expander('Review translation'):
                language_translator(response, lang=lang_dict[lang.title()])

