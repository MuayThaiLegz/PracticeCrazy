import streamlit as st
import openai
import os, base64
from gtts import gTTS
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
      engine="text-davinci-003",
      prompt=message,
      temperature=0.8,
      max_tokens=495
    )
    return response.choices[0].text.strip()


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size:  1925px 950px;
    }
    .stApp > div > div, .dataframe-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size:  600px 955px;
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
    

def app():
    set_background('denoised_aipictureResized-PhotoRoom (2).png')
    sidebar_bg('AiDoc.jpg')
    st.markdown("<h1 style='text-align: center; color: white;'>Healthcare Query</h1>"
                        "<h1 style='text-align: center; color: white;'>Ask anything about your health.</h1>", unsafe_allow_html=True)
    
    
    

    with st.expander("Ask a health-related question in box below:"):
        user_input = st.text_input("")
        if user_input:
            response = get_chat_response(user_input)
            st.info(response)
    with st.expander("Get Translation"):
        lang = st.text_input('Please provide desired language')
        if lang != '':
            language_translator(response, lang=lang_dict[lang.title()])

