### App Topicos de noticias ###

import streamlit as st
from streamlit_option_menu import option_menu

from components.sidebar import sidebar
from core.functions import *

from opensearch_io import init_opensearch

import warnings
warnings.filterwarnings('ignore')



# Header-------------------------------------------------------------
st.set_page_config(page_title = 'Topicos de Noticias', # Nombre de la pagina, sale arriba cuando se carga streamlit
                   page_icon = ':notebook_with_decorative_cover:', # https://www.webfx.com/tools/emoji-cheat-sheet/
                   layout="wide") 

# Barra de menu lateral---------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Topicos de noticias",
        options=["Ingesta", "Base de Noticias", "Detección de Tópicos", "Topicos", "Gestión de Tópicos"],
    )

sidebar()

# Validación de base de datos iniciada e indices creados
init_opensearch()

# OpenAI API key --------------------------------------------------
openai_api_key = st.session_state.get("OPENAI_API_KEY")

if not openai_api_key:
    st.warning(
        "Enter your OpenAI API key in the sidebar."
    )
    
if selected == "Ingesta":   

    # Ingesta de datos
    data_ingestion()


if selected == "Base de Noticias":

    # Visualizar las noticias
    view_news()


if selected == "Detección de Tópicos":

    # Proceso de detección de tópicos
    topic_process(openai_api_key)


if selected == "Topicos":

    # Visualizar los topicos que existan en la base
    view_all_topics()


if selected == "Gestión de Tópicos":

    # Visualizar los topicos que existan en la base
    control()



    
    










