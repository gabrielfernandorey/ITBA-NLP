import streamlit as st

from components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
            "## Uso\n"
            "1. Ingresar [OpenAI API key](https://platform.openai.com/account/api-keys) abajo🔑\n"  # noqa: E501
            "2. Subir archivos de noticias 📄\n"
            "3. Consultar tópicos encontrados\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Ingrese su OpenAI API key aquí (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )

        st.session_state["OPENAI_API_KEY"] = api_key_input

        st.markdown("---")
        st.markdown("# Alcance")
        st.markdown(
            "Esta aplicación está desarrollada con fines académicos para el TP Final de NLP en ITBA-2024, "
            "esta aplicación permite generar tópicos de noticias. "
            
        )
        
        st.markdown("Desarrollado por Gabriel Rey")
        st.markdown("---")

        faq()