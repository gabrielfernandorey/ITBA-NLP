import streamlit as st


def faq():
    st.markdown(
        """
# Resumen
Esta app fue desarrollada en python utilizando las técnicas de Topic Modeling
- Herramientas utilizadas:
    - BERTopic: es una técnica de modelado de temas que utiliza embeddings
de lenguaje de modelos como BERT para generar representaciones semánticas de documentos
y agruparlos en tópicos interpretables.
    - spaCy: es una biblioteca de procesamiento de lenguaje natural (NLP) en Python diseñada
para realizar tareas como tokenización, etiquetado de partes del discurso, lematización,
reconocimiento de entidades nombradas y análisis de dependencias de manera rápida y eficiente.
    - OpenSearch: es un motor de búsqueda y análisis de código abierto desarrollado como un fork de Elasticsearch que permite la operación de embeddings. 
"""
    )