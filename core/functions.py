import os, re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import Counter, defaultdict

from datetime import datetime, timedelta
from dateutil.parser import parse
#----------------------------------------------------------------------------------
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode
#----------------------------------------------------------------------------------
from opensearch_data_model import Topic, TopicKeyword, News, os_client, TOPIC_INDEX_NAME, NEWS_INDEX_NAME
from opensearch_io import save_news, get_news, update_news, get_topics_opensearch, select_data_from_news, delete_index_opensearch, get_topics_date

#----------------------------------------------------------------------------------
from NLP_tools import Cleaning_text, top_keywords, top_entities, topic_documents, get_topic_name, best_document, clean_all, keywords_with_neighboards
#----------------------------------------------------------------------------------
import spacy
#----------------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
#----------------------------------------------------------------------------------
from openai import OpenAI


#----------------------------------------------------------------------------------
def process_text(PATH, input_text):
    """
    Función para estimar el topico de una noticia ingresada manualmente como texto
    """
    try:
        topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")

        new_doc_embedding = topic_model.embedding_model.embed(input_text)

        # Buscamos en la base a que topico pertenece el nuevo documento
        query = {
            "size": 1,
            "query": {
                "knn": {
                    "vector": {
                        "vector": list(new_doc_embedding),
                        "k" : 3
                    }
                }
            }
        }
        response = os_client.search(index='topic', body=query)
    
        st.write(f" Topico: {response['hits']['hits'][0]['_source']['index']} - {response['hits']['hits'][0]['_source']['name']}")

        estim = round(response['hits']['hits'][0]['_score'],4)
        if estim >= 0.8:
            color = "#80EE19"
            font_size=20
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
        elif estim >= 0.5 and estim < 0.8:
            color = "#EEA719"
            font_size=16
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
        else:
            color = "#FF5733"
            font_size=14
            st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)

    except:
        st.warning("Error al estimar el Tópico")

    return
#----------------------------------------------------------------------------------
def data_ingestion():     
    """
    Ingesta de noticias - Proceso por lote o estima un topico de una noticia ingresada manualmente
    """
    st.title("Ingesta de noticias")
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    # Carga de archivo 
    uploaded_file = st.file_uploader(
                        "Cargar un nuevo archivo Parquet",
                        type=["parquet"],
                        help="Debe cargar un archivo de noticias compatible",
    )

    if not uploaded_file:

        # Crear un cuadro de entrada de texto
        input_text = st.text_area("O ingrese una noticia para estimar un tópico", height=200)

        if st.button("Estimar tópico"):
            
            process_text(PATH, input_text) # Estimar topico de noticia ingresada como texto

        else:
            st.stop()
    else:
        uploaded_file.read()
        st.success(f"File {uploaded_file.name} uploaded successfully")
    
        # Leer el archivo Parquet
        df = pd.read_parquet(uploaded_file)
    
        # Mostrar el DataFrame
        st.write("Vista previa del archivo Parquet:")
        st.write(df)
        
        # Mostrar total de registros
        st.write("Registros: ", len(df))

        # Boton para procesar el archivo
        if st.button("Process file"):

            process_file(df, PATH) # Procesar lote de noticias
            
    return
        
#----------------------------------------------------------------------------------
def process_file(df, PATH):
    """
    Proceso de lote de noticias 
    """
        
    data = list(df['in__text'])
    
    # Cargar el modelo de spaCy para español
    spa = spacy.load("es_core_news_lg")

    # Stopwords
    SPANISH_STOPWORDS = list(pd.read_csv(PATH+'spanish_stop_words.csv' )['stopwords'].values)
    SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv(PATH+'spanish_stop_words_spec.csv' )['stopwords'].values)
    
    # Crear un espacio para la barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("Procesando entidades y keywords...")    

    # Detectar ENTIDADES para todos los documentos usando spaCy
    entities = []
    keywords_spa = []
    for i, data_in in enumerate(data):

        # Actualizar la barra de progreso
        progress = (i + 1) / len(data)
        progress_bar.progress(progress)
        status_text.text(f'Procesando documento {i+1} de {len(data)}')

        # Contabilizar palabras en doc original
        normalized_text = re.sub(r'\W+', ' ', data_in.lower())
        words_txt_without_stopwords = [word for word in normalized_text.split() if word not in SPANISH_STOPWORDS+SPANISH_STOPWORDS_SPECIAL]
        words_txt_counter = Counter(words_txt_without_stopwords)
        words_counter = {elemento: cuenta for elemento, cuenta in sorted(words_txt_counter.items(), key=lambda item:item[1], reverse=True) if cuenta > 1}

        # Extraer entidades del doc segun atributos
        extract = spa(data_in)
        entidades_spacy = [(ent.text, ent.label_) for ent in extract.ents]
        ent_select = [ent for ent in entidades_spacy if ent[1] == 'PER' or ent[1] == 'ORG' or ent[1] == 'LOC' ]

        # Extraer keywords del doc
        keywords_spa.append([(ext.text, ext.pos_) for ext in extract]) 

        # Extraer entidades de "maximo 3 palabras"
        ent_max_3 = [ent[0] for ent in ent_select if len(ent[0].split()) <= 3]
        ent_clean = clean_all(ent_max_3, accents=False)
        ent_unique = list(set([ word for word in ent_clean if word not in SPANISH_STOPWORDS+SPANISH_STOPWORDS_SPECIAL] ))

        ents_proc = {}
        for ent in ent_unique:
            
            # Criterio de selección 
            weight = 0
            for word in ent.split():
                if word in words_counter:
                    weight += 1 /len(ent.split()) * words_counter[word]
            
            ents_proc[ent] = round(weight,4)

        ents_proc_sorted = {k: v for k, v in sorted(ents_proc.items(), key=lambda item: item[1], reverse=True) if v > 0}

        # Crear la lista preliminar de entidades procesadas por noticia 
        pre_entities = [key for key, _ in ents_proc_sorted.items()] 

        # Obtener las últimas palabras de cada entidad que tenga mas de una palabra por entidad
        last_words = list(set([ent.split()[-1] for ent in pre_entities if len(ent.split()) > 1 ]))

        # Eliminar palabra única si la encuentra al final de una compuesta
        pre_entities_without_last_word_equal = []
        for idx, ent in enumerate(pre_entities):
            if not (len(ent.split()) == 1 and ent in last_words):
                pre_entities_without_last_word_equal.append(ent)

        # Obtener las palabras únicas
        unique_words = [ ent.split()[0] for ent in pre_entities_without_last_word_equal if len(ent.split()) > 1 ]

        # Eliminar palabra única si la encuentra al comienzo de una compuesta
        pre_entities_without_first_word_equal = []
        for idx, ent in enumerate(pre_entities_without_last_word_equal):
            if not (len(ent.split()) == 1 and ent in unique_words):
                pre_entities_without_first_word_equal.append(ent)

        # obtener entidades filtradas
        if len(pre_entities_without_first_word_equal) > 10:
            umbral = 10 + (len(pre_entities_without_first_word_equal)-10) // 2
            filter_entities = pre_entities_without_first_word_equal[:umbral] 
        else:
            filter_entities = pre_entities_without_first_word_equal[:10]

        pre_original_entities = []
        # capturar las entidades en formato original
        for ent in filter_entities:
            pre_original_entities.append([elemento for elemento in ent_max_3 if elemento.lower() == ent.lower()])

        sort_original_entities = sorted(pre_original_entities, key=len, reverse=True)
        
        try:
            entities.append( [ent[0] for ent in sort_original_entities if ent] ) 
        except Exception as e:
            entities.append([])

    
    # Detectar KEYWORDS with neighboards y keywords mas frecuentes
    k_w_n, keyword_single = keywords_with_neighboards(keywords_spa)
    
    # filtramos las que al menos se repiten una vez
    filtered_k_w_n = [ [tupla[0] for tupla in sublista if tupla[1] > 1] for sublista in k_w_n ]
    
    # Si un keyword unigrama coincide en los bigramas elegidos se descarta.
    # la cantidad de keywords se obtiene utilizando la media como umbral de corte
    
    values = [value for sublist in keyword_single for _, value in sublist]
    threshold = np.mean(values) # Umbral

    for i, sublist in enumerate(keyword_single):
        lista_k_w_n = list(set([word for sentence in filtered_k_w_n[i] for word in sentence.split()]))
        for tupla in sublist:
            if tupla[1] >= threshold and tupla[0] not in lista_k_w_n:
                filtered_k_w_n[i].append(tupla[0])

    keywords = filtered_k_w_n 

    # Grabar noticias en la base
    st.write("Actualizando noticias en OpenSearch...")

    response = save_news(data, df, entities, keywords)  

    if response['errors']:
        st.write("Errores encontrados al insertar los documentos")
    else:
        st.write("Actualizado.")

    return
#----------------------------------------------------------------------------------
def view_news():

    def style_tags(tags):
        styled_tags = ' | '.join([f' {tag} ' for tag in tags])
        return styled_tags
    
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    index_name = 'news'
    db_news = []
    for doc in News.search().query().scan():
        index       = doc.meta.id
        title       = doc.to_dict()['title']
        author      = doc.to_dict()['author']
        try:
            keywords =  doc.to_dict()['keywords']
        except:
            keywords = [""]
        try:
            entities =  doc.to_dict()['entities']
        except:
            entities = [""]

        created_at  = doc.to_dict()['created_at'] 
        process     = doc.to_dict()['process'] 

        db_news.append([index, title, style_tags(keywords), style_tags(entities), author, created_at, process])

    # Convertir a DataFrame
    df = pd.DataFrame(db_news, columns=["indice", "titulo", "keywords", "entidades", "autor", "fecha", "procesado" ])
    

    # Mostrar el DataFrame como una grilla en Streamlit
    st.title("Base de Noticias")
    
    # Configurar opciones de la grilla
    gb = GridOptionsBuilder.from_dataframe(df)

    # Definir anchos de columna específicos
    gb.configure_column('indice', width=70) 
    gb.configure_column('titulo', width=400) 
    gb.configure_column('keywords', width=200) 
    gb.configure_column('entidades', width=200) 
    gb.configure_column('autor', width=100)   
    gb.configure_column('fecha', width=100) 
    gb.configure_column('procesado', width=100) 
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_selection('single')
    grid_options = gb.build()

    # Mostrar la grilla con AgGrid
    response = AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=300,
        width=1400,
    )
    
    # Obtener la selección de la grilla
    selected_rows = response['selected_rows'] if response['selected_rows'] is not None else []

    # Mostrar los datos de la fila seleccionada
    if len(selected_rows) > 0:
        idx = selected_rows["indice"].values
        
        index_name = 'news'
        document_id = int(idx)
        # Realizar la búsqueda del documento por _id
        response = os_client.get(index=index_name, id=document_id)

        st.text_area(f":orange[Texto de la noticia seleccionada] | {response['_source']['title']}", response['_source']['news'], height=200)

        st.text_area(":orange[Keywords de la noticia seleccionada]", style_tags(response['_source']['keywords']), height=100)

        st.text_area(":orange[Entidades de la noticia seleccionada]", style_tags(response['_source']['entities']), height=100)

        # Boton para estimar tópico
        if st.button("Estimar Tópico"):

            try:
                topic_model = BERTopic.load(PATH+"modelos/bertopic_model_app")
                new_doc_embedding = topic_model.embedding_model.embed(response['_source']['news'])

                # Buscamos en la base a que topico pertenece el nuevo documento
                knn_query = {
                    "size": 1,
                    "query": {
                        "knn": {
                            "vector": {
                                "vector": new_doc_embedding,
                                "k" : 3
                            }
                        }
                    }
                }
                response = os_client.search(index='topic', body=knn_query)

                if response['hits']['total']['value'] > 0:
                    st.write(f" Topico: {response['hits']['hits'][0]['_source']['index']} - {response['hits']['hits'][0]['_source']['name']}")

                    estim = round(response['hits']['hits'][0]['_score'],4)
                    if estim >= 0.8:
                        color = "#80EE19"
                        font_size=20
                        st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
                    elif estim >= 0.5 and estim < 0.8:
                        color = "#EEA719"
                        font_size=16
                        st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
                    else:
                        color = "#FF5733"
                        font_size=14
                        st.write(f"Score: <span style='color:{color};font-size:{font_size}px'>{estim}</span>", unsafe_allow_html=True)
                else:
                    st.warning("No se puede estimar el Tópico")
            except:
                st.warning("Error al estimar el Tópico")
        
    else:
        st.write("No hay filas seleccionadas.")

    return

#----------------------------------------------------------------------------------
def topic_process(openai_api_key):
    """
    Proceso para la generación de tópicos
    """
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    client = OpenAI(api_key= openai_api_key)

    db_news = get_news()

    if db_news == []:
        st.write(":orange[Sin nuevos datos a procesar...]")
        return

    # Crear un diccionario para agrupar los registros por fecha (solo día, mes y año)
    fechas_dict = defaultdict(list)

    # Agrupar registros por fecha
    for registro in db_news:
        fecha_completa = registro[-1]
        fecha_solo_dia = fecha_completa.split('T')[0]  # Tomar solo el día, mes y año
        if fecha_solo_dia not in fechas_dict:
            fechas_dict[fecha_solo_dia] = 1
        else:
            fechas_dict[fecha_solo_dia] += 1

    # Convertir el diccionario en un DataFrame
    df_lotes = pd.DataFrame(list(fechas_dict.items()), columns=['Fecha', 'Cantidad de noticias'])

    
    # Mostrar el DataFrame en una grilla seleccionable
    st.subheader("Noticias a procesar")
    st.dataframe(df_lotes)
    st.write(f"Proximo lote de noticias a procesar: {next(iter(fechas_dict))}")    


    # Boton para procesar el archivo
    if st.button("Iniciar Proceso"):        

        db_news = get_news( next(iter(fechas_dict)) )

        df_news = pd.DataFrame(db_news , columns=["indice", "titulo", "noticia", "keywords", "entidades", "creado"])

        idx_data     = list(df_news.index)
        id_data      = list(df_news['indice'])
        title_data   = list(df_news['titulo'])
        data         = list(df_news['noticia'])
        keywords     = list(df_news['keywords'])
        entities     = list(df_news['entidades'])
       
        # Vocabulario - Unificar Entities + Keywords
        vocab = list(set().union(*entities, *keywords))
           
        st.write("Datos recuperados de la base...")

        # Modelo
        tfidf_vectorizer = TfidfVectorizer(
            tokenizer=None,
            max_df=0.9,
            min_df=0.1,
            ngram_range=(1, 2),
            vocabulary=vocab,
            # max_features=100_000
        )
        tfidf_vectorizer.fit(data)

        # Step 1 - Extract embeddings
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Step 2 - Reduce dimensionality
        umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
        # Step 3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        # Step 4 - Tokenize topics
        vectorizer_model = tfidf_vectorizer
        # Step 5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        # Step 6 - (Optional) Fine-tune topic representations with a `bertopic.representation` model
        # representation_model = KeyBERTInspired()

        # All steps together
        topic_model = BERTopic(
            embedding_model=embedding_model,                # Step 1 - Extract embeddings
            umap_model=umap_model,                          # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,                    # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,              # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                      # Step 5 - Extract topic words
            # representation_model=representation_model,    # Step 6 - (Optional) Fine-tune topic represenations
            # language='multilingual',                      # This is not used if embedding_model is used.
            # calculate_probabilities=True,                 
            verbose=True,
        )

        st.write("Entrenando...")

        # Entrenamiento
        topics, probs = topic_model.fit_transform(data)
        st.write(f"Topicos encontrados: {len(set(topics))-1}")

        #
        from_date  = [ next(iter(fechas_dict)) for _ in range(len(topic_model.get_topics().keys())) ] 
        to_date = [ datetime.strptime(next(iter(fechas_dict)), '%Y-%m-%d') + timedelta(days=1) for _ in range(len(from_date)-1) ]

        topics_to_save = topic_model.get_topics().keys()

        # Verificar si ya existe un modelo previo
        if os.path.isfile(PATH+"modelos/bertopic_model_app"):
            
            st.write("Iniciando proceso de fusion de modelos")
            
            # cargar modelo anterior
            topic_model_last = BERTopic.load(PATH+"modelos/bertopic_model_app")
            
            st.write(f"Topicos anteriores: {len(set(topic_model_last.get_topics().keys()))-1}")
            
            # respaldar modelo anterior
            topic_model_last.save(PATH+f"modelos/bertopic_model_app_old")

            # Obtener las fechas desde/hasta de los topicos existentes de opensearch
            from_date, to_date = get_topics_date()

            # Combine all models into one
            topic_model = BERTopic.merge_models([topic_model_last, topic_model])

            # Obtener los nuevos topicos
            topics_to_save = topic_model.get_topics().keys()
            st.write(f"Nueva cantidad de topicos: {len(set(topics_to_save))-1}")

            # Visualizar fusion de modelos
            df_combined = merged_results(topic_model_last, topic_model)
            st.write(df_combined)

            # actualizacion de fechas desde/hasta de los topicos
            from_date, to_date = update_topics_date(from_date, to_date, df_combined, fechas_dict)

            # Eliminar topicos de la base ( para que guardar los del modelo fusionado)
            delete_index_opensearch("topic")


        topic_model.save(PATH+f"modelos/bertopic_model_app")
        st.write("Modelo guardado.")

        st.write("Generando embeddings...")
        # Obtenemos embeddings de todos los documentos
        docs_embedding = topic_model.embedding_model.embed(data)
        np.save(PATH+f"modelos/docs_embeddings_app.npy", docs_embedding)
        st.write("Embeddings guardados.")
        

        # Crear un espacio para la barra de progreso
        st.write("Actualizando tópicos encontrados...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Grabar todos los topicos en la base
        for i, topic in enumerate(topics_to_save):

            # Actualizar la barra de progreso
            progress = (i+1) / len(set(topics))
            progress_bar.progress(progress)
            status_text.text(f'Procesando topico {i} de {len(set(topics))-1}')
            
            if topic > -1:
               
                topic_keywords_top  = top_keywords(topic, topic_model, PATH)
                topic_entities_top  = top_entities(topic, topic_model, topics, docs_embedding, idx_data, entities)
                topic_documents_ids, topic_documents_title, threshold  = topic_documents(topic, topic_model, probs, df_news, data)
                id_best_doc, title_best_doc, best_doc = best_document(topic, topic_model, docs_embedding, id_data, title_data, data)

                if topic_entities_top:

                    topic_doc = Topic(
                        index = topic,   
                        name = get_topic_name(' '.join(topic_documents_title), client),
                        vector = list(topic_model.topic_embeddings_[topic + 1 ]), 
                        similarity_threshold = threshold,                      
                        created_at     = datetime.now(),
                        from_date      = parse(from_date[topic]),
                        to_date        = to_date[topic],         
                        keywords       = topic_keywords_top,
                        entities       = topic_entities_top,
                        id_best_doc    = id_best_doc,
                        title_best_doc = title_best_doc,
                        best_doc       = best_doc,
                    ) 
                    
                    # Grabar topico
                    topic_doc.save()
                else:
                    st.error("Errores de procesamiento", icon="🔥")                       
                    return
                
        st.write("Actualizando base de noticias...")           
        # Marcar registros de noticias como procesados y grabar sus embeddings, topicos 
        update_news( id_data, docs_embedding, topics, probs )

        st.write("Proceso finalizado.")

    return
#----------------------------------------------------------------------------------
def view_all_topics():

    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')
    
    try:
        # Mostrar el DataFrame como una grilla en Streamlit
        st.title("Tópicos de noticias")

        # Seleccionador de fechas
        selected_date = st.date_input( "Selecciona una fecha", last_topic_date() )  
        
        # Convertir la fecha seleccionada a string en formato 'yyyy-MM-dd'
        date_str = selected_date.strftime('%Y-%m-%d')

        # Obtener los tópicos filtrados por la fecha seleccionada
        topics = get_topics_opensearch(date_filter=date_str)

        if topics:
            
            db_topics = []
            data_topics = {} 
            for reg in topics:
                index = reg['index']
                name = reg['name']
                similarity_threshold = reg['similarity_threshold']
                create_at = format_date(reg['created_at'])
                from_date = reg['from_date'][:10]
                to_date = reg['to_date'][:10]
                title_best_doc = reg['title_best_doc']
                id_best_doc = reg['id_best_doc']
        
                db_topics.append([index, name, round(similarity_threshold, 4), create_at, from_date, to_date, title_best_doc, id_best_doc])
                data_topics[index] = [name,
                                    title_best_doc,
                                    reg['best_doc'],
                                    reg['entities'],
                                    reg['keywords'],
                                    similarity_threshold,
                                    reg['vector']
                                    ] 
            
            # Ordenamos por idx de topico de menor a mayor
            db_topics = sorted(db_topics)

            # Convertir a DataFrame
            topics_df = pd.DataFrame(db_topics, columns=["indice", "nombre", "umbral", "creado", "desde", "hasta", "titulo noticia mas cercana", "id noticia"]) 

            st.write(f"Tópicos para la fecha:")

            # Configurar opciones de la grilla
            gb = GridOptionsBuilder.from_dataframe(topics_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_selection('single')
            grid_options = gb.build()

            # Mostrar la grilla con AgGrid
            response = AgGrid(
                topics_df,
                gridOptions=grid_options,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                width=1400,
            )  

            # Obtener la selección de la grilla
            selected_rows = response['selected_rows'] if response['selected_rows'] is not None else []

            # Mostrar los datos de la fila seleccionada
            if len(selected_rows) > 0:
            
                idx = selected_rows["indice"].values
                closest_id_news = selected_rows["id noticia"].values # id noticia mas cercana

                st.subheader(f":orange[tópico seleccionado:] {data_topics[idx[0]][0]}")
                st.write(f"<span style='color:#EDB70D'>Noticia mas cercana:</span> {data_topics[idx[0]][1]}", unsafe_allow_html=True)
                
                html_code, css = text_format(data_topics[idx[0]][2],[ent for ent, _ in data_topics[idx[0]][3].items()],
                                                                    [key for key, _ in data_topics[idx[0]][4].items()]) 
                st.markdown(css, unsafe_allow_html=True )
                st.markdown(html_code, unsafe_allow_html=True )

                st.text_area(":orange[Entidades del tópico seleccionado]", " | ".join([ent for ent, _ in data_topics[idx[0]][3].items()]), height=50)

                st.text_area(":orange[Palabras clave del tópico seleccionado]", " | ".join([key for key, _ in data_topics[idx[0]][4].items()]), height=50)

                st.subheader(f":orange[otras noticias relacionadas al tópico] : {data_topics[idx[0]][0]}")
                
                # Buscamos las noticias relacionadas al topico seleccionado
                id_news, title_news, text_news, probs = select_data_from_news(idx[0])

                # eliminamos la noticia mas cercana y ordenamos
                id_news, title_news, text_news, probs = umbral_and_delete_closest_id_news(id_news, title_news, text_news, probs, data_topics[idx[0]][5], closest_id_news[0])

                # Ajuste de noticias del topico  
                #id_news, title_news, probs = K_filter(id_news, title_news, text_news, probs, data_topics[idx[0]][0])
                
                # Preparamos otros documentos
                others_docs = []
                for i in range(len(id_news)):
                    others_docs.append([id_news[i], title_news[i], probs[i] ]) 

                df_other_docs = pd.DataFrame(others_docs, columns=['indice', 'titulo', 'prob']) 

                # Configurar opciones de la grilla
                gbo = GridOptionsBuilder.from_dataframe(df_other_docs)

                gbo.configure_pagination(paginationAutoPageSize=True)
                gbo.configure_column('indice', width=120) 
                gbo.configure_column('titulo', width=1280)
                gbo.configure_column('prob', width=120)
                gbo.configure_selection('single')
                
                grid_options_o = gbo.build()

                # Mostrar la grilla con AgGrid
                response_2 = AgGrid(
                    df_other_docs,
                    gridOptions=grid_options_o,
                    enable_enterprise_modules=True,
                    allow_unsafe_jscode=True,
                    fit_columns_on_grid_load=True,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    height=350,
                    width=1400,
                ) 
            
                # Obtener la selección de la grilla
                selected_rows_2 = response_2['selected_rows'] if response_2['selected_rows'] is not None else []

                # Mostrar los datos de la fila seleccionada
                if len(selected_rows_2) > 0:

                    idx_2 = selected_rows_2["indice"].values

                    index_name = 'news'
                    search_query = {
                        'query': {
                            'match': {
                                '_id': int(idx_2)
                            }
                        }
                    }

                    # Realizar la búsqueda
                    response = os_client.search(
                                                body=search_query,
                                                index=index_name
                    )
                    
                    st.write(f":orange[Texto de la noticia] | {response['hits']['hits'][0]['_source']['author']}" \
                            f" | {response['hits']['hits'][0]['_source']['created_at'][:10]}")

                    html_code, css = text_format(response['hits']['hits'][0]['_source']['news'],
                                                [ent for ent in response['hits']['hits'][0]['_source']['entities']],
                                                [key for key in response['hits']['hits'][0]['_source']['keywords']]) 
                
                    st.markdown(css, unsafe_allow_html=True )
                    st.markdown(html_code, unsafe_allow_html=True )
 
                    st.text_area(f":orange[Keywords de la noticia]", " | ".join([ent for ent in response['hits']['hits'][0]['_source']['keywords']]) ) 
                    st.text_area(f":orange[Entidades de la noticia]", " | ".join([ent for ent in response['hits']['hits'][0]['_source']['entities']]) )  


    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        st.write("Error en el proceso.")

    return

#----------------------------------------------------------------------------------
def umbral_and_delete_closest_id_news(id_news, title_news, text_news, probs_news, umbral, closest_id_news):

    id_news = np.array(id_news).reshape(-1,1)
    title_news = np.array(title_news).reshape(-1,1)
    text_news = np.array(text_news).reshape(-1,1)
    probs = np.array(probs_news, dtype=float ).reshape(-1,1)

    combined_array = np.hstack((id_news, title_news, text_news, probs))
    combined = pd.DataFrame(combined_array, columns=['ID','title','news','probs'])
    combined_clean = combined[combined['ID'] != str(closest_id_news)]
    combined_final = combined_clean[combined_clean["probs"].astype(float) >= float(umbral)].sort_values('probs', ascending=False)

    id_news     = combined_final['ID'].tolist()
    title_news  = combined_final['title'].tolist()
    text_news  = combined_final['title'].tolist()
    probs_news  = combined_final['probs'].tolist()

    return id_news, title_news, text_news, probs_news
#----------------------------------------------------------------------------------
def K_filter(id_news, title_news, text_news, prob_news, title_topic):

    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')
    
    # Stopwords
    SPANISH_STOPWORDS = list(pd.read_csv(PATH+'spanish_stop_words.csv' )['stopwords'].values)
    SPANISH_STOPWORDS_SPECIAL = list(pd.read_csv(PATH+'spanish_stop_words_spec.csv' )['stopwords'].values)

    count_vectorizer = CountVectorizer(
                    tokenizer=None,
                    token_pattern=r'(?u)\b\w\w+\b',
                    encoding ='utf-8',
                    ngram_range=(1, 2),
                    max_df=0.8, # significa que cualquier término que aparezca en más del 0.x % de los documentos será ignorado.
                    min_df=2,
                    stop_words=SPANISH_STOPWORDS,
                    # vocabulary=all_tokens,
                    )

    clean_data = Cleaning_text()

    proc_data = []
    for data_in in text_news:
        aux = clean_data.unicode(data_in)
        aux = clean_data.urls(aux)
        aux = clean_data.simbols(aux)
        aux = clean_data.escape_sequence(aux)
        aux = clean_data.str_lower(aux)
        aux = " ".join([ word for word in aux.split() if word.lower() not in SPANISH_STOPWORDS_SPECIAL])
        proc_data.append(aux)

    corpus_vect = count_vectorizer.fit_transform(proc_data)

    sim = cosine_similarity(corpus_vect, corpus_vect)

    # Aplica K-means clustering para dividir en 2 grupos
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(sim)

    # Obtiene las etiquetas de los clusters
    labels = kmeans.labels_

    # Obtenemos los dos grupos
    A_ids    = [ ID for i, ID in enumerate(id_news) if labels[i] == 1 ]
    A_titles = [ title for i, title in enumerate(title_news) if labels[i] == 1 ]
    A_probs  = [ prob for i, prob in enumerate(prob_news) if labels[i] == 1 ]

    B_ids    = [ ID for i, ID in enumerate(id_news) if labels[i] == 0 ]
    B_titles = [ title for i, title in enumerate(title_news) if labels[i] == 0 ]
    B_probs  = [ prob for i, prob in enumerate(prob_news) if labels[i] == 0 ]

    # Determinar a que grupo pertenece el titulo del topico
    corpus = [title_topic] + A_titles + B_titles

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Calcular la similitud del coseno
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Obtener las similitudes del titulo de la noticia con cada cadena en los grupos
    similarities_A = similarity_matrix[0, 1:len(A_titles)+1]  # Similitudes con A
    similarities_B = similarity_matrix[0, len(A_titles)+1:]  # Similitudes con B

    # Calcular la similitud promedio para cada grupo
    avg_similarity_A = np.mean(similarities_A)
    avg_similarity_B = np.mean(similarities_B)

    

    if avg_similarity_A > avg_similarity_B:
        print("\nGrupo Descarte B:", (avg_similarity_A, avg_similarity_B))
        for i in B_titles[:5]:
            print(i)
        return A_ids, A_titles, A_probs
        
    if avg_similarity_A < avg_similarity_B:
        print("\nGrupo Descarte A:", (avg_similarity_A, avg_similarity_B))
        for i in A_titles[:5]:
            print(i)
        return B_ids, B_titles, B_probs
    
    else:
        print("\nSin Descarte:", (avg_similarity_A, avg_similarity_B))

        return A_ids+B_ids, A_titles+B_titles, A_probs+B_probs

#----------------------------------------------------------------------------------
def control():

    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    index_name = 'topic'

    db_topics = []
    for i, doc in enumerate(Topic.search().query().scan()):
        db_topics.append(doc.to_dict())

    db_view = [ item['name'] for item in db_topics]

    # Crear una caja de selección múltiple
    seleccionados = st.multiselect("Seleccionar dos o más opciones para agrupar:", db_view)

    # Boton para agrupar
    if st.button("Agrupar topicos"):

        # Mostrar las opciones seleccionadas
        if seleccionados:
            st.write("Has seleccionado:", seleccionados)


        else:
            st.write("No has seleccionado ninguna opción.")

    else:

        st.stop()

    return
#--------------------------------------------------
def format_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
    return formatted_date
#--------------------------------------------------
def last_topic_date():
    topics = get_topics_opensearch()
    topics_dates = []
    if topics:
        for reg in topics:
            topics_dates.append(reg['to_date'][:10])
        dates = list(set(topics_dates))
        
        # Convertir las fechas a objetos datetime
        date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

        # Ordenar los objetos datetime
        date_objects.sort()
        
        return date_objects[-1]
    else:
        return datetime.now()
#-------------------------------------------------
def text_format(texto: str, keywords: list, entities: list):

    # Función para resaltar las palabras en el texto
    def resaltar_palabras(texto, palabras, color):
        def reemplazo(m):
            palabra_original = m.group(0)  # Obtiene la palabra original encontrada
            return f'<span style="color:{color}">{palabra_original}</span>'

        # Crea una expresión regular que coincida con cualquiera de las palabras en la lista
        patron = '|'.join([re.escape(palabra) for palabra in palabras])
        
        # Reemplaza todas las ocurrencias de las palabras en el patrón
        texto_resaltado = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)
        return texto_resaltado
    
    if keywords:
        texto_resaltado = resaltar_palabras(texto, keywords, "orange")
    
    if entities:
        texto_resaltado = resaltar_palabras(texto_resaltado, entities, "orange")

    if not keywords and not entities:
        texto_resaltado = texto

    # Estilo CSS para hacer el texto scrollable
    css = """
    <style>
    .scrollable-text {
        height: 400px;
        overflow-x: scroll;
        padding: 10px;
        
        border-radius: 8px; /* Esquinas redondeadas del recuadro */
        background-color: #1E2020; /* Color de fondo del recuadro */
    }
    </style>
    """

    # Contenedor HTML con clase CSS
    html_code = f"""
    <div class="scrollable-text">
        {texto_resaltado}
    </div>
    """

    return html_code, css

#---------------------------------------------------------------------------
def merged_results(topic_model_1, merged_model):

    topic_freq_1 = topic_model_1.get_topic_freq()
    topic_freq_m = merged_model.get_topic_freq()

    df1 = topic_freq_1.sort_values(by='Topic').reset_index(drop=True)
    dfm = topic_freq_m.sort_values(by='Topic').reset_index(drop=True)

    # Renombrar las columnas 'Count' para diferenciar DataFrames
    df1 = df1.rename(columns={'Count': 'Count1'})
    dfm = dfm.rename(columns={'Count': 'Merged'})

    df_combined = pd.merge(df1, dfm, on='Topic', how='outer')

    # Calcular la nueva columna 'Count2' como la resta de 'Merged' y 'Count1'
    # Asegurarse de manejar NaN correctamente
    df_combined['Count2'] = df_combined['Merged'].fillna(0) - df_combined['Count1'].fillna(0)

    # Reordenar las columnas en el orden deseado
    df_combined = df_combined[['Topic', 'Count1', 'Count2', 'Merged']]

    return df_combined

#----------------------------------------------------------------------
def update_topics_date(from_date, to_date, df_combined, fechas_dict):

    # completamos con las fechas de los nuevos topicos
    while len(from_date) < len(df_combined):
        from_date.append(next(iter(fechas_dict)))
        to_date.append(datetime.strptime(next(iter(fechas_dict)), '%Y-%m-%d') + timedelta(days=1)) 

    # modifcamos la fecha del topico que contenga nuevos documentos
    for i in range(0, len(df_combined)-1):
        if df_combined.iloc[i+1]['Count2'] != 0:
            to_date[i] = datetime.strptime(next(iter(fechas_dict)), '%Y-%m-%d') + timedelta(days=1)

    return from_date, to_date
#---------------------------------------------------------------------
def delete_model():
    """
    Elimina los modelos si no existen topicos en el indice topic
    """
    load_dotenv()
    PATH=os.environ.get('PATH_LOCAL')

    if not get_topics_opensearch():
        files = ["modelos/bertopic_model_app", "modelos/bertopic_model_app_old"]
        for file_name in files:
            try:
                # Borrar modelos 
                os.remove(PATH+file_name)
                print(f"Modelo {file_name} eliminado")  
            except FileNotFoundError:
                print(f"El archivo {file_name} no existe.")
            except PermissionError:
                print(f"No tienes permiso para borrar el archivo {file_name}.")
            except Exception as e:
                print(f"Ocurrió un error al intentar borrar el archivo {file_name}: {e}")
    return

    
