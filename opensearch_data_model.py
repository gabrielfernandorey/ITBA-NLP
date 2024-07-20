from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, Boolean, DenseVector, Nested, Date, Object, connections, InnerDoc, helpers
import os

# local test
# docker pull opensearchproject/opensearch:latest
# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=PassWord#1234! -e "discovery.type=single-node"  --name opensearch-node opensearchproject/opensearch:latest
# docker stop opensearch-node
# docker start opensearch-node
# curl -X GET "https://localhost:9200" -ku admin:PassWord#1234!
# Al usar el plugin de Chrome, primero acceder al link --> When using https make sure that your browser trusts the clusters ssl certificate. Help
# y luego configurar el acceso: abrir en navegador https://localhost:9200, luego loguearse.
# Nota: verificar "https://...."



OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', "localhost")
auth = ('admin', 'PassWord#1234!')
port = 9200
os_client = connections.create_connection(
    hosts = [{'host': OPENSEARCH_HOST, 'port': port}],
    http_auth = auth,
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    alias='default'
    # ssl_assert_hostname = False,
    # ssl_show_warn = False
)

# Index Topic
TOPIC_DIMENSIONS = 384
TOPIC_INDEX_NAME = 'topic'
TOPIC_INDEX_PARAMS = {
    'number_of_shards': 1,      # 1 fragmento (no hay replicas)
    'knn': True                 # aplicamos knn para que la base utilice como algoritmo de clusterizacion
}

# Index News
NEWS_INDEX_NAME = 'news'
NEWS_INDEX_PARAMS = {
    'number_of_shards': 1,      # 1 fragmento (no hay replicas)
    'knn': True                 # aplicamos knn para que la base utilice como algoritmo de clusterizacion
}


knn_params = {                      # parámetros de knn
    "name": "hnsw",
    "space_type": "cosinesimil",    # usamos similitud coseno
    "engine": "nmslib"              # tipo de algortimo que va a utilizar
}
#-------------------------------------------------------------------------------------------------
class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class TopicEntities(InnerDoc):
    name = Keyword()
    score = Float()

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class DocsProbs(InnerDoc):
    doc_ID = Keyword()
    score = Float()


class Topic(Document):
    index = Integer()                                   # nro topico
    name = Text()                                       # nombre del topico
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)    # vector
    similarity_threshold = Float()                      # umbral
    created_at = Date()                                 # fecha de creacion
    to_date = Date()                                    # fecha de entrenamiento
    from_date = Date()                                  # fecha de entrenamiento
    keywords = Object(TopicKeyword)                     # keywords del topico (representacion) 
    entities = Object(TopicEntities)                    # entities del topico
    id_best_doc = Integer()                             # id del mejor documento
    title_best_doc = Text()                             # titulo del mejor documento
    best_doc = Text()                                   # texto del mejor documento
    docs = Object(DocsProbs)                            # documentos del tópico y probs ( filtrado por similarity_threshold )
    
    class Index:
        name = TOPIC_INDEX_NAME
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + '_' + self.name.replace(' ', '_')
        return super(Topic, self).save(** kwargs)
    
#-------------------------------------------------------------------------------------------------------------
class News(Document):
    title = Text()
    news = Text()
    author = Text()
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)    # vector
    entities = Keyword()
    keyboards = Keyword()
    created_at = Date() 
    process = Boolean()                         

    class Index:
        name = NEWS_INDEX_NAME                          
        if not os_client.indices.exists(index=NEWS_INDEX_NAME):
            settings = {
                'index': NEWS_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}'
        return super(News, self).save(** kwargs)


    