# ITBA-NLP

### Trabajo Práctico NLP Detección de Tópicos y clasificación

### Instrucciones
  1. Abrir terminal y clonar el repositorio: ```git clone https://github.com/gabrielfernandorey/ITBA-NLP.git```
  2. Crear un ambiente virtual: ```virtualenv itba-nlp-env```
  3. Activar ambiente virtual: ```.\itba-nlp-env\Scripts\activate```
  4. Cambiar a carpeta ITBA-NLP e instalar dependencias: ```pip install -r requirements.txt```
  5. Instalar Docker
  6. Instalacion de OpenSearch:
     -  ```docker pull opensearchproject/opensearch:latest```
     -  ```docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=PassWord#1234! -e "discovery.type=single-node"  --name opensearch-node opensearchproject/opensearch:latest```
     

