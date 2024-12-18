import os
from dotenv import load_dotenv

url_raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
url_dotenv = os.path.join(url_raiz_projeto, ".env")
load_dotenv(url_dotenv)

class Environment:
    def __init__(self):
        self.URL_BANCO_VETORES = os.getenv('URL_BANCO_VETORES')
        self.URL_LLAMA=os.getenv('URL_LLAMA')
        self.URL_HOST=os.getenv('URL_HOST')
        self.TAGS_SUBSTITUICAO_HTML={
            'TAG_INSERCAO_URL_HOST': self.URL_HOST,
            'TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM': os.getenv('TAG_INSERCAO_FLAG_ENCERRAMENTO_MENSAGEM')
            }

        self.THREADPOOL_MAX_WORKERS=int(os.getenv('THREADPOOL_MAX_WORKERS'))
        self.NOME_COLECAO_DE_DOCUMENTOS=os.getenv('COLECAO_DE_DOCUMENTOS')
        self.EMBEDDING_INSTRUCTOR=os.getenv('EMBEDDING_INSTRUCTOR')
        self.EMBEDDING_SQUAD_PORTUGUESE=os.getenv('EMBEDDING_SQUAD_PORTUGUESE')
        self.MODELO_LLAMA=os.getenv('MODELO_LLAMA')
        self.DEVICE=os.getenv('DEVICE') # ['cpu', cuda']
        self.NUM_DOCUMENTOS_RETORNADOS=int(os.getenv('NUM_DOCUMENTOS_RETORNADOS'))

        self.MODELO_DE_EMBEDDINGS = self.EMBEDDING_INSTRUCTOR

        self.CONTEXTO_BASE = []

        self.DOCUMENTOS =  {
            'lei_maria_da_penha': {
                'url': 'datasets/lei_maria_da_penha.txt',
                'titulo': 'LEI Nº 11.340, DE 7 DE AGOSTO DE 2006 - Lei Maria da Penha',
                'autor': 'Governo Federal - República Federativa do Brasil',
                'fonte': 'http://www.planalto.gov.br/ccivil_03/_ato2004-2006/2006/lei/l11340.htm'
            }
        }

environment = Environment()