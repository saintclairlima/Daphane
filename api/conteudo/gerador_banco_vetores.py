import os
import sys
from ..environment.environment import environment
from ..utils.utils import FuncaoEmbeddings
from torch import cuda

from sentence_transformers import SentenceTransformer
from chromadb import chromadb

URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
DEVICE='cuda' if cuda.is_available() else 'cpu'

# Valores padrão, geralmente não usados
NOME_BANCO_VETORES=os.path.join(URL_LOCAL,"bancos_vetores/banco_teste_default")
NOME_COLECAO='colecao_teste_default'
COMPRIMENTO_MAX_FRAGMENTO = 300    

class GeradorBancoVetores:
    def obter_documentos_env(self, comprimento_max_fragmento=COMPRIMENTO_MAX_FRAGMENTO):
        documentos = []
        titulos = []
        id=1

        for k, v in environment.DOCUMENTOS.items():
            URL_DADOS = os.path.join(URL_LOCAL, v['url'])
            print(f'''Lendo o arquivo {URL_DADOS}...''')
            with open(URL_DADOS, 'r', encoding='UTF-8') as arq:
                texto = arq.read()
            # Listagens até 9, em português, são marcadas por números ordinais. Isso faz com que
            # posteriormente os números ordinais recebam mais atenção do que deveriam, na
            # representação TF-IDF. O código abaixo remove a marcação de ordinais e coloca a
            # mesma notação utilizada nos demais itens
            for num in range(1, 10):
                texto = texto.replace(f'Art. {num}º', f'Art. {num}.')
                texto = texto.replace(f'art. {num}º', f'art. {num}.')
                texto = texto.replace(f'§ {num}º', f'§ {num}.')
            texto = texto.split('\n')
            if '' in texto: texto.remove('')
            print(f'''Dividindo em artigos...''')
            artigos = []
            for art in texto:
                item = art.split(' ')
                qtd_palavras = len(item)
                if qtd_palavras > comprimento_max_fragmento:
                    item = (
                        art.replace('. §', '.\n§')
                        .replace('; §', ';\n§')
                        .replace(': §', ':\n§')
                        .replace(';', '\n')
                        .replace(':', '\n')
                        .replace('\n ', '\n')
                        .replace(' \n', '\n')
                        .split('\n')
                    )
                    caput = item[0]
                    fragmento_artigo = '' + caput
                    for i in range(1, len(item)):
                        if len(fragmento_artigo.split(' ')) + len(item[i]) <= comprimento_max_fragmento:
                            fragmento_artigo = fragmento_artigo + ' ' + item[i]
                        else:
                            artigos.append(fragmento_artigo)
                            fragmento_artigo = '' + caput + ' ' + item[i]
                    artigos.append(fragmento_artigo)
                else:
                    artigos.append(art)
            print('''Criando os documentos com base nos artigos...''')
            for artigo in artigos:
                tit = artigo.split('. ')[1]
                titulos.append(tit)
                doc = {
                    'id': id,
                    'page_content': artigo,
                    'metadata': {
                        'titulo': f'{v["titulo"]}',
                        'subtitulo': f'Art. {tit} - {titulos.count(tit)}',
                        'autor': f'{v["autor"]}',
                        'fonte': f'{v["fonte"]}',
                    },
                }
                documentos.append(doc)
                id += 1
        return documentos
    
    def gerar_banco(self,
            documentos,
            nome_banco_vetores=NOME_BANCO_VETORES,
            nome_colecao=NOME_COLECAO,
            instrucao=None):
        
        # Utilizando o ChromaDb diretamente
        client = chromadb.PersistentClient(path=nome_banco_vetores)
        
        funcao_de_embeddings_sentence_tranformer = FuncaoEmbeddings(
            nome_modelo=EMBEDDING_INSTRUCTOR,
            tipo_modelo=SentenceTransformer,
            device=DEVICE,
            instrucao=instrucao)
        
        collection = client.create_collection(name=nome_colecao, embedding_function=funcao_de_embeddings_sentence_tranformer, metadata={'hnsw:space': 'cosine'})
        
        print(f'Gerando >>> Banco {nome_banco_vetores} - Coleção {nome_colecao} - Instrução: {instrucao}')
        qtd_docs = len(documentos)
        for idx in range(qtd_docs):
            print(f'\r>>> Incluindo documento {idx+1} de {qtd_docs}', end='')
            doc = documentos[idx]
            collection.add(
                documents=[doc['page_content']],
                ids=[str(doc['id'])],
                metadatas=[doc['metadata']],
            )
        client._system.stop()

        #query_result = collection.query(query_texts=["O que é uma legislatura?"], n_results=5)
        #print(query_result)

        # client.get_collection(name='legisberto', embedding_function=funcao_de_embeddings_sentence_tranformer)
    def run(self,
            nome_banco_vetores=NOME_BANCO_VETORES,
            nome_colecao=NOME_COLECAO,
            comprimento_max_fragmento=COMPRIMENTO_MAX_FRAGMENTO,
            instrucao=None):
        
        docs = self.obter_documentos_env(
            comprimento_max_fragmento=comprimento_max_fragmento
        )
        
        self.gerar_banco(
            documentos=docs,
            nome_banco_vetores=nome_banco_vetores,
            nome_colecao=nome_colecao,
            instrucao=instrucao
        )
        
        
if __name__ == "__main__":   
    gerador_banco_vetores = GeradorBancoVetores()
    
    nome_banco_vetores=os.path.join(URL_LOCAL,"bancos_vetores/" + sys.argv[1])
    nome_colecao=sys.argv[2]
    comprimento_max_fragmento = int(sys.argv[3])
    try:
        instrucao = sys.argv[4]
        gerador_banco_vetores.run(
            nome_banco_vetores=nome_banco_vetores,
            nome_colecao=nome_colecao,
            comprimento_max_fragmento=comprimento_max_fragmento,
            instrucao=instrucao)
    except:
        gerador_banco_vetores.run(
            nome_banco_vetores=nome_banco_vetores,
            nome_colecao=nome_colecao,
            comprimento_max_fragmento=comprimento_max_fragmento,
            instrucao=None)
    
    
