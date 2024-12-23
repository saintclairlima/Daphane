import os
import sys
from ..environment.environment import environment
from ..utils.utils import FuncaoEmbeddings
from torch import cuda

from sentence_transformers import SentenceTransformer
from chromadb import chromadb
from pypdf import PdfReader
from bs4 import BeautifulSoup

URL_LOCAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
DEVICE='cuda' if cuda.is_available() else 'cpu'

# Valores padrão, geralmente não usados
NOME_BANCO_VETORES=os.path.join(URL_LOCAL,"bancos_vetores/banco_teste_default")
NOME_COLECAO='colecao_teste_default'
COMPRIMENTO_MAX_FRAGMENTO = 300    

class GeradorBancoVetores:
    
    def processar_texto_articulado(self, texto, info, comprimento_max_fragmento):
        '''Processa textos legais, divididos em artigos. Mantém o Caput dos artigos em cada um dos fragmentos.'''
        
        texto = texto.replace('\n', ' ')
        while '  ' in texto: texto = texto.replace('  ', ' ')
        texto = texto.replace(' Art. ', '\nArt. ')
        
        for num in range(1, 10):
                texto = texto.replace(f'Art. {num}º', f'Art. {num}.')
                texto = texto.replace(f'art. {num}º', f'art. {num}.')
                texto = texto.replace(f'§ {num}º', f'§ {num}.')
                
        texto = texto.split('\n')
        while '' in texto: texto.remove('')
        
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
                # AFAZER: considerar casos em que, mesmo após divisão das
                # partes do artigo, haja alguma com mais palavras que o compr. máximo
                for i in range(1, len(item)):
                    if len(fragmento_artigo.split(' ')) + len(item[i].split(' ')) <= comprimento_max_fragmento:
                        fragmento_artigo = fragmento_artigo + ' ' + item[i]
                    else:
                        artigos.append(fragmento_artigo)
                        fragmento_artigo = '' + caput + ' ' + item[i]
                artigos.append(fragmento_artigo)
            else:
                artigos.append(art)
        
        fragmentos = []
        titulos = []
        for artigo in artigos:
            tit = artigo.split('. ')[1]
            titulos.append(tit)
            fragmento = {
                'page_content': artigo,
                'metadata': {
                    'titulo': f'{info["titulo"]}',
                    'subtitulo': f'Art. {tit} - {titulos.count(tit)}',
                    'autor': f'{info["autor"]}',
                    'fonte': f'{info["fonte"]}',
                    'pagina': None
                },
            }
            fragmentos.append(fragmento)
        return fragmentos
    
    def processar_texto(self, texto, info, comprimento_max_fragmento, pagina=None):
        texto = texto.replace('\n', ' ').replace('\t', ' ')
        while '  ' in texto: texto = texto.replace('  ', ' ')
        
        if info['texto_articulado']:
            return self.processar_texto_articulado(texto, info, comprimento_max_fragmento)
        
        if len(texto.split(' ')) <= comprimento_max_fragmento:
            return [{
                    'page_content': texto,
                    'metadata': {
                        'titulo': f'{info["titulo"]}',
                        'subtitulo':
                            f'Página {pagina} - Fragmento 1' if pagina
                            else f'Fragmento 1',
                        'autor': f'{info["autor"]}',
                        'fonte': f'{info["fonte"]}',
                        'pagina': pagina if pagina else None
                    },
                }]
            
        linhas = texto.replace('. ', '.\n')
        linhas = linhas.split('\n')
        while '' in linhas: linhas.remove('')
        
        fragmentos = []
        texto_fragmento = ''
        for idx in range(len(linhas)):
            if len(texto_fragmento.split(' ')) + len(linhas[idx].split(' ')) < comprimento_max_fragmento:
                texto_fragmento += ' ' + linhas[idx]
            else:
                fragmento = {
                    'page_content': texto_fragmento,
                    'metadata': {
                        'titulo': f'{info["titulo"]}',
                        'subtitulo':
                            f'Página {pagina} - Fragmento {len(fragmentos)+1}' if pagina
                            else f'Fragmento {len(fragmentos)+1}',
                        'autor': f'{info["autor"]}',
                        'fonte': f'{info["fonte"]}',
                        'pagina': pagina if pagina else None
                    },
                }
                fragmentos.append(fragmento)
                texto_fragmento = ''
            
        return fragmentos       
    
    def extrair_fragmento_txt(self, rotulo, info, comprimento_max_fragmento):
        with open(os.path.join(URL_LOCAL,info['url']), 'r') as arq:
            texto = arq.read()
        
        fragmentos = self.processar_texto(texto, info, comprimento_max_fragmento)
        
        for idx in range(len(fragmentos)): fragmentos[idx]['id'] = f'{rotulo}:{idx+1}'
        return fragmentos
    
    def extrair_fragmento_pdf(self, rotulo, info, comprimento_max_fragmento):
        fragmentos = []
        arquivo = PdfReader(os.path.join(URL_LOCAL,info['url']))
        for idx in range(len(arquivo.pages)):
            pagina = arquivo.pages[idx]
            texto = pagina.extract_text()
            fragmentos += self.processar_texto(texto, info, comprimento_max_fragmento, pagina=idx+1)
        for idx in range(len(fragmentos)): fragmentos[idx]['id'] = f'{rotulo}:{idx+1}'
        return fragmentos
        
    def extrair_fragmento_html(self, rotulo, info, comprimento_max_fragmento):
        with open(os.path.join(URL_LOCAL,info['url']), 'r', encoding='utf-8') as arq:
            conteudo_html = arq.read()
            
        pagina_html = BeautifulSoup(conteudo_html, 'html.parser')
        tags = pagina_html.find_all()
        texto = '\n'.join([tag.get_text() for tag in tags])
        
        fragmentos = self.processar_texto(texto, info, comprimento_max_fragmento)
        for idx in range(len(fragmentos)): fragmentos[idx]['id'] = f'{rotulo}:{idx+1}'
        return fragmentos
    
    extrair_fragmento_por_tipo = {
        'txt':  extrair_fragmento_txt,
        'pdf':  extrair_fragmento_pdf,
        'html': extrair_fragmento_html,
    }    
    
    def extrair_fragmentos(self,
        indice_documentos=environment.DOCUMENTOS,
        comprimento_max_fragmento=COMPRIMENTO_MAX_FRAGMENTO):
        fragmentos = []
        for rotulo, info in indice_documentos.items():
            print(f'Processando {rotulo}')
            url = info['url']
            tipo = url.split('.')[-1]
            fragmentos += self.extrair_fragmento_por_tipo[tipo](self, rotulo=rotulo, info=info, comprimento_max_fragmento=comprimento_max_fragmento)
        
        return fragmentos
        
    
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
        
    def run(self,
            nome_banco_vetores=NOME_BANCO_VETORES,
            nome_colecao=NOME_COLECAO,
            comprimento_max_fragmento=COMPRIMENTO_MAX_FRAGMENTO,
            instrucao=None):
        
        docs = self.extrair_fragmentos(
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
