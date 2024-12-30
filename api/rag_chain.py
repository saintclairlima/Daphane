#from api.environment.environment import environment
import argparse
import json
from langchain_chroma import Chroma # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables.base import RunnableMap, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

class CustomStreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")
        # You can add any custom logic here (e.g., update a UI component)

class RAGChain:
    def __init__(self, url_banco_vetores=None, colecao_de_documentos=None, funcao_de_embeddings=None, streaming_callback=None):
        
        environment=Environment()
        if not url_banco_vetores: url_banco_vetores = environment.URL_BANCO_VETORES
        if not colecao_de_documentos: colecao_de_documentos = environment.COLECAO_DE_DOCUMENTOS
        if not funcao_de_embeddings: funcao_de_embeddings = HuggingFaceEmbeddings(
            model_name=environment.MODELO_DE_EMBEDDINGS,
            show_progress=False,
            model_kwargs={'device': environment.DEVICE}
        )
        if not streaming_callback: streaming_callback = CustomStreamingCallbackHandler()
            
        print('Inicialização do componente')
        print(url_banco_vetores)
        print(colecao_de_documentos)
        
        self.retriever = Chroma(
            persist_directory=url_banco_vetores,
            collection_name=colecao_de_documentos,
            embedding_function=funcao_de_embeddings
        ).as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": environment.NUM_DOCUMENTOS_RETORNADOS,
                "include": ["documents", "metadatas", "distances",]
            }
        )

        self.papel_do_LLM='''Você é uma assistente que responde a dúvidas de mulheres sobre a Lei Maria da Penha.
Assuma um tom formal, porém caloroso, com gentileza nas respostas. Utilize palavras e termos que sejam claros, autoexplicativos e linguagem simples, próximo do que o cidadão comum utiliza.Use as informações dos DOCUMENTOS fornecidos para gerar uma resposta clara para a PERGUNTA.
Na resposta, não mencione que foi fornecido documentos de referência. Cite os nomes dos DOCUMENTOS e números dos artigos em que a resposta se baseia.
A resposta não deve ter saudação, vocativo, nem qualquer tipo de introdução que dê a entender que não houve interação anterior.
Se você não souber a resposta, assuma um tom gentil e diga que não tem informações suficientes para responder.'''
        
        self.cliente_ollama = ChatOllama(
            model=environment.MODELO_LLAMA,
            temperature=0,
            base_url=environment.URL_LLAMA,
            streaming=True,
            callbacks=[streaming_callback]
        )
        
        self.prompt = ChatPromptTemplate.from_messages(
            [    # Estabelece o papel que o LLM vai assumir ao responder as perguntas. Pode incluir o "tom" das respostas
                ('system', self.papel_do_LLM),  

                # Placeholder para o histórico do chat manter o contexto. Durante a execução será substituído pelo histórico real do chat
               ("placeholder", "{chat_history}"), 

                # Placeholder para o input a ser fornecido durante a execução
                # Será substituído pela pergunta do usuário e o contexto vindo do banco de vetores
                ("human", "DOCUMENTOS: {documentos} \nPERGUNTA: {pergunta}"),
            ]
        )

        self.rag_chain = (
            self.recuperar_documentos
            | RunnableMap({
                # Retrieve and format documents
                "documentos_recuperados": RunnableLambda(lambda inputs: inputs["documentos_recuperados"]),
                
                # Generate LLM response using formatted documents
                "llm_response": {
                    "documentos": RunnableLambda(lambda inputs: inputs["documentos_formatados"]),
                    "pergunta": RunnableLambda(lambda inputs: inputs["pergunta"]),
                }
                | self.prompt
                | self.cliente_ollama,
            })
        )
    
    def formatar_documentos_recuperados(self, documentos):
        return "\n".join([documento.page_content for documento in documentos])
    
    def recuperar_documentos(self, inputs):
        print('Recuperando documentos')
        documentos_recuperados = self.retriever.invoke(inputs["pergunta"])
        documentos_formatados = self.formatar_documentos_recuperados(documentos_recuperados)
        return {
            "pergunta": inputs["pergunta"],
            "documentos_recuperados": documentos_recuperados,
            "documentos_formatados": documentos_formatados
        }

    def consultar(self, pergunta):
        resultado = self.rag_chain.invoke({"pergunta": pergunta})
        documentos = resultado["documentos_recuperados"],
        resposta_llm = resultado["llm_response"]
        return {'documentos': documentos, "resposta": resposta_llm}

class Environment:
    def __init__(self):

        self.URL_BANCO_VETORES='api/conteudo/bancos_vetores/documentos_mulher'
        self.URL_INDICE_DOCUMENTOS='api/conteudo/datasets/index.json'
        self.COLECAO_DE_DOCUMENTOS='daphane'
        self.URL_LLAMA='http://localhost:11434'
        self.URL_HOST='http://localhost:8000'
        self.THREADPOOL_MAX_WORKERS=10
        self.EMBEDDING_INSTRUCTOR="hkunlp/instructor-xl"
        self.EMBEDDING_SQUAD_PORTUGUESE="pierreguillou/bert-base-cased-squad-v1.1-portuguese"
        self.MODELO_LLAMA='llama3.1'
        self.DEVICE='cuda'
        self.NUM_DOCUMENTOS_RETORNADOS=5
        
        self.MODELO_DE_EMBEDDINGS = self.EMBEDDING_INSTRUCTOR

        self.CONTEXTO_BASE = []

        with open(self.URL_INDICE_DOCUMENTOS, 'r') as arq:
            self.DOCUMENTOS = json.load(arq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testa a recuperação de documentos do banco vetorial e a formulação de uma resposta")
    
    parser.add_argument('--pergunta', type=str, required=True, help='pergunta a ser respondida')
    parser.add_argument('--url_banco_vetor', type=str, help="caminho para banco de vetores")
    parser.add_argument('--nome_colecao', type=str, help="nome da coleção a ser consultada")
    
    args = parser.parse_args()
    
    pergunta=args.pergunta
    url_banco_vetor = None if not args.url_banco_vetor else args.url_banco_vetor
    nome_colecao = None if not args.nome_colecao else args.nome_colecao
    ragchain = RAGChain(url_banco_vetores=url_banco_vetor, colecao_de_documentos=nome_colecao)
    print(ragchain.consultar(pergunta))