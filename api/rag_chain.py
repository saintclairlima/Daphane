from api.environment.environment import environment

from langchain_chroma import Chroma # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.schema import RunnableMap, RunnableLambda

class RAGChain:
    def __init__(self, url_banco_vetores, colecao_de_documentos, funcao_de_embeddings):
        

        self.retriever = Chroma(
            persist_directory=url_banco_vetores,
            collection_name=colecao_de_documentos,
            embedding_function=funcao_de_embeddings
        ).as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": environment.NUM_DOCUMENTOS_RETORNADOS,
                "include": ["documents", "metadatas", "distances", "ids"]
            }
        )
        
        self.cliente_ollama = ChatOllama(
            model=environment.MODELO_LLAMA,
            temperature=0,
            base_url=environment.URL_LLAMA
        )
        
        self.prompt = ChatPromptTemplate.from_messages(
            [    # Estabelece o papel que o LLM vai assumir ao responder as perguntas. Pode incluir o "tom" das respostas
                ('system', self.papel_do_LLM + self.diretrizes),  

                # Placeholder para o histórico do chat manter o contexto. Durante a execução será substituído pelo histórico real do chat
               ("placeholder", "{chat_history}"), 

                # Placeholder para o input a ser fornecido durante a execução
                # Será substituído pela pergunta do usuário e o contexto vindo do banco de vetores
                ("human", "DOCUMENTOS: {documentos} \nPERGUNTA: {pergunta}"),
            ]
        )

        self.formatador_saida = StrOutputParser()

        self.rag_chain = (
            RunnableMap({
                # Retrieve and format documents
                "resultados_busca": RunnableLambda(self.recuperar_documentos),
                
                # Generate LLM response using formatted documents
                "llm_response": {
                    "documentos": RunnableLambda(lambda inputs: inputs["resultados_busca"]["documentos_formatados"]),
                    "pergunta": RunnablePassthrough(),
                }
                | self.prompt
                | self.cliente_ollama
                | self.formatador_saida,
            })
        )
    
    def formatar_documentos_recuperados(self, documentos):
        return "\n".join([documento.page_content for documento in documentos])
    
    def recuperar_documentos(self, inputs):
        # Run the retriever and format the documents
        documentos_recuperados = self.retriever.invoke(inputs["pergunta"])
        documentos_formatados = self.formatar_documentos_recuperados(documentos_recuperados)
        return {
            "documentos_recuperados": documentos_recuperados,
            "documentos_formatados": documentos_formatados
        }

    def consultar(self, pergunta):
        resultado = self.rag_chain.invoke({"pergunta": pergunta})
        documentos = resultado['resultado_busca']["documentos_recuperados"]
        resposta_llm = resultado["llm_response"]

class Environment:
    def __init__(self):
        self.URL_BANCO_VETORES = os.getenv('URL_BANCO_VETORES')
        self.URL_LLAMA=os.getenv('URL_LLAMA')
        self.NOME_COLECAO_DE_DOCUMENTOS=os.getenv('COLECAO_DE_DOCUMENTOS')
        self.EMBEDDING_INSTRUCTOR=os.getenv('EMBEDDING_INSTRUCTOR')
        self.EMBEDDING_SQUAD_PORTUGUESE=os.getenv('EMBEDDING_SQUAD_PORTUGUESE')
        self.MODELO_LLAMA=os.getenv('MODELO_LLAMA')
        self.DEVICE=os.getenv('DEVICE') # ['cpu', cuda']
        self.NUM_DOCUMENTOS_RETORNADOS=int(os.getenv('NUM_DOCUMENTOS_RETORNADOS'))

        self.MODELO_DE_EMBEDDINGS = self.EMBEDDING_INSTRUCTOR

        self.CONTEXTO_BASE = []

        with open(os.getenv('URL_INDICE_DOCUMENTOS'), 'r') as arq:
            self.DOCUMENTOS = json.load(arq)