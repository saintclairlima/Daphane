import json
from time import time
from api.environment.environment import environment
import argparse
from langchain_chroma import Chroma # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables.base import RunnableMap, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from api.utils.utils import DadosChat

class CustomStreamingCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")

class RAGChain:
    def __init__(self, url_banco_vetores=None, colecao_de_documentos=None, funcao_de_embeddings=None, streaming_callback=None):
        
        if not url_banco_vetores: url_banco_vetores = environment.URL_BANCO_VETORES
        if not colecao_de_documentos: colecao_de_documentos = environment.NOME_COLECAO_DE_DOCUMENTOS
        if not funcao_de_embeddings: funcao_de_embeddings = HuggingFaceEmbeddings(
            model_name=environment.MODELO_DE_EMBEDDINGS,
            show_progress=False,
            model_kwargs={'device': environment.DEVICE}
        )
        if not streaming_callback: streaming_callback = CustomStreamingCallbackHandler()
            
        print('Inicialização do componente')
        print(url_banco_vetores)
        print(colecao_de_documentos)
        
        self.banco_vetores = Chroma(
            persist_directory=url_banco_vetores,
            collection_name=colecao_de_documentos,
            embedding_function=funcao_de_embeddings
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
            [                
                # Estabelece o papel que o LLM vai assumir ao responder as perguntas. Pode incluir o "tom" das respostas
                ('system', self.papel_do_LLM),

                # Placeholder para o histórico do chat manter o contexto. Durante a execução será substituído pelo histórico real do chat
                ('placeholder', '{contexto}'),
                
                # Placeholder para o input a ser fornecido durante a execução
                # Será substituído pela pergunta do usuário e o contexto vindo do banco de vetores
                ("human", "DOCUMENTOS: {documentos} \nPERGUNTA: {pergunta}"),
            ]
        )

        self.rag_chain = (
            self.recuperar_documentos
            | RunnableMap({
                "documentos_recuperados": RunnableLambda(lambda inputs: inputs["documentos_recuperados"]),
                "tempo_consulta": RunnableLambda(lambda inputs: inputs["tempo_consulta"]),
                "llm_response": self.prompt
                | RunnableLambda(self.inspecionar)
                | self.cliente_ollama,
            })
        )
    
    def formatar_documentos_recuperados(self, documentos):
        return "\n".join([documento['conteudo'] for documento in documentos])
    
    def recuperar_documentos(self, inputs):
        print('Recuperando documentos')
        tempo_inicio = time()
        documentos_recuperados = [
            {
                "conteudo": doc.page_content,
                "metadados": doc.metadata,
                "score_distancia": 1 - score
            }
            for doc, score in self.banco_vetores.similarity_search_with_score(inputs['pergunta'])
        ]
        tempo_consulta = time() - tempo_inicio
        documentos_formatados = self.formatar_documentos_recuperados(documentos_recuperados)
        inputs['documentos_recuperados'] = documentos_recuperados
        inputs['documentos'] = documentos_formatados
        inputs['tempo_consulta'] = tempo_consulta
        return inputs

    def inspecionar(self, inputs):
        print(f'Inputs:\n{inputs}')
        return inputs

    def consultar(self, dados_chat: DadosChat):
        pergunta = dados_chat['pergunta']
        contexto = [
            ('human', "meu nome é Herman. Lembre-se disso"),
            ('assistant', "Claro! Vou me lembrar disso")
        ]
        #dados_chat['contexto']
        
        resultado = self.rag_chain.invoke({"pergunta": pergunta, 'contexto': contexto})
        
        resposta_llm = resultado["llm_response"]
        resposta_llm.response_metadata['message'] = str(resposta_llm.response_metadata['message'])
        contexto.append(tuple(('human', pergunta)))
        contexto.append(tuple(('assistant', resposta_llm.content)))
        
        return {'pergunta': pergunta,
                'documentos': resultado["documentos_recuperados"],
                "resposta_llama": resposta_llm.response_metadata,
                'resposta': resposta_llm.content,
                'contexto': contexto,
                'tempo_consulta': resultado['tempo_consulta'],
                'tempo_bert': None,
                'tempo_inicio_resposta': None,
                'tempo_llama_total': resposta_llm.response_metadata['total_duration'] / 1000000000.}
        '''AIMessage(
        content="A Lei Maria da Penha garante à mulher em situação de violência doméstica e familiar o acesso a serviços de Defensoria Pública ou de Assistência Judiciária Gratuita, nos termos da lei. Além disso, ela tem direito ao atendimento policial e pericial especializado, ininterrupto e prestado por servidores previamente capacitados.\n\nA autoridade policial deve garantir proteção policial quando necessário, comunicando de imediato ao Ministério Público e ao Poder Judiciário. Ela também deve encaminhar a ofendida ao hospital ou posto de saúde e ao Instituto Médico Legal, fornecer transporte para a ofendida e seus dependentes para abrigo ou local seguro, quando houver risco de vida.\n\nA mulher em situação de violência doméstica e familiar tem direito à informação sobre os direitos conferidos pela Lei Maria da Penha e aos serviços disponíveis, inclusive os de assistência judiciária. Além disso, ela tem o direito ao atendimento especializado e humanizado.\n\nÉ importante ressaltar que a mulher em situação de violência doméstica e familiar deve estar acompanhada de advogado em todos os atos processuais, exceto se previsto no artigo 19 da Lei Maria da Penha.",
        additional_kwargs={},
        response_metadata={
            "model": "llama3.1",
            "created_at": "2024-12-30T12:12:07.231005007Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": 42873700782,
            "load_duration": 31836667500,
            "prompt_eval_count": 864,
            "prompt_eval_duration": 1571000000,
            "eval_count": 288,
            "eval_duration": 8633000000,
            "message": Message(
                role="assistant", content="", images=None, tool_calls=None
            )'''

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
    print(json.dumps(ragchain.consultar({'pergunta': pergunta, 'contexto': []}), indent=4, ensure_ascii=False))