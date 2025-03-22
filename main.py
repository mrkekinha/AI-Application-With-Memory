
# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory                   # Permite criar Históricos de mensagens
from langchain_core.chat_history import BaseChatMessageHistory                              # Classe base para histórico de mensagens
from langchain_core.runnables.history import RunnableWithMessageHistory                     # Permite gerenciar o histórico de mensagens
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder                  # Permite criar prompts / mensagens
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages   # Mensagens humanas, do sistema e do AI
from langchain_core.runnables import RunnablePassthrough                                    # Permite criar fluxos de execução e reutilizaveis
from operator import itemgetter                                                             # Facilita a extração de valores de dicionários


# Carregar as variáveis de ambiente do arquvo .env (para proteger as credenciais)
load_dotenv(find_dotenv())

# Obter a chave da API do GROQ armazenada no arquivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o modelo de AI utilizando a API da GROQ
model = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = GROQ_API_KEY
)
#exemplo 1 ----------------------------
# Dicionário para armazenar o histórico de mensagens
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recura ou cria um histórico de mansagens para uma determinada sesão.
    Isso permite manter o contexto contínuo para diferentes usuários e interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Criar um gerenciador de histórico que conecta o modelo ao armazenamento de mensagens
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Configuração da sessão (Identificador único para cada chat/usuário)
config = {"configurable":{"session_id":"chat1"}}

# Exemplo de interação inicial do usuário
response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Maria e sou Engenheira.")],
    config=config
)

# Exibir a resposta do modelo
print("Resposta do modelo:", response.content)

# criação de um prompt template para estruturar a entrada do modelo
prompt = ChatPromptTemplate.from_messages(
  [
      ("system", "você é um assistente útil. responda todas ass perguntas com precisão "),
      MessagesPlaceholder (variable_name="messages") # permitir adicionar mensagens de forma dinamica
  ]
)

#conectar o modelo ao template de prompt
chain = prompt | model #usando lcel para conectar o prompt ao modelo

#exemplo de interação com o modelo usando o template
response = chain.invoke (
    { "messages": [HumanMessage(content= "oi, o meu nome é maria!")]}
)

#gerenciamento da memoria do chatbot

trimmer = trim_messages(
    max_tokens = 45, #define um limite maximo de tokens para evitar ultrapassar o consumo de memoria
    strategy = "last", #define a estrategia de corte para remover mensagens antigas
    token_counter = model, #usa o modelo para contar os tokens
    include_system = True, #inclui mensagens do sistema no historico
    allow_partial = False, #evita que as mensagens sejam cortadas parcialmente
    start_on ="human" #começa a contagem com a mensagem humana
)

#exemplo de historico de mensagens
messages = [
    SystemMessage(content="você é um assistente. responda todas as perguntas com precisão."),
    HumanMessage(content=" oi, o meu nome é raquel."),
    AIMessage(content="oi, raquel! como posso te ajudar hoje?"),
    HumanMessage(content="eu gosto de sorvete de doce de leite.")

]

#aplicar o limitador de memoria ao historico
response = trimmer.invoke(messages)

#criando um pipeline de execução para otimizar a passafem de informações entre os componentes
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

#exemplo de interação utilizando o pipeline otimizado
response = chain.invoke(
    {
        "messages": [HumanMessage(content="qual é o sorvete que eu gosto?")],
        
    }
)

#exibir a resposta final do modelo
print ("resposta final do motelo", response.content)