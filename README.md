# Chatbot com Memória e Gerenciamento de Contexto

## Introdução
Este projeto implementa um chatbot utilizando a API da **GROQ** e a biblioteca **LangChain** para permitir interações inteligentes com memória de histórico. O sistema armazena o contexto da conversação, possibilitando um diálogo mais fluido e eficiente.

A implementação inclui o uso de **memória de histórico de mensagens**, **limitação de tokens** para gerenciamento de recursos e **integração de um modelo de IA** para responder dinamicamente às mensagens do usuário.

---

## Sumário
1. [Glossário](#glossario)
2. [Instalação](#instalacao)
   - Clonando o Repositório
   - Criando e Ativando o Ambiente Virtual
   - Instalando Dependências
3. [Explicação dos Blocos de Código](#explicacao-dos-blocos-de-codigo)
   - Importação de Bibliotecas
   - Carregamento de Variáveis de Ambiente
   - Configuração do Modelo de IA
   - Gerenciamento de Histórico
   - Criação de Prompt Template
   - Limitação de Memória
   - Pipeline de Processamento
4. [Execução](#execucao)
5. [Arquivos do Projeto](#arquivos-do-projeto)

---

## Glossário <a id="glossario"></a>

- **Memória do Chatbot**: Capacidade do sistema de armazenar e recuperar mensagens anteriores para manter o contexto da conversa.
- **Token**: Unidade de texto utilizada no processamento de linguagem natural.
- **Pipeline de Execução**: Fluxo otimizado para passagem de informações entre componentes do chatbot.
- **LangChain**: Biblioteca que facilita o desenvolvimento de aplicações que utilizam modelos de linguagem.
- **GROQ API**: API que permite a integração de modelos de IA para processamento de linguagem natural.

---

## Instalação <a id="instalacao"></a>

### 1. Clonando o Repositório
```bash
# Clone este repositório
git clone https://github.com/seu_usuario/seu_repositorio.git

# Acesse a pasta do projeto
cd seu_repositorio
```

### 2. Criando e Ativando o Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/Mac
source venv/bin/activate
```

### 3. Instalando Dependências
```bash
pip install -r requirements.txt
```

---

## Explicação dos Blocos de Código <a id="explicacao-dos-blocos-de-codigo"></a>

### 1. Importação de Bibliotecas
O projeto utiliza bibliotecas como `os`, `dotenv` e `LangChain` para gerenciamento de IA, histórico e processamento de mensagens.

```python
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
```

### 2. Carregamento de Variáveis de Ambiente
As credenciais da API da **GROQ** são protegidas usando um arquivo `.env`.

```python
load_dotenv(find_dotenv())
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### 3. Configuração do Modelo de IA
Inicialização do modelo `gemma2-9b-it`.

```python
model = ChatGroq(
    model="gemma2-9b-it",
    groq_api_key=GROQ_API_KEY
)
```

### 4. Gerenciamento de Histórico
Cria e recupera históricos de mensagens para diferentes sessões.

```python
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

### 5. Criação de Prompt Template
Define um template para as mensagens enviadas ao modelo de IA.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil. Responda com precisão."),
    MessagesPlaceholder(variable_name="messages")
])
```

### 6. Limitação de Memória
Evita que o histórico ultrapasse um limite de tokens.

```python
trimmer = trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
```

### 7. Pipeline de Processamento
Conecta componentes para otimizar o processamento das mensagens.

```python
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
```

---

## Execução <a id="execucao"></a>
Para rodar o chatbot, utilize:
```bash
python main.py
```

---

## Arquivos do Projeto <a id="arquivos-do-projeto"></a>

### `.gitignore`
Evita que arquivos sensíveis sejam versionados.
```plaintext
venv/
.env
```

### `requirements.txt`
Lista todas as dependências necessárias para o projeto.
```plaintext
langchain==0.3.21
langchain-groq==0.3.1
python-dotenv==1.0.1
...
```

---

## Contribuição
Se desejar contribuir, faça um fork do repositório, crie uma branch e envie um pull request com suas melhorias.



