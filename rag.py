import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama-3.1-70b-versatile")


def answer_question(question, retriever):

    system_prompt = (
        "Você é um FAQ INTELIGENTE"
        "Pesosas que estão fazendo um curso de CPSI(Contrato Público de Solução Inovadora). Irão tirar dúvidas com você"
        "Use o contexto para responder a essas questões"
        "\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": question})
    return results

