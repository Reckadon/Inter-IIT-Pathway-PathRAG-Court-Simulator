from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="mixtral-8x7b-32768",
               temperature=0.5,
               max_retries=2)

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="You are a helpful assistant who decides if the user should work right now or can take a rest based on the input tasks they provide. Respond with 'work' or 'rest'"),
    ("human", "tasks: {tasks}")
])

work_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="plan out how to work on the tasks provided"),
    ("human", "tasks: {tasks}"),
])

rest_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(
        content="give some books to read related to the tasks provided while resting for a bit"),
    ("human", "tasks: {tasks}"),
])

# for custom chain units, use RunnableLambda
# word_counter = RunnableLambda(lambda x: len(x.split()))

work_branch = RunnableLambda(lambda x: work_prompt_template | llm | StrOutputParser())
rest_branch = RunnableLambda(lambda x: rest_prompt_template | llm | StrOutputParser())

chain = prompt_template | llm | StrOutputParser() | RunnableBranch((lambda x: "work" in x, work_branch),
                                                                   (lambda x: "rest" in x, rest_branch), RunnableLambda(lambda x: "cant decide"))

tasks = input("Enter tasks: ")
print(chain.invoke({"tasks": tasks}))
# res = llm.invoke(messages)
# print(res.content)
