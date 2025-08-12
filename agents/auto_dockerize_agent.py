from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.2)

def generate_dockerfile(requirements: str) -> str:
    prompt = f"""
    Generate a Dockerfile for a Python project with the following requirements:

    {requirements}

    The Dockerfile should use python:3.10-slim, install dependencies using pip, and copy the project files appropriately.
    """
    response = llm.predict(prompt)
    return response

def generate_docker_compose(requirements: str) -> str:
    prompt = f"""
    Generate a docker-compose.yml for a Python project with these requirements:

    {requirements}

    Assume the service is named 'app' and expose port 8000.
    """
    response = llm.predict(prompt)
    return response


tools = [
    Tool(
        name="GenerateDockerfile",
        func=generate_dockerfile,
        description="Generates a Dockerfile given requirements.txt content."
    ),
    Tool(
        name="GenerateDockerCompose",
        func=generate_docker_compose,
        description="Generates a docker-compose.yml given requirements.txt content."
    )
]


prompt_template = """
You are an expert DevOps assistant. You can use the following tools:

{tools}

Please follow this format:

Question: {input}
Thought: you should always think about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the output of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

{agent_scratchpad}
"""


prompt = PromptTemplate.from_template(prompt_template)
print("Tools:", [tool.name for tool in tools])

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def save_file(filename: str, content: str):
    with open(filename, "w") as f:
        f.write(content)
    print(f"Saved {filename}")

with open("requirements.txt", "r") as f:
    requirements_content = f.read()
    result = agent_executor({"input": requirements_content})
    print(result)