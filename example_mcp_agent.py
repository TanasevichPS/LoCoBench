"""Simple example of using MCP file tools with LangChain agent"""

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
from locobench.tools.file_tools import file_tools

# Initialize the model
model = ChatOpenAI(
    model="gpt-oss",
    temperature=0.0,
    base_url="http://localhost:8080/v1",
    api_key="111",
    streaming=True,
    timeout=30.0
)

# Get the prompt from LangChain hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Bind tools to the model
tools_bound_model = model.bind_tools(file_tools)

# Create the agent
agent = create_tool_calling_agent(tools_bound_model, file_tools, prompt)

# Create the executor
executor = AgentExecutor(agent=agent, tools=file_tools, verbose=True)

# Example usage
if __name__ == "__main__":
    result = executor.invoke({
        "input": "List all files in the current directory"
    })
    print(result)
