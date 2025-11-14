"""Simple example of using MCP file tools with LangChain agent"""

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: The path to the file to read (relative to workspace root)
    
    Returns:
        The contents of the file as a string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        return path.read_text(encoding='utf-8')
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """List files and directories in a given directory.
    
    Args:
        directory_path: The path to the directory to list (defaults to current directory)
    
    Returns:
        A formatted string listing files and directories
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory '{directory_path}' does not exist"
        if not path.is_dir():
            return f"Error: '{directory_path}' is not a directory"
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"[DIR]  {item.name}/")
            else:
                items.append(f"[FILE] {item.name}")
        
        if not items:
            return f"Directory '{directory_path}' is empty"
        
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory '{directory_path}': {str(e)}"


# Initialize the model
model = ChatOpenAI(
    model="gpt-oss",
    temperature=0.0,
    base_url="http://10.199.178.176:8080/v1",  # Removed trailing slash
    api_key="111",
    streaming=False,  # Disable streaming to avoid template issues
    timeout=30.0
)

# Get the prompt from LangChain hub
try:
    prompt = hub.pull("hwchase17/openai-functions-agent")
except Exception as e:
    print(f"Warning: Could not pull prompt from hub: {e}")
    print("Using default prompt template...")
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided tools to answer questions."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

# Define tools
tools = [read_file, list_directory]

# Bind tools to the model
tools_bound_model = model.bind_tools(tools)

# Create the agent
agent = create_tool_calling_agent(tools_bound_model, tools, prompt)

# Create the executor
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Example usage
if __name__ == "__main__":
    try:
        result = executor.invoke({
            "input": "List all files in the current directory"
        })
        print("\n" + "="*50)
        print("RESULT:")
        print("="*50)
        print(result)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
