from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq

import os
import dotenv

dotenv.load_dotenv()

# Initialize the language model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_retries=2
)

# Set up embeddings
embeddings_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_API_KEY")
)

# Load documents
loader = DirectoryLoader("planets", glob="*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split documents into chunks
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=50, chunk_overlap=0
)

split_documents = []
for doc in documents:
    chunks = splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            split_documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": i
                }
            ))

# Store in Chroma
db = Chroma.from_documents(split_documents, embeddings_model)

# -----------------------
# TOOLS
# -----------------------
@tool("PlanetDistanceSun")
def planet_distance_sun(planet_name: str) -> str:
    """Returns the approximate distance of a planet from the Sun in AU."""
    distances = {
        "Earth": "Earth is approximately 1 AU from the Sun.",
        "Mars": "Mars is approximately 1.5 AU from the Sun.",
        "Jupiter": "Jupiter is approximately 5.2 AU from the Sun.",
        "Pluto": "Pluto is approximately 39.5 AU from the Sun."
    }
    return distances.get(
        planet_name,
        f"Information about the distance of {planet_name} from the Sun is not available in this tool."
    )


@tool("PlanetRevolutionPeriod")
def planet_revolution_period(planet_name: str) -> str:
    """Returns the revolution period of a planet around the Sun in Earth years."""
    periods = {
        "Earth": "Earth takes approximately 1 Earth year to revolve around the Sun.",
        "Mars": "Mars takes approximately 1.88 Earth years to revolve around the Sun.",
        "Jupiter": "Jupiter takes approximately 11.86 Earth years to revolve around the Sun.",
        "Pluto": "Pluto takes approximately 248 Earth years to revolve around the Sun."
    }
    return periods.get(
        planet_name,
        f"Information about the revolution period of {planet_name} is not available in this tool."
    )


@tool("PlanetGeneralInfo")
def planet_general_info(planet_name: str) -> str:
    """Returns general information about a planet using vector similarity search."""
    results = db.similarity_search(planet_name)
    if results:
        return results[0].page_content
    else:
        return f"Additional information for {planet_name} is not available in this tool."

# Tool setup
tools_list = [planet_distance_sun, planet_revolution_period, planet_general_info]
tool_map = {tool.name: tool for tool in tools_list}

# -----------------------
# Chain 1: PromptTemplate
# -----------------------
first = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant who answers questions users may have. You are asked: {question}."
)

# -----------------------
# Chain 2: LLM with tools
# -----------------------
middle = llm.bind_tools(tools_list)

# -----------------------
# Chain 3: Custom tool executor
# -----------------------
def execute_tools(result):
    """
    Executes tools based on the model's output.
    This function should only print the final result, not intermediate steps.
    """
    if result.tool_calls:
        # Loop through each tool call the model wants to make
        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_fn = tool_map.get(tool_name)

            # Check if the tool exists and execute it
            if tool_fn:
                output = tool_fn.invoke(tool_args)
                # Only print the final output of the tool
                print(output)
    else:
        # If no tool is called, print the model's direct response
        print(result.content)

last = RunnableLambda(execute_tools)

# -----------------------
# Compose the full pipeline
# -----------------------
chain = RunnableSequence(first | middle | last)

# -----------------------
# Run the pipeline
# -----------------------
user_query = input()
# First, invoke the chain to produce the main answer
chain.invoke({"question": user_query})

# -----------------------
# ✅ Required print statement - MOVED TO THE END
# -----------------------
# After producing the main output, print the chain information
print(chain)
