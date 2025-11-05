import os, json, requests
from typing import Optional, Dict, Any

# LangChain + OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool

# LangChain agent creation
from langchain.agents import create_agent

# Retriever + geospatial
from langchain_classic.tools.retriever import create_retriever_tool
from geopy.geocoders import Nominatim
from pyproj import Transformer
from dotenv import load_dotenv

load_dotenv()  # reads .env with OPENAI_API_KEY=...

# Validate OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your environment or create a .env file.\n"
        "Example: OPENAI_API_KEY=sk-..."
    )

ARCGIS_URL = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/arcgis/rest/services/Zoning_BaseDistricts/FeatureServer/0/query"

def _query_zoning(projected_x: float, projected_y: float) -> Dict[str, Any]:
    params = {
        "where": "1=1",
        "geometry": f"{projected_x},{projected_y}",
        "geometryType": "esriGeometryPoint",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "json"
    }
    r = requests.get(ARCGIS_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _geocode_to_web_mercator(address: str) -> Optional[Dict[str, float]]:
    geolocator = Nominatim(user_agent="philly-zoning-rag")
    location = geolocator.geocode(address)
    if not location:
        return None
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    proj_x, proj_y = transformer.transform(location.longitude, location.latitude)
    return {"x": proj_x, "y": proj_y}


@tool("get_zoning_for_address", return_direct=False)
def get_zoning_for_address(address: str) -> str:
    """Look up zoning district for a Philadelphia street address."""
    coords = _geocode_to_web_mercator(address)
    if not coords:
        return json.dumps({"ok": False, "error": "Address not found or outside Philadelphia."})
    data = _query_zoning(coords["x"], coords["y"])
    if not data.get("features"):
        return json.dumps({"ok": False, "error": "No zoning data found for this point."})
    attrs = data["features"][0]["attributes"]
    return json.dumps({
        "ok": True,
        "address": address,
        "projected_xy": coords,
        "zoninggroup": attrs.get("zoninggroup"),
        "zoning": attrs.get("zoning"),
        "objectid": attrs.get("objectid"),
        "raw": data
    })


# --- Build RAG retriever ------------------------------------------------------

# Fetch the vectorstore
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to data/faiss_index (go up 2 levels from models/rag/ to project root)
vectorstore_path = os.path.join(current_dir, "../../data/faiss_index")
vectorstore_path = os.path.normpath(vectorstore_path)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_lot_knowledge",
    description="Search notes about Philadelphia building codes relevant to the lot at the address."
)

# --- System Prompt ------------------------------------------------------------

system_prompt = """You are a helpful planning assistant for Philadelphia lots.
You can (1) retrieve background zoning info from a small knowledge base, and
(2) call a zoning API tool to look up the zoning district for a specific address.

Guidelines:
- If the user mentions an address, call get_zoning_for_address to fetch the zoning.
- Use search_zoning_knowledge to define or summarize what a zoning code permits.
- When both apply, do BOTH: call the API, then provide a concise explanation retrieved from the KB.
- Be clear about non-authoritative notes and suggest checking the official Philadelphia Code.
- Prefer structured JSON when the user asks for it.
"""

# --- Build and Run Agent ------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [retriever_tool, get_zoning_for_address]

agent = create_agent(llm, tools, system_prompt=system_prompt)

# Example 1
user_q = """What's kind of residential buildings can I develop on a lot at "4042 Chestnut Street, Philadelphia, PA 19104", 
and what other constraints do I need to know about the lot before buliding residential buildings?"""
result = agent.invoke({"messages": [{"role": "user", "content": user_q}]})
# Extract the last AI message content (AIMessage objects have .content attribute)
ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage) and msg.content]
if ai_messages:
    print(ai_messages[-1].content)
else:
    print("No AI response found")


