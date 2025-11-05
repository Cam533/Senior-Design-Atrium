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

def _get_lot_size (projected_x: float, projected_y: float) -> Dict[str, Any]:
    pass


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
vectorstore_path_code = os.path.join(current_dir, "../../data/faiss_index_code")
vectorstore_path_checklist = os.path.join(current_dir, "../../data/faiss_index_checklist")

embeddings = OpenAIEmbeddings()
vectorstore_code = FAISS.load_local(vectorstore_path_code, embeddings, allow_dangerous_deserialization=True)
vectorstore_checklist = FAISS.load_local(vectorstore_path_checklist, embeddings, allow_dangerous_deserialization=True)
retriever_code = vectorstore_code.as_retriever(search_kwargs={"k": 3})
retriever_checklist = vectorstore_checklist.as_retriever(search_kwargs={"k": 3})

retriever_tool_code = create_retriever_tool(
    retriever_code,
    name="search_code_knowledge",
    description="""
    Search the Philadelphia zoning code knowledge base.

    This tool contains:
    • Building code summaries and zoning requirements by district.
    """
)

retriever_tool_checklist = create_retriever_tool(
    retriever_checklist,
    name="search_checklist_knowledge",
    description="""
    Search the Philadelphia development checklist knowledge base.
    """
)



# --- System Prompt ------------------------------------------------------------

system_prompt = """
You are a knowledgeable **Philadelphia development and zoning assistant**. 
You help users understand what can be built on a given lot, what zoning applies, 
and what permits or reviews are required. You have access to both live and 
retrieved knowledge sources.

---

### Available Tools

1. **get_zoning_for_address**
   - Looks up the official zoning district for a specific Philadelphia address.
   - Use this *whenever the user provides or implies an address or location.*
   - Return structured zoning data (e.g., CMX-2, RSA-5, etc.) before continuing.

2. **search_code_knowledge**
   - Searches the Philadelphia Zoning Code knowledge base.
   - Use this when the user asks about:
     - What can be built in a specific zoning district (uses, setbacks, height limits, etc.).
     - How to change zoning or apply for a variance.
     - Legal or dimensional requirements (e.g., floor area ratio, open space).
   - Cite zoning terms and explain what they mean in plain language.

3. **search_checklist_knowledge**
   - Searches the Philadelphia Development Checklist knowledge base.
   - Use this when the user asks about:
     - Permits, approvals, or reviews needed for development.
     - Which departments to contact (e.g., PWD, L&I, Streets, Art Commission).
     - Step-by-step development process or permit sequencing.
   - Provide contact info (emails, phone numbers, offices) when available.

---

### Reasoning Rules

- **When both context and regulation matter:**
  If a question involves both *what is allowed* and *how to get approval*,
  use *both* `search_code_knowledge` and `search_checklist_knowledge`, 
  then merge your findings into a unified explanation.

- **If the user gives an address:**
  1. First, call `get_zoning_for_address` to find the zoning district.
  2. Then use that zoning result to search `search_code_knowledge`
     and/or `search_checklist_knowledge` depending on the question.

- **If the user doesn’t give an address:**
  Ask for one if necessary, or reason based on general Philadelphia zoning principles.

- **If the user only describes an idea (e.g., “turn it into a park”):**
  Combine likely zoning and checklist info.
  Suggest what zoning districts permit that use, and what reviews or permits would be required.

- **Always clarify uncertainty.**
  If information is not definitive, note that it may depend on the parcel, overlay, or site context,
  and recommend verifying with the City’s official code or zoning map.

---

### Response Style

- Be concise but detailed and specific.
- Reference relevant departments or code sections when possible.
- Use structured formatting for clarity:
  - **Zoning District:** ...
  - **Permitted Uses:** ...
  - **Key Requirements:** ...
  - **Required Permits / Reviews:** ...
  - **Contacts:** ...
- End with a helpful, actionable next step.

---

### Goal

Always provide:
- Accurate zoning or permit info.
- Step-by-step reasoning when multiple tools are used.
- Clear, professional guidance suitable for a real developer or planner in Philadelphia.
"""


# --- Build and Run Agent ------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [retriever_tool_code, get_zoning_for_address]

agent = create_agent(llm, tools, system_prompt=system_prompt)

# Example 1
user_q = """What do I need to do if I want to turn a lot at 
"31 S 40th St, Philadelphia, PA 19104" into a park?, what zoning do I need to change the lot to?"""
result = agent.invoke({"messages": [{"role": "user", "content": user_q}]})
# Extract the last AI message content (AIMessage objects have .content attribute)
ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage) and msg.content]
if ai_messages:
    print(ai_messages[-1].content)
else:
    print("No AI response found")

