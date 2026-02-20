from crewai_tools import YoutubeChannelSearchTool
import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env')
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

api_version = "2024-12-01-preview" 

from crewai_tools import YoutubeChannelSearchTool
import os
from dotenv import load_dotenv
from chromadb.config import Settings

load_dotenv('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env')
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig',
    config=dict(
        llm=dict(
            provider="azure",
            config=dict(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2025-04-01-preview",
            ),
        ),
        embedder=dict(
            provider="azure",
            config=dict(
                deployment_id=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
                model="text-embedding-ada-002",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-12-01-preview",
                api_type="azure",
            ),
        ),
        vectordb=dict(            
            provider="chromadb",
            config={},
        ),
    )
)