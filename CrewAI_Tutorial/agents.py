import os
from crewai import Agent, LLM
from dotenv import load_dotenv
from tools import yt_tool

load_dotenv('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env')

llm = LLM(
    model="azure/"+os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-04-01-preview",
    max_tokens=2048,
)

blog_researcher_agent = Agent(
    role='Blog Researcher for Youtube Channel',
    goal='get the relevant video content for the topis{topic} from YT Channel',
    backstory=('You are a blog researcher for a youtube channel. Your job is to research and gather information on a given topic to create engaging and informative blog posts. You will use various online resources, including search engines, databases, and social media platforms, to find relevant information.'),
    verbose=True,
    llm=llm,
    tools=[yt_tool],
    memory=True,
)

blog_writer_agent = Agent(
    role='Blog Writer',
    goal='Narrate the content in an engaging and informative manner to create a blog post on the topic{topic} from YT Channel',
    backstory=('Your job is to write engaging and informative blog posts based on the research conducted by the Blog Researcher. You will take the information gathered and craft it into a well-structured and compelling blog post that captures the essence of the topic and resonates with the target audience.'),
    verbose=True,
    llm=llm,
    tools=[yt_tool],
    allow_delegation=True,
    memory=True,
)
    