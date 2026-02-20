from crewai import Task
from tools import yt_tool
from agents import blog_researcher_agent, blog_writer_agent

#Research Task
research_task = Task(
    description=(
        'Identify the given topic {topic}'
        'Research and gather information on a given topic to create engaging and informative blog posts.'
    ),
    agent=blog_researcher_agent,
    tools=[yt_tool]
)

#Writing Task
writing_task = Task(
    description=(
        'Narrate the content in an engaging and informative manner to create a blog post on the topic {topic} from YT Channel'
    ),
    expected_output=(
        'A well-structured and compelling blog post that captures the essence of the topic and resonates with the target audience.'
    ),
    agent=blog_writer_agent,
    tools=[yt_tool],
    async_execution=False,
    output_file='blog_post.md'
)