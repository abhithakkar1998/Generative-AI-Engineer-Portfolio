from crewai import Crew, Process
from agents import blog_researcher_agent, blog_writer_agent
from tasks import research_task, writing_task

crew = Crew(
    agents=[blog_researcher_agent, blog_writer_agent],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result = crew.kickoff(inputs={'topic': 'Generative AI'})
print(result)