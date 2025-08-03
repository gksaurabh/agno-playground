from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType

knowledge = UrlKnowledge(
    urls=[
        "https://www2.hse.ie/conditions/colic/",
        "https://www2.hse.ie/conditions/babies-who-are-small-growing-slowly/",
        "https://www2.hse.ie/conditions/diarrhoea-babies-children/",
        "https://www2.hse.ie/conditions/eczema-babies-children/"
    ],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name='hse_guidelines',
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small", dimensions=1536),
    ),
)

storage = SqliteStorage("agent_sessions", db_file="tmp/agents.db")

agent = Agent(
    name="HSE Guidelines Agent",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    agent.knowledge.load(recreate=False)
    agent.print_response(
        "How do i know if my child has colic?",
        stream=True,
        max_tokens=1000,
    )
