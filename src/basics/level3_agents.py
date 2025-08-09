from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

memory = Memory(
    model=OpenAIChat(id="gpt-4o"),
    db=SqliteMemoryDb(table_name="user_memory", db_file="tmp/agent.db"),
    delete_memories=True,
    clear_memories=True
)

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        ReasoningTools(add_instructions=True),
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True),
    ],
    user_id="test_user",
    instructions=[
       "Use tables to display data.",
       "Include sources in your response.",
       "Only include the report in your response. No other text.",
    ],
    memory=memory,
    enable_agentic_memory=True,
    markdown=True,
)

if __name__ == "__main__":
    
    agent.print_response(
        "My favorite stocks are NVIDIA, SHOPIFY, APPLE, and BlackBerry.",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True
    )
    

    agent.print_response(
        "Can you compare my favorite stocks?",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True
    )


