from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def planner_node(state: ResearchState):
    structured_llm = llm.with_structured_output(SearchPlan)
    prompt = f"Break this topic down into 3 academic search queries: {state['topic']}"
    
    plan = structured_llm.invoke(prompt)
    return {"search_queries": plan.queries, "critique_count": 0}

from tavily import TavilyClient
tavily = TavilyClient(api_key="tvly-...")

def scraper_node(state: ResearchState):
    new_results = []
    for q in state["search_queries"]:
        response = tavily.search(query=q, search_depth="advanced", max_results=2)
        for obj in response["results"]:
            new_results.append(f"Source ({obj['url']}):\n{obj['content']}\n---")
            
    return {"scraped_content": new_results}

def critic_node(state: ResearchState):
    structured_critic = llm.with_structured_output(Critique)
    
    context = "\n".join(state["scraped_content"])
    prompt = f"""
    Original Topic: {state['topic']}
    Collected Research: {context}
    
    Evaluate if this research is comprehensive enough to write an exhaustive technical brief. 
    If not, provide 2 new targeted search queries to fill the gaps.
    """
    
    evaluation = structured_critic.invoke(prompt)
    
    current_count = state.get("critique_count", 0) + 1
    
    return {
        "is_satisfactory": evaluation.is_sufficient,
        "search_queries": evaluation.missing_angles, # Overwrites old queries with the gap-fillers!
        "critique_count": current_count
    }
  def synthesis_node(state: ResearchState):
    context = "\n".join(state["scraped_content"])
    prompt = f"Write a comprehensive, professional technical brief on {state['topic']} using ONLY this data:\n{context}"
    
    # Switch to a heavier model for the actual prose generation
    writer_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    report = writer_llm.invoke(prompt)
    
    return {"final_brief": report.content}
