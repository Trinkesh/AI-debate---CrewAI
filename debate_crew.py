"""
AI debate - project 

three agent will debate a topic 
pro debater - argues for a topic 
con debater - debate againts the topic 
judge - reads both argument and return an answer."""

import os
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# --- workaround for crewAI issue #5886 (cache_breakpoint sent to Groq) ---
import litellm

_original_completion = litellm.completion

def _completion_without_cache_breakpoint(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)

litellm.completion = _completion_without_cache_breakpoint
# --------------------------------------------------------------------------

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")


## now we will do the llm configuration
llm = LLM(model="groq/llama-3.3-70b-versatile",api_key=groq_api, temperature=0.7)

## agnets 

#an agent = role + goal + backstory 
pro_debator  = Agent(role = "pro debator",
                     goal  = "Argue persuasively IN FAVOR of the given topic",
                     backstory  = (
        "You are a sharp, articulate debater known for building "
        "compelling, evidence-based arguments. You never concede ground "
        "easily."
    ),
    llm=llm,
    verbose=True,  # prints the agent's reasoning/output live to the console
)

con_debater = Agent(
    role="Con Debater",
    goal="Argue persuasively AGAINST the given topic",
    backstory=(
        "You are a skeptical, rigorous debater who excels at finding "
        "holes in arguments and presenting strong counter-cases."
    ),
    llm=llm,
    verbose=True,
)

judge = Agent(
    role="Impartial Judge",
    goal="Evaluate both sides objectively and declare a well-reasoned winner",
    backstory=(
        "You are a veteran debate judge, respected for fairness and for "
        "grading arguments purely on logic and evidence, not rhetoric."
    ),
    llm=llm,
    verbose=True,
)

TOPIC  = "Artificial Intelligence will create  more jobs than it destroys"

## task 
## A task = description + expected output + agent reponsible for it 
## 

pro_task  = Task(
    description=f"write a strong 3-point opening argument in favour of topic: {TOPIC}"
    , expected_output="3 concise, persuasive bullet points supporting the topic.",
    agent= pro_debator
)


con_task  = Task(
    description=f"write a strong 3-point opening argument in favour of topic: {TOPIC}"
    , expected_output="3 concise, persuasive bullet points supporting the topic.",
    agent= con_debater
)


judge_task  = Task(
    description=  ("Review the Pro and Con arguments above. Score each side on "
        "logic, evidence, and persuasiveness, then declare a winner with "
        "a 2-3 sentence justification.")

    , expected_output="A verdict naming the winner and explaning why.",
    agent= judge,
    context=[pro_task,con_task]
)

## crew 
## crew ties the agent and the task toghter and control the execution order

crew = Crew(
    agents=[pro_debator, con_debater, judge],
    tasks=[pro_task, con_task, judge_task],
    process=Process.sequential,
    verbose=True,
)
if __name__=="__main__":
    result  = crew.kickoff()
    print(result)



