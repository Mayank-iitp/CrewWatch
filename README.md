## How to use
```
from CrewWatch import CrewWatchExtra

# Initialize Agent Watch
rag_watch = AgentWatchExtended(model="gpt-4o")

# Start monitoring
rag_watch.start()

# Your Crew AI operations go here
# For example:
# output = your_crew_ai_function()

# Set token counts (replace with actual values)
input_text = "Your input text here."
output_text = "Your output text here."
rag_watch.set_token_counts(input_text=input_text, output_text=output_text)

# End monitoring
rag_watch.end()

# Visualize the results
rag_watch.visualize(method='cli')  

```
## A CrewAI Example 

```

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import os
from CrewWatch import CrewWatchExtra
from langchain_groq import ChatGroq
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
llm = ChatGroq(
    model_name="groq/llama3-8b-8192"
)

watch= CrewWatchExtra(model="gpt-4o")
watch.start()
Problem_Definition_Agent = Agent(
        role='Problem_Definition_Agent',
        goal="""clarify the machine learning problem the user wants to solve,
            identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
        backstory="""You are an expert in understanding and defining machine learning problems.
            Your goal is to extract a clear, concise problem statement from the user's input,
            ensuring the project starts with a solid foundation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
Model_Recommendation_Agent = Agent(
        role='Model_Recommendation_Agent',
        goal="""suggest the most suitable machine learning models based on the problem definition
            and  providing reasons for each recommendation.""",
        backstory="""As an expert in machine learning algorithms, you recommend models that best fit
            the user's problem and data. You provide insights into why certain models may be more effective than others,
            considering classification vs regression and supervised vs unsupervised frameworks.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

Starter_Code_Generator_Agent = Agent(
        role='Starter_Code_Generator_Agent',
        goal="""generate starter Python code for the project, including data loading,
            model definition, and a basic training loop, based on findings from the problem definitions,
            data assessment and model recommendation""",
        backstory="""You are a code wizard, able to generate starter code templates that users
            can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
user_question= input("Enter your question")
task_define_problem = Task(
        description="""Clarify and define the machine learning problem,
            including identifying the problem type and specific requirements.

            Here is the user's problem:

            {ml_problem}
            """.format(ml_problem=user_question),
        agent=Problem_Definition_Agent,
        expected_output="A clear and concise definition of the machine learning problem."
        )

task_recommend_model = Task(
        description="""Suggest suitable machine learning models for the defined problem
            and assessed data, providing rationale for each suggestion.""",
        agent=Model_Recommendation_Agent,
        expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
        )
task_generate_code = Task(
        description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s),
            including snippets for package import, data handling, model definition, and training
            """,
        agent=Starter_Code_Generator_Agent,
        expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        )
crew = Crew(
            agents=[Problem_Definition_Agent, Model_Recommendation_Agent,  Starter_Code_Generator_Agent], #, Summarization_Agent],
            tasks=[task_define_problem, task_recommend_model,  task_generate_code], #, task_summarize],
            verbose=True
        )
result = crew.kickoff()
print (result)
watch.set_token_usage_from_crew_output(result)
watch.end()

watch.visualize(method='cli')
```


