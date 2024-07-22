from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

# Initialize the Ollama model
model = Ollama(model="llama3")

email = """Dear User,

We have detected suspicious activity on your account. Please verify your information immediately to avoid account suspension.

Verify Now

Thank you,
BankAlert Security Team"""
is_verbose = False

classifier = Agent(
    role="email classifier",
    goal="accurately classify emails based on their importance. give every email one of these ratings: important, casual, or spam",
    backstory="You are an AI assistant whose only job is to classify emails accurately and honestly. Do not be afraid to give emails a bad rating if they are not important. Your job is to help the user manage their inbox.",
    verbose=True,
    allow_delegation=False,
    llm=model
)

responder = Agent(
    role="email responder",
    goal="Based on the importance of the email, write a concise and simple response. If the email is rated 'important' write a formal response, if the email is rated 'casual' write a casual response, and if the email is rated 'spam' ignore the email. No matter what, be very concise.",
    backstory="You are an AI assistant whose only job is to write short responses to emails based on their importance. The importance will be provided to you by the 'Classifier agent'.",
    verbose=True,
    allow_delegation=False,
    llm=model
)

classifier_email = Task(
    description=f"Classify the email '{email}'",
    agent=classifier,
    expected_output="One of these three options: 'important', 'casual', or 'spam'",
)

respond_to_email = Task(
    description=f"Respond to the email: '{email}'",
    agent=responder,
    expected_output="A short response to the email"
)

crew = Crew(
    agents=[classifier, responder],
    tasks=[classifier_email, respond_to_email],
    verbose=2,
    process=Process.sequential
)

output = crew.kickoff()
print(output)


