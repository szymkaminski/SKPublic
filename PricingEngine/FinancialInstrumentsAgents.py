import os
import openai

class Agent:
    def __init__(self, name, model, api_key=None):
        self.name = name
        self.model = model
        self.api_key = ""

    def call_openai_api(self, prompt, temperature=0.7, max_tokens=150):
        openai.api_key = self.api_key

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": f"You are {self.name}."}, {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        message = response.choices[0].message.content
        return message
    
class Supervisor:
    def __init__(self, agents):
        self.agents = agents

    def manage_conversation(self, prompt):
        conversation_history = []
        current_prompt = prompt
    # w tej części byłoby kluczowe napisanie kodu tak, aby w właściwy sposób były requesty kierowane do właściwych agentów
    # to może być trudne - może w system message agenta powinno to się zdefiniować?
    # if prośba o dane then {Agent1}, if prośba o obliczenia then {Agent2}? 
        for i, agent in enumerate(self.agents):
            print(f"\nSupervisor: Sending prompt to {agent.name}")
            response = agent.call_openai_api(current_prompt)
            conversation_history.append({"agent": agent.name, "response": response})
            current_prompt = f"{response}\n\nWhat do you think, {self.agents[(i+1) % len(self.agents)].name}?"

        return conversation_history
    
# Initialize two agents
agent_API_key=""
agent1 = Agent(name="Quant", model="gpt-4", api_key=agent_API_key)
agent2 = Agent(name="Analyst", model="gpt-4", api_key=agent_API_key)

# Create a supervisor to manage conversation
supervisor = Supervisor(agents=[agent1, agent2])

# Define a user prompt to start the conversation
initial_prompt = "You are discussing the valuation of financial instruments. Quant, how would you approach this?"

# Supervisor handles the conversation
conversation = supervisor.manage_conversation(initial_prompt)

# Print the conversation
for exchange in conversation:
    print(f"\n{exchange['agent']} says: {exchange['response']}")
            
            
