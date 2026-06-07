"""

Denoser intake is responsible clasifying the input according to a prompt
and if matches the criteria of a `signal` - preserve it
"""

from typing import TypedDict, Annotated

# decorator to mark a func as a tool
from langchain.tools import tool
# Ollama wrapper
from langchain_ollama import ChatOllama 
# wrappers around messages

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    email: dict | None
    messages: Annotated[list, add_messages]




@tool
def preserve_signal(summary: str) -> str:
    """Preserve an email classified as signal. Pass a short summary including the from and topic."""
    return f"Signal preserved: {summary}"

@tool
def log_reason(summary: str) -> str:
    """Log why an email was skipped as noise. Pass a one-sentence reason, up to 10 words."""
    return f"Signal skipped: {summary}"


TOOLS = [preserve_signal, log_reason]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# an agent brain, responsible for actions like:
# - invoke models
# - call tools
# - invoke llm again

def call_model(state) -> dict:
    """
    The actual flow differs from one in the book, where clearly 2 pieces are provided: order_id and human message.
    to keep the semantic equivalent (2 pieces), here is translation map:

    order_id -> email_message
    human_message: 'please cancel my order' -> 'please categorise this email'

    that ist tradeoff to keep the semantic close with the one from a guide and make the following easier

    """
    # extract messages from a state
    messages = state['messages']

    # extract other useful context, in this case email message
    email_message = state.get('email', {'email_message': 'EMPTY'})

    # system prompt that tells the model exactly what to do.
    prompt = (
    f'''You are an email filtering agent. Decide whether an email is SIGNAL or NOISE.
email_message: {email_message}

CORE TEST — apply this first:
  Does the email require the recipient to take a NEW action (pay a bill, respond, decide)?
  - Pending bill / obligation the recipient must still settle      → SIGNAL
  - Record or confirmation of something already done               → NOISE

SIGNAL (preserve):
- Utility/commodity bills still owed: electricity, water, heating, gas,
  rent, housing-office fees, municipal taxes, internet/phone service
- Technical info: mailing-list digests, dev/infra updates, release notes,
  security advisories
- Anything else demanding the recipient act, respond, or decide

NOISE (skip):
- Purchase activity of any kind — order confirmations, receipts, shipping/
  delivery updates, "payment received" — the action is already complete,
  even though money is mentioned
- Generic notifications: social, app, promo, "someone did X", newsletters

KEY DISTINCTION: a utility BILL is money you still owe (signal); a purchase
RECEIPT is money already spent (noise). If the only payment mentioned is one
already made, it is NOISE.

EXAMPLES:
- "Housing office: heating bill 1,240 UAH due Jun 20"  → SIGNAL (unpaid obligation)
- "Your Rozetka order #5512 has shipped"               → NOISE (purchase already placed)
- "Payment received — thank you for your order"        → NOISE (action complete)
- "Water utility: submit meter reading by Jun 15"      → SIGNAL (action required)

If SIGNAL — summarise it, include the `from` and `topic`, call tool: preserve_signal.
If NOISE — give the skip reason in one sentence, up to 10 words, call tool: log_reason.
'''
)

    full = [SystemMessage(prompt)] + messages

    # 1st LLM pass: 
    first: AIMessage = ai_client.invoke(full)
    out = [first]

    if getattr(first, 'tool_calls', None):

        # iterate over tools calls
        for tc in first.tool_calls:
            tool = TOOLS_BY_NAME.get(tc['name'])
            if tool is None:
                print(f'Unknown tool: {tc["name"]}')
                continue
            result = tool.invoke(tc['args'])
            out.append(ToolMessage(content=result,  tool_call_id=tc['id']))
            print(f'called tools: {tc}')
    
    return {'messages': out}



def cosntruct_graph():

    g = StateGraph(State)
    g.add_node('assistant', call_model)
    g.set_entry_point('assistant')
    return g.compile()



ai_client = ChatOllama(
    model='qwen3:4b', 
    temperature=0,
    reasoning=False,
    num_ctx=4096,
    keep_alive=-1,
).bind_tools(TOOLS)
graph = cosntruct_graph()




messages = [
    '''
Welcome to the reading list, a weekly roundup of news and links related to buildings, infrastructure and industrial technology. This week we look at chatbots replacing realtors, Chinese synthetic diamonds, Australian batteries, Meta’s data center tents, and more. Roughly 2/3rds of the reading list is paywalled, so for full access become a paid subscriber.

Iran war
Iran breaks off negotiations with the US and vows to “completely block” the Strait of Hormuz. [CNBC]

Analysis of satellite data by the BBC suggests the damage Iran has inflicted on US military facilities is more extensive than has been previously reported. [BBC]

Housing
A NYT reporter successfully uses an AI chatbot instead of a realtor to sell their house. “A flurry of bookings to view our house over the coming weekend arrived in my inbox within hours. I struggled to manage the appointments until, again, I just let the chatbot do everything for me. I told agents that they had to email or text — no phone calls. Whatever they wrote, I copied and pasted into the chatbot; whatever it replied, I copied and pasted right back to the agents. I was worried that pushy ones would prey on my inexperience, so I had the chatbot come up with a list of potential conflicts and write confident responses I could have ready.” [NYT]

Opposition to property taxes is having a political moment, which as we’ve mentioned previously is a pretty bad idea. Now Florida really seems like it might be on the verge of effectively eliminating homeowner property taxes. “Gov. Ron DeSantis’ property tax plan for the November ballot would raise the homestead exemption to $250,000 and require the Legislature to enact a plan to eliminate property taxes entirely for the vast majority of Floridians who own the homes they live in, he announced Wednesday DeSantis said he was calling the Legislature back to Tallahassee on Monday to add an amendment to the ballot that would eventually eliminate property taxes for 92% of those Floridians by raising the homestead exemption to $500,000.” [Governing]

The urban benefits of allowing tall buildings. “Land-use regulations, including height limits, affect housing affordability and urban productivity. This column analyses over 11,000 urban agglomerations and 300,000 tall buildings to explore the effect of height restrictions on welfare. Vertical growth enhances land efficiency, reduces commuting, and boosts worker welfare. While higher density can increase housing demand and rents, the associated gains more than offset the costs.” [VoxEU]

It’s apparently easier to get planning permission to build a skyscraper in London (a city which has notoriously made it almost impossible to build new housing) if you include a publicly accessible roof deck, and thus quite a few London skyscrapers have them. [Diamond Geezer]
A census map of where air conditioning is uncommon in the US. [X]
''',
'Дякуємо за замовлення 3434242'

]

if __name__ == '__main__':
    for idx, m in enumerate(messages):
        print(f"{'-'*20} Start {idx} {'-'*20}")
        email_message = {'email_message': m}
        convo = [HumanMessage(content='Please categorise  the email content')]
        result = graph.invoke(State({'email': email_message, 'messages': convo}))
        for msg in result['messages']:
            print(f'{msg.type}: {msg.content}')
        
        print(f"{'-'*20} End {idx} {'-'*20}")