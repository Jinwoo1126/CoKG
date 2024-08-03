from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def Base_Summary(llm, sample):
    base_prompt = '''
    Summarize the below Document which include multiple texts with similar topics.
    Document: {Document}
    '''
    prompt = PromptTemplate(template=base_prompt, input_variables=['Document'])
    chain = prompt | llm

    return print(chain.invoke({'Document': sample['document']}).content)

def CoD_Summary(llm, sample):
    cod_prompt = '''
    Article: {ARTICLE}
    You will generate increasingly concise, entity-dense summaries of the above article. 

    Repeat the following 2 steps 5 times. 

    Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary. 
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 

    A missing entity is:
    - relevant to the main story, 
    - specific yet concise (5 words or fewer), 
    - novel (not in the previous summary), 
    - faithful (present in the article), 
    - anywhere (can be located anywhere in the article).

    Guidelines:

    - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
    - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
    - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 

    Remember, use the exact same number of words for each summary.
    Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
    Answer in JSON without necessarily adding any introductory text or meta comments.
    '''
    prompt = PromptTemplate(template=cod_prompt, input_variables=['Document'])
    chain = prompt | llm | JsonOutputParser()

    return print(chain.invoke({'Document': sample['document']}).content)

def CoE_Summary(llm, sample):
    coe_prompt = """
    Task description: Now you are a multi-document summary generator. 
    Your <Input> is multiple texts with similar topics. 
    You will extract common key information from them and integrate them into a summary.

    ----

    [The process of summary reasoning]:
    (1) Specific event extraction: for each text in <Input>, extract the fine-grained events of the main event body.
    (2) Event abstraction and generalization: analyze each fine-grained event extracted in step (1), extract their common feature, abstract and generalize them into a more concise and refined result. For example, [children's program], [weather forecast], [singing program], [TV shopping] can be generalized as [TV program].
    (3) Common event statistics: for each abstract event generalized in step (2), find out which text it covers and select the event that covers most of the texts.
    (4) Summary generation: based on chronological order or level of importance, integrate the events selected in step (3) to form a concise summary.

    ----

    [The example of summary reasoning]:
    <Input>:
    text 1: LAHORE • A bomb targeting Pakistani police outside a major Sufi shrine in the city of Lahore yesterday killed at least 10 people and
    wounded more than 20. The blast, occurring a day after the start of the Muslim holy month of Ramadan, went off at a police checkpoint near the
    Data Darbar, one of the largest Muslim shrines in South Asia, which attracts tens of thousands of visitors a year.
    text 2: Pakistan bombing kills at least nine people, police the target outside Lahore shrine A powerful bomb has exploded near security forces
    guarding a famous Sufi shrine in the eastern city of Lahore, Pakistan, killing at least nine people and wounding several others.Militant violence
    has declined sharply in Pakistan following a sustained crackdown in recent years and over the past two years Lahore, Pakistan's second-largest city,
    has been free of the kind of attacks that were once common.
    text 3: LAHORE, Pakistan (AP) - A powerful bomb exploded near security forces guarding a famous Sufi shrine in the eastern city of Lahore on
    Wednesday, killing at least eleven people and wounding 20 others, police said.City police chief Ghazanfar Ali said police officers were the
    apparent target of bombing outside the shrine which is known as Data Darbar, and where a famous Sufi saint Ali Hajveri is buried.
    text 4: LAHORE, Pakistan (AP)—A powerful bomb exploded near security forces guarding a famous Sufi shrine in Pakistan onWednesday,
    killing at least five people and wounding 23 others, police said. ”It seems police officers who were doing their routine duty outside the Data
    Darbar shrine were the target,” Ali said, adding that some pilgrims and passers-by were among those wounded.
    text 5: LAHORE, Pakistan: A bomb targeting Pakistani police outside a Sufi shrine in the city of Lahore onWednesday (May 8) killed nine people,
    officials said.We are collecting forensic evidences to ascertain the nature of the blast," said Ashfaq Khan, deputy inspector general of police
    operations in Lahore, adding that nine people were killed and 24 wounded.
    text 6: LAHORE, Pakistan (Reuters) - A bomb targeting Pakistani police outside a major Sufi shrine in the city of Lahore onWednesday killed at
    least 10 people and wounded more than 20, officials said.Militant violence has since declined sharply in Pakistan after a sustained crackdown
    following the country’ s deadliest attack in 2014, which killed more than 150 people, many children, at a school in the western city of Peshawar.

    <Reasoning process>:
    (1)Specific event extraction result
    text 1: A bomb targeting Pakistani police outside a major Sufi shrine in the city of Lahore ; A bomb killed at least 10 people ; The blast went off at
    a police checkpoint near the Data Darbar
    text 2: Pakistan bombing kills at least nine people; A powerful bomb has exploded near security forces guarding a famous Sufi shrine in the
    eastern city of Lahore; Militant violence has declined sharply in Pakistan following a sustained crackdown
    text 3: A powerful bomb exploded near security forces guarding a famous Sufi shrine in the eastern city of Lahore ; A bomb killed at least eleven
    people ; City police chief said police officers were the apparent target of bombing
    text 4: A powerful bomb exploded near security forces guarding a famous Sufi shrine in Pakistan ; A bomb killed at least five people
    text 5: A bomb targeting Pakistani police outside a Sufi shrine in the city of Lahore ; A bomb killed nine people ; Police are collecting forensic
    evidences to ascertain the nature of the blast
    text 6: A bomb targeting Pakistani police outside a major Sufi shrine in the city of Lahore ; A bomb killed at least 10 people ; Militant violence has
    since declined sharply ; A sustained crackdown attack in 2014 at a school in the western city of Peshawar
    (2)Event abstraction and statistical results
    abstract event 1: A bomb targeting police near the Data Darbar Sufi shrine in Lahore, Pakistan (text1, text2, text3, text4, text5, text6)
    abstract event 2: A bomb killed at least 10 people (text1, text2, text3, text4, text5, text6)
    abstract event 3: Militant violence has declined sharply (text2, text6)
    abstract event 4: Police officers were the apparent target of bombing (text3, text5)
    abstract event 5: A sustained crackdown attack in 2014 at a school in the western city of Peshawar (text6)
    Select abstract event 1 and abstract event 2.
    (3)Final summary
    A bomb targeting police near the Data Darbar Sufi shrine in Lahore, Pakistan, killed at least 10 people.

    <Output>:
    A bomb targeting police near the Data Darbar Sufi shrine in Lahore, Pakistan, killed at least 10 people.

    ----

    Referring to [The process of summary reasoning] and [The example of summary reasoning], integrate the following input into a concise summary
    <Input>: {Document}
    <Output>:
    """

    prompt = PromptTemplate(template=coe_prompt, input_variables=['Document'])
    chain = prompt | llm 

    return print(chain.invoke({'Document': sample['document']}).content)

def CoKG_Summary(llm, sample):
    cokg_prompt = """
    Document: {Document}
    Based on the given document, identify the entities and relationships that are the components of the knowledge graph for summarization.

    knowledge Graph Components:
    - Head (Subject): The entity to be described
    - Relation (Predicate): The relationship between the subject and object
    - Tail (Object): The entity which describes or be related to the subject

    Remember that you should obtain the knowledge graph components with sequential steps as followings.

    Step 1. Identify Entities:
    Identify and extract entities from the document.
    Ensure that you capture the entities and describe them in simple and intuitive words. (e.g., people, places, organizations, key concepts, etc.).

    Step 2. Identify Relations:
    Identify and extract relations between the entities extracted from the document.
    Focus on specifying how predictates have connections with the entities.
    Ensure that you capture the relations and describe them in simple and intuitive words.

    Step 3. Evaluate Relation Strength:
    Identify and evaluate the strength of relations between the entities extracted from the document, assigning a score from 1 to 10 (1 being weakest, 10 strongest, and 5 middle). 
    Evaluate strength based on frequency (how often the relation appears) and contextual importance (how critical the relation is to the document's main themes). 
    Combine these evaluations to assign an overall score for each relation.

    Step 4. Review Document Information:
    Please check that the knowledge graph includes the information from the given document.
    Compare the knowledge graph with the given document to find any missing entities and relationships. 
    If you find any, please update the knowledge graph accordingly.

    Step 5. Consolidate Entities and Relations:
    Consolidate entities considered the same based on context into a consistent representation. (e.g., "IBM" and "International Business Machines" into "IBM", "Korean Netflix series" and "Korea Netflix series" into "Korean Netflix series").
    Consolidate relationships considered the same based on context into a consistent representation. (e.g., "works at" and "employed by" into "employed by", "resides in" and "lives in" into "resides in").

    Step 6. Summarize Document:
    Summarize the given document based on entities that have the strongest and most frequent relations in the extracted knowledge graph.
    Exclude unnecessary details and include only the essential core content in the summary.
    The summary is cohesive, flow naturally, and include only objective facts.

    The result in the specified JSON format. The keys in the JSON should be 'Knowledge Graph' and 'Summary'.
    For the knowledge graph, the key should be 'Head', and the values should be a list containing 'Relation', 'Tail' and 'Relation Strength', respectively.
    Answer in JSON without unnecessary introductory text or meta comments.
    """

    prompt = PromptTemplate(template=cokg_prompt, input_variables=['Document'])
    chain = prompt | llm | JsonOutputParser()

    return print(chain.invoke({'Knowledge_Graph': sample['knowledge_graph']}).content)