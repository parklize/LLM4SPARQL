SPARQL_GENERATION_TEMPLATE = """[INST]<<SYS>>You are a helpful agent to write SPARQL query. 
Your task is to provide a SPARQL SELECT or ASK statement for querying a RDF graph to answer a given question.
Use only the node types and properties provided in the schema, and include subjects or objects needed.
Do not use any node types and properties that are not explicitly provided.
Schema is as below between [SCHEMA] and [/SCHEMA] tags.
[SCHEMA]
{schema}
[/SCHEMA]

Note: Your answer is always a SPARQL statement which has correct syntax.
Write your SPARQL query between [SPARQL] and [/SPARQL] tags.
Include all necessary prefixes in your SPARQL query.
Do not contain any other text except SPARQL query between [SPARQL] and [/SPARQL] tags.
<</SYS>>

The question to be answered by your SPARQL query is between [QUESTION] and [/QUESTION] tags below.
[QUESTION]
{prompt}
[/QUESTION]

[SPARQL]
<FILL_ME>
[/SPARQL][/INST]"""


SPARQL_GENERATION_WITH_TYPE_PROPERTY_TEMPLATE = """[INST]<<SYS>>You are a helpful agent to write SPARQL query. 
Your task is to provide a SPARQL SELECT or ASK statement for querying a RDF graph to answer a given question. 
Use only the node types and properties provided in the schema, and include subjects or objects needed based on their IRI patterns regarding each property or node type from the examples in the RDF graph provided below.
Do not use any node types and properties that are not explicitly provided.
Schema is as below between [SCHEMA] and [/SCHEMA] tags.
[SCHEMA]
{schema}
[/SCHEMA]

Note: Your answer is always a SPARQL query which has correct syntax.
Write your SPARQL query between [SPARQL] and [/SPARQL] tags.
Include all necessary prefixes in your SPARQL query.
Do not contain any other text except SPARQL query between [SPARQL] and [/SPARQL] tags.
<</SYS>>

Some examples from the RDF graph in the form of (subject, property, object) are between [SUBSET] and [/SUBSET] tags below. Use these naming and capitalization conventions to infer IRI patterns of subjects or objects regarding each property or node type that could be relevant to the question below. 
[SUBSET]
{subset}
[/SUBSET]

The question to be answered by your SPARQL query is between [QUESTION] and [/QUESTION] tags below.
[QUESTION]
{prompt}
[/QUESTION]

{previous_answer}
From the given schema, the SPARQL query to answer the above question is:
[SPARQL]
<FILL_ME>
[/SPARQL][/INST]"""


IMPORTANT_TYPE_PROPERTY_TEMPLATE = """[INST]<<SYS>>You are a helpful agent to pick a list of node types and properties from a RDF graph regarding a question. Your task is to provide ranked list of node types and properties from a given schema which are highly relevant to answer a given question.
Use only the node types and properties provided in the schema.
Do not use any node types and properties that are not explicitly provided.
Schema is as below between [SCHEMA] and [/SCHEMA] tags.
[SCHEMA]
{schema}
[/SCHEMA]
<</SYS>>

The question is between [QUESTION] and [/QUESTION] tags below.
[QUESTION]
{prompt}
[/QUESTION]

{previous_answer}

From the given schema and the above question, the five most relevant node types are separated by comma in between [TYPE] and [/TYPE] tags, followed by the five most relevant properties are seperated by comma in between [PROPERTY] and [/PROPERTY] tags as follows:
<FILL_ME>
"""
