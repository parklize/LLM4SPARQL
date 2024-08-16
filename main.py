from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.chains import GraphSparqlQAChain
from langchain.prompts import PromptTemplate
from langchain.graphs import RdfGraph
from langchain.schema.output_parser import BaseOutputParser
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from prompts import SPARQL_GENERATION_TEMPLATE, IMPORTANT_TYPE_PROPERTY_TEMPLATE, SPARQL_GENERATION_WITH_TYPE_PROPERTY_TEMPLATE

import re
import transformers
import rdflib
import os
import time
import json
import torch
import langchain
import data_utils

# Print langchain package location
print(langchain.__file__)
set_seed(42)
torch.manual_seed(42)


class TypePropertyParser(BaseOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, output):
        print(f'Prompt output: {output.strip()}')
        type_list, p_list = get_type_properties(output)
        return type_list, p_list, output


class SPARQLSyntaxParser(BaseOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, output):
        print(f'SPARQL prompt output: {output.strip()}')
        # Extract 1st SPARQL query from prompt answer
        gen_query = re.search('(.*?\}).*\[\/SPARQL\].*', output, flags=re.DOTALL)
        if gen_query:
            gen_query = gen_query.group(1)
            # Add missing end bracket if any
            for x in range(gen_query.count('{') - gen_query.count('}')):
                gen_query += '}'
            # Add missing bracket with AVG(?s) and similar patterns with variable
            pattern = re.compile(r'(?:MAX|MIN|AVG)\(.*\?.*\)', re.IGNORECASE)
            matches = pattern.findall(gen_query)
            print(f'gen query: {gen_query}')
            for m in matches:
                if 'AS' not in m:
                    var = m[m.index('(')+1:m.index(')')]
                    agg = m[:m.index('(')]
                    rep = '(' + m + ' AS ' + var.strip() + '_' + agg + ')'
                    gen_query = gen_query.replace(m, rep)
            print(f'parsed gen query: {gen_query}')

        return gen_query


def get_type_properties(output: str, top_k=5):
    """ Extract node types and properties from TYPE and PROPERTY tags """
    type_list = []
    types = re.search('.*?\[TYPE\](.*?)\[\/TYPE\]', output, flags=re.DOTALL)
    if types:
        types = types.group(1)
        type_list = [x.replace('\n','') for x in re.split(',|\n', types)]
    p_list = []
    if '[/PROPERTY]' not in output:
        output += '[/PROPERTY]' 
    properties = re.search('.*?\[PROPERTY\](.*?)\[\/PROPERTY\]', output, flags=re.DOTALL)
    if properties:
        properties = properties.group(1)
        p_list = [x.replace('\n','') for x in re.split(',|\n', properties)]
    p_list = [x.strip() for x in p_list if len(x)>1]
    type_list = [x.strip() for x in type_list if len(x)>1]
    print(f'\nNode types: {type_list}')
    print(f'Properties: {p_list}')
    if top_k != -1:
        return type_list[:top_k], p_list[:top_k]
    else:
        return type_list, p_list


def get_llm(hub_id: str):
    """ Return HuggingFacePipeline in Langchain """    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        hub_id,
    )
    
    model_config = transformers.AutoConfig.from_pretrained(
        hub_id,
        num_return_sequences=1,
        repeatition_penalty=1.1,
        return_full_text=False,
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        hub_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        torch_dtype='auto',
        device_map='auto',
    )
    
    llm.eval()
    torch.no_grad()
    
    pipeline = transformers.pipeline(
        model=llm, 
        tokenizer=tokenizer, 
        task="text-generation",
        max_new_tokens=1024,
    )
    
    hf = HuggingFacePipeline(
        pipeline=pipeline,
    )

    return hf


def get_graph(source_file: str):
    """ Return graph in Langchain """

    graph = RdfGraph(
        #source_file='http://www.w3.org/People/Berners-Lee/card',
        source_file=source_file,
        serialization='xml',
        standard='rdf',
        local_copy='test.rdf'
    )
    
    schema = re.sub(' \(.*?, None\)', '', graph.get_schema)
    schema = schema.replace('In the following, each IRI is followed by the local name and optionally its description in parentheses.','')
    schema = schema.replace('The RDF graph supports the following node types:','The RDF graph supports the following node types:\n[TYPE]')
    schema = schema.replace('The RDF graph supports the following relationships:','[/TYPE]\nThe RDF graph supports the following relationships:\n[PROPERTY]')
    schema += '[/PROPERTY]'
    print(schema)

    # Extract properties and node types as list
    type_list, p_list = get_type_properties(schema, top_k=-1)

    return graph, schema, type_list, p_list


def inference(
        hf, 
        graph, 
        schema, 
        gt_types, 
        gt_properties, 
        question, 
        id, 
        results,
        type_property_extraction_retry=5,
        sparql_gen_retry=0,
        top_k_tp=5,
        triple_lim=1,
    ):
    """ 
    Infer SPARQL query for the given question and append to results dict 

    Parameters:
    -------------
    :parameter hf: HuggingFace model
    :parameter graph: Knowedge Graph loaded
    :parameter gt_types: Types in KG
    :parameter gt_properties: Properties in KG
    :parameter question: question to be answered with SPARQL
    :parameter id: id of the question in the given dataset with qlad format
    :parameter results: results dictionary to be updated, will be seriealized to file later
    :parameter: type_property_extraction_retry: 0 to turn off, 1 to turn on
    :parameter: sparql_gen_retry: the number of times to retry after first try of getting SPARQL
    :parameter: top_k_tp: the number of top-k types/properties relevant to the question
    :parameter: triple_lim: the number of triples to be extracted for each type/property

    Returns:
    -------------
    Nothing, but update the given results dict

    """
    
    ## Ranked list of node types and properties

    IMPORTANT_TYPE_PROPERTY_PROMPT = PromptTemplate(
        input_variables=['schema', 'prompt'],
        template=IMPORTANT_TYPE_PROPERTY_TEMPLATE
    )
    type_property_parser = TypePropertyParser()
    type_property_chain = IMPORTANT_TYPE_PROPERTY_PROMPT | hf | type_property_parser
    type_list, p_list, prompt_output = type_property_chain.invoke({
        'prompt': question, 
        'schema': schema,
        'previous_answer': '',
    })

    if type_property_extraction_retry != 0:
        ## Component for retrying
        if len(type_list) == 0 or len(p_list) == 0:
            print('\nRetrying as the length of types or properties is zero')
            previous_answer = f"""
            Your previous answer is as below in between [PA] and [/PA] tags:
            [PA]
            {prompt_output}
            [/PA]
            which is not formatted according to the requirements.
            Please refine your answer based on the following requirements.
            """
            type_list, p_list, _ = type_property_chain.invoke({
                'prompt': question,
                'schema': schema,
                'previous_answer': previous_answer
            })
        elif (len([x for x in type_list if x not in gt_types]) > 0) or (len([x for x in p_list if x not in gt_properties]) > 0):
            print(f'Ground truth types: {gt_types}')
            print('\nRetrying as there are types and properties not exist in the schema')
            print(f'Non-existing node type: {[x for x in type_list if x not in gt_types]}')
            print(f'Non-existing properties: {[x for x in p_list if x not in gt_properties]}')
            previous_answer = f"""
            Your previous answer is as below in between [PA] and [/PA] tags:
            [PA]
            {prompt_output}
            [/PA]
            which contains node types or properties not in the given schema.
            Please update your answer based on the following requirements.
            """
            type_list, p_list, _ = type_property_chain.invoke({
                'prompt': question,
                'schema': schema,
                'previous_answer': previous_answer
            })

        type_property_extraction_retry -= 1

    ## Get list of triples as string
    context_triples = []
    top_k_tp = top_k_tp
    limit = triple_lim
    print(f'\nTop node types: {type_list[:top_k_tp]}')
    print(f'Top properties: {p_list[:top_k_tp]}\n')
    for t in type_list[:top_k_tp]:
        _query = 'SELECT ?s WHERE {?s a '+t+'} ORDER BY RAND() LIMIT '+str(limit)
        print(_query)
        res = graph.query(_query)
        for r in res:
            print(r)
            r0 = '<'+r[0].toPython()+'>' if type(r[0]) is rdflib.term.URIRef else r[0].toPython()
            context_triples.append(f'({r0} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {t})')
    for p in p_list[:top_k_tp]:
        _query = 'SELECT ?s ?o WHERE {?s '+p+' ?o} ORDER BY RAND() LIMIT '+str(limit)
        print(_query)
        res = graph.query(_query)
        for r in res:
            print(r)
            r0 = '<'+r[0].toPython()+'>' if type(r[0]) is rdflib.term.URIRef else r[0].toPython()
            r1 = '<'+r[1].toPython()+'>' if type(r[1]) is rdflib.term.URIRef else r[1].toPython()
            context_triples.append(f'({r0} {p} {r[1]})')

    context_triples = '\n'.join(context_triples)
    print(f'\nContext_triples: \n{context_triples}')

    ## Component: SPARQL generation
    sparql_parser = SPARQLSyntaxParser()
    ## Baseline
    #SPARQL_GENERATION_SELECT_PROMPT = PromptTemplate(
    #    input_variables=['schema', 'prompt'],
    #    template=SPARQL_GENERATION_TEMPLATE
    #)
    #sparql_gen_chain = SPARQL_GENERATION_SELECT_PROMPT | hf | sparql_parser

    # With enhancer
    SPARQL_GENERATION_WITH_TYPE_PROPERTY_PROMPT = PromptTemplate(
        input_variables=['schema', 'prompt', 'subset', 'previous_answer'],
        template=SPARQL_GENERATION_WITH_TYPE_PROPERTY_TEMPLATE
    )
    # Define chain
    sparql_gen_chain = SPARQL_GENERATION_WITH_TYPE_PROPERTY_PROMPT | hf | sparql_parser

    def _get_query(previous_answer='', sparql_gen_retry=5):
        """ Get SPARQL query
        
        Parameters
        ----------------
            previous_answer: Previous answer extracted from LLM
            sparql_gen_retry: The times of retrying SPARQL query generation
        
        Return
        ----------------
            query and exectuable (boolean) 
        """
        initial_query = ''
        query = ''
        executable = False

        gen_query = sparql_gen_chain.invoke({
            'prompt': question,
            'schema': schema,
            'subset': context_triples,
            'previous_answer': previous_answer
        })
        if previous_answer == '':
            initial_query = gen_query
   
        if gen_query:
            print('###########################')
            print(f'Chain result:\n{gen_query.strip()}')
            print('###########################\n')
        else:
            print('No query extracted')

        # Regex
        #gen_query = re.search('(SELECT.*?\}).*\[\/SPARQL\].*', res, flags=re.DOTALL)
        
        if gen_query:
            #gen_query = gen_query.group(1)
            # Refine syntax errors
            gen_query = gen_query.replace('?', ' ?')
            print('#############################')
            print(f'Generated SPARQL:\n{gen_query}')
            print('#############################\n')
            query = gen_query
            # Execute
            try:
                sparql_res = graph.query(gen_query)
                #print(f'SPARQL res:\n{str(sparql_res)}')
                print(f'Results len: {len(sparql_res)}')
                executable = True
                # SPARQL enhancer
                if sparql_gen_retry > 0:
                    sparql_gen_retry -= 1
                    print(f'\nChecking and refining previous SPARQL, retry quota left:{sparql_gen_retry}')
                    previous_answer = f"""
                    Your previously generated SPARQL query for the given question is as below in between [PA] and [/PA] tags:
                    [PA]
                    {gen_query}
                    [/PA]
                    You need to add, remove, modify parts of the SPARQL query to answer the question if needed. Let's check step by step.  
                    Check which SPARQL keywords (e.g., COUNT, MIN, MAX, AVG, ...) are relevant to the question but missing from the query, modify the query by adding, removing, replacing keywords if needed.
                    Check the node types used in the query whether they are in the given schema and relevant to the question, modify them using the ones from the schema if needed.
                    Check the properties used in the query whether they are in the given schema and relevant to the question, modify them using the ones from the schema if needed.
                    Check each subject and object in the query is relevant to the question, modify them if needed.
                    Check the naming and capitalization conventions of each subject and object in the query whether they follow the same conventions of the examples from RDF graph above between [SUBSET] and [/SUBSET] tags, modify them if needed. 
                    """
                    _, query, executable = _get_query(
                        previous_answer = previous_answer,
                        sparql_gen_retry = sparql_gen_retry
                    ) 
            except Exception as e:
                print(e)
                if sparql_gen_retry > 0:
                    sparql_gen_retry -= 1
                    print(f'\nRetrying due to non-executable SPARQL, retry quota left:{sparql_gen_retry}')
                    previous_answer = f"""
                    Your previous generated SPARQL query for the given question is as below between [PA] and [/PA] tags.
                    It cannot be executed due to the error as below in between [ERR] and [/ERR] tags:
                    [PA]
                    {gen_query}
                    [/PA]
                    [ERR]
                    {e}
                    [/ERR]
                    Check the SPARQL query and provide an updated query to answer the given question based on the following requirements.
                    Do not use any node types or properties that are not in the given schema.
                    """
                    _, query, executable = _get_query(
                        previous_answer = previous_answer,
                        sparql_gen_retry = sparql_gen_retry
                    ) 
        else:
            print('No valide query')
            query = 'No valide query'
            if sparql_gen_retry > 0:
                sparql_gen_retry -= 1
                print(f'\nRetrying due to non-valid SPARQL, retry quota left:{sparql_gen_retry}')
                _, query, executable = _get_query(
                    previous_answer = '',
                    sparql_gen_retry = sparql_gen_retry
                ) 
        
        return initial_query, query, executable

    gen_query, query, executable = _get_query(sparql_gen_retry=sparql_gen_retry)

    results['results'].append({
        'id': id,
        'question': question,
        'initial_query': gen_query,
        'query': query,
        'executable': executable
    })

    print('+++++++++++++++++++++++++++++++++++++++\n')


if __name__ == '__main__':
    JSON_FILE = 'data/beastiary_subset.json'
    RDF_FILE = 'data/beastiary_kg.rdf'
    HUB_ID = 'codellama/CodeLlama-7b-Instruct-hf'
    ids_to_test = []
    print(f'Testing {len(ids_to_test) if len(ids_to_test)>0 else "All"} examples')

    hf = get_llm(HUB_ID)
    graph, schema, gt_types, gt_properties = get_graph(RDF_FILE) 
    data = data_utils.get_data(JSON_FILE)
    ids = [q['id'] for q in data]

    if len(ids_to_test) == 0:
        ids_to_test = ids
    print(len(set(ids_to_test)), ids_to_test)

    results = {'results': []}
    num = 10
    sparql_gen_retry = 5
    separate = True
    correct_count_dict = {}
    correct_count_dict_with_initial_query = {}

    for i, question in enumerate(data):
        q = question['question'][0]['string']
        id = question['id']
        correct_count_dict[id] = 0
        correct_count_dict_with_initial_query[id] = 0

        for j in range(num):
            start_time = time.perf_counter()
            #print(f'\nStart inference for QID{id} for {j}-th run')
            #print(f'Question: {q}')
            if separate:
                if id in ids_to_test:
                    #correct_count_dict[id] = 0
                    print(f'\nStart inference for QID{id} for {j}-th run')
                    print(f'Question: {q}')
                    stime = time.time()
                    inference(
                        hf, 
                        graph, 
                        schema, 
                        gt_types, 
                        gt_properties, 
                        q, 
                        id, 
                        results, 
                        sparql_gen_retry=sparql_gen_retry
                    )
                    elapsed_time = time.time() - stime
                    print(f'{elapsed_time: .2f} seconds used for inference...')
                    # Save details of generated SPARQL queries
                    with open(f'./results/id{id}_run{j}_baseline_results.json', 'w') as f:
                        json.dump(results, f)
                    # Eval
                    initial_score, score = data_utils.eval(graph, data, results)
                    print(initial_score, score)
                    correct_count_dict[id] += score
                    correct_count_dict_with_initial_query[id] += initial_score
                    print('correct_count_dict_with_initial_query', correct_count_dict_with_initial_query)
                    print('correct_count_dict', correct_count_dict)
                    # Reset
                    results = {'results': []}
            else:
                pass
                #inference(hf, graph, schema, gt_types, gt_properties, q, id, results)

            finish_time = time.perf_counter()
            print(f'Finished {finish_time-start_time} seconds')

    # Dump results
    cur_timestamp = time.time()
    timestamp_str = time.strftime('%Y%m%d%H%M%S', time.localtime(cur_timestamp))
    with open(f'./results/llm_sparql_results_{timestamp_str}_wt_enhancer.json', 'w') as f:
        json.dump(correct_count_dict, f)
    with open(f'./results/llm_sparql_results_{timestamp_str}.json', 'w') as f:
        json.dump(correct_count_dict_with_initial_query, f)
