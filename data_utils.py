from rdflib import Graph, Literal, URIRef
import re
import json


def get_data(json_file: str):
    """ Return list from JSON formatted according to QALD format """
    with open(json_file, 'r') as f:
        data = json.load(f)['questions']

    return data


def get_results(g, query):
    """ Return list results for a given query on graph g """
    qres = []
    try:
        query_results = g.query(query)
        print(f'query results: {query_results}')
        for row in query_results:
            if isinstance(row, bool):
                qres.append(row)
            else:
                qres.append(row[0].toPython())
        print(qres)
        qres = list(set(qres))
    
    except Exception as e:
        print(e)   
        print('Cannot execute')

    return qres


def get_count(gt, qres, count=0):
    # Check executed SPARQL results are right or not
    if 'boolean' in gt['answers'][0]:
        if len(qres) == 1 and str(gt['answers'][0]['boolean']).lower() == str(qres[0]).lower():
            count += 1
            #correct_ids.append((id,1))
    else:
        results = gt['answers'][0]['results']['bindings']
        if len(results) == 1:
            if 'boolean' in results[0]:
                r = results[0]['boolean']
            else:
                r = results[0]['result']['value']
            if len(qres) == 1 and str(r) == str(qres[0]):
                count += 1
                #correct_ids.append((id,1))
        else:
            results = [str(x['result']['value']) for x in gt['answers'][0]['results']['bindings']]
            per = len([str(x) for x in qres if str(x) in results]) / float(len(results))
            if per > 0:
                count += per
                #correct_ids.append((id,count))
    return count


def eval(g, data, results):
    """ Evaluate results and return count 

    Parameters
    ----------------
    :parameter graph: KG
    :parameter data: ground truth results
    :parameter results: results from LLM

    Returns
    ----------------
    count of corrects
    """

    initial_count = 0
    count = 0
    
    for i, res in enumerate(results['results']):
        id = res['id']
        q = res['question']
        initial_query = res['initial_query']
        query = res['query']
        print(id, q)

        gt = [x for x in data if x['id']==id][0]
        print('Ground truth SPARQL:')
        gt_query = gt['query']['sparql']
        print(gt_query)
        print('Ground truth answer:')
        print(gt['answers'][0])

        ## Eval
        print('Generated SPARQL:')
        print(query)
        print('Generated res:')
        qres = []
        # Get initial query results
        initial_qres = get_results(g, initial_query)
        # Get refined query results
        qres = get_results(g, query)
        
        initial_count = get_count(gt, initial_qres, initial_count) 
        count = get_count(gt, qres, count)

        print(f'::::::: Initial Count: {initial_count}, Count: {count}')      
        print('=================================================')
                           
    return initial_count, count                       


def create_subset(data, g):
    """ Create 30 subset easy questions """
    ## Filter subset to construct easy set for the experiment
    filtered_questions = [question for question in data if question['id'] in [0,9,64,13,14,58,83,94]]
    for q in filtered_questions:
        q['question'][0]['string'] = q['question'][0]['string'].replace('creatures','beasts')

    # Populate with manually added questions
    additional = [
        {
            'id': 104,
            'question': [{'language': 'en', 'string': 'which creatures not speaking giant language do have chaotic neutral alignment?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#chaoticNeutral> .
                    MINUS {?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#GiantL>}
                }
            """}
        },
        {
            'id': 101,
            'question': [{'language': 'en', 'string': 'how many creatures are in the dataset?'}],
            'query': {'sparql': 'SELECT (COUNT(?creatures) AS ?n_creatures) WHERE {?creatures rdf:type <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#Beast>}'}
        },
        {
            'id': 105,
            'question': [{'language': 'en', 'string': 'which creatures not speaking terran language do have chaotic evil alignment?'}],
            'query': {'sparql': """ 
                SELECT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#chaoticEvil> .
                    MINUS {?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#TerranL>}
                }
            """}
        },
        {
            'id': 106,
            'question': [{'language': 'en', 'string': 'what creatures do have true neutral alignment?'}],
            'query': {'sparql': """
                SELECT DISTINCT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#trueNeutral> .
                }
            """}
        },
        {
            'id': 107,
            'question': [{'language': 'en', 'string': 'what creatures do speak boggard language?'}],
            'query': {'sparql': """
                SELECT DISTINCT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#BoggardL>
                }
            """}
        },
        {
            'id': 108,
            'question': [{'language': 'en', 'string': 'what is the average speed of creatures?'}],
            'query': {'sparql': """
                SELECT (AVG(?speed) AS ?avg_speed) WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasSpeedValue> ?speed
                }
            """}
        },
        {
            'id': 109,
            'question': [{'language': 'en', 'string': 'what is the average will value of creatures?'}],
            'query': {'sparql': """
                SELECT (AVG(?will) AS ?avg_will) WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasWillValue> ?will
                }
            """}
        },
        {
            'id': 102,
            'question': [{'language': 'en', 'string': 'how many creatures have chaotic neutral alignment?'}],
            'query': {'sparql': """
                SELECT (COUNT(?creatures) AS ?n_creatures) WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#chaoticNeutral>
                }
            """}
        },
        {
            'id': 103,
            'question': [{'language': 'en', 'string': 'how many creatures speak dwarven language?'}],
            'query': {'sparql': """
                SELECT (COUNT(?creatures) AS ?n_creatures) WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#DwarvenL>
                }
            """}
        },
        {
            'id': 117,
            'question': [{'language': 'en', 'string': 'what creatures speak draconic language and has neutral good alignment?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#DraconicL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#neutralGood>
                }
            """}
        },
        {
            'id': 118,
            'question': [{'language': 'en', 'string': 'what creatures do speak both giant language and auran language?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#GiantL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#AuranL>
                }
            """}
        },
        {
            'id': 119,
            'question': [{'language': 'en', 'string': 'how many creatures do speak aquan, draconic and common languages?'}],
            'query': {'sparql': """
                SELECT (COUNT(?creatures) AS ?n_creatures) WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#DraconicL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#AquanL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#CommonL>
                }
            """}
        },
        {
            'id': 118,
            'question': [{'language': 'en', 'string': 'what creatures do speak terran, ignan and giant languages?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#GiantL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#IgnanL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#TerranL>
                }
            """}
        },
        {
            'id': 122,
            'question': [{'language': 'en', 'string': 'what creatures have true neutral alignment do have armor class greater than 10?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasAlignment> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#trueNeutral>;  
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasACValue> ?AC_value FILTER (?AC_value > 10) 
                }
            """}
        },
        {
            'id': 121,
            'question': [{'language': 'en', 'string': 'what creatures speaking terran language do have armor class greater than 30?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#TerranL>;  
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasACValue> ?AC_value FILTER (?AC_value > 30) 
                }
            """}
        },
        {
            'id': 110,
            'question': [{'language': 'en', 'string': 'which creatures speaking common and draconic languages do have wisdom attribute more than 5?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#CommonL>;  
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#DraconicL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#wis> ?wis FILTER (?wis > 5) 
                }
            """}
        },
        {
            'id': 111,
            'question': [{'language': 'en', 'string': 'which creatures speaking goblin and terran languages do have dexterity attribute more than 5?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#TerranL>;  
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasLanguages> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#GoblinL>;
                        <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#dex> ?dex FILTER (?dex > 5) 
                }
            """}
        },
        {
            'id': 112,
            'question': [{'language': 'en', 'string': 'what is the average wisdom attribute for Phlogiston and PhlegmaticOozeSwarm?'}],
            'query': {'sparql': """
                SELECT ((?Phlogiston_wis+?PhlegmaticOozeSwarm_wis)/2 AS ?avg_wis) WHERE { 
                    <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#Phlogiston> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#wis> ?Phlogiston_wis. 
                    <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#PhlegmaticOozeSwarm> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#wis> ?PhlegmaticOozeSwarm_wis 
                }
            """}
        },
        {
            'id': 113,
            'question': [{'language': 'en', 'string': 'what is the average speed for Phasm and PhaseSpider?'}],
            'query': {'sparql': """
                SELECT ((?Phasm_speed+?PhaseSpider_speed)/2 AS ?avg_speed) WHERE { 
                    <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#Phasm> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasSpeedValue> ?Phasm_speed . 
                    <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#PhaseSpider> <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasSpeedValue> ?PhaseSpider_speed
                }
            """}
        },
        {
            'id': 114,
            'question': [{'language': 'en', 'string': 'what creatures have armor class greater than 40?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#hasACValue> ?AC_value FILTER (?AC_value > 40) 
                }
            """}
        },
        {
            'id': 115,
            'question': [{'language': 'en', 'string': 'what creatures do have wisdom attribute more than 15?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#wis> ?wis FILTER (?wis > 15) 
                }
            """}
        },
        {
            'id': 116,
            'question': [{'language': 'en', 'string': 'what creatures do have dexterity attribute less than 10?'}],
            'query': {'sparql': """
                SELECT ?creatures WHERE {  
                    ?creatures <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#dex> ?dex FILTER (?dex < 10) 
                }
            """}
        },
    ]

    for q in additional:
        print(f'Processing {q["id"]}')
        sparql_q = q['query']['sparql']
        # Construct results list
        res = []
        for r in g.query(sparql_q):
            res.append(r)
        # QALD format
        if len(res) == 1:
            q['answers'] = [{
                'head': {
                    'vars': ['result']
                },
                'results': {
                    'bindings': [
                        {
                            'result': {
                                'datatype': res[0][0].datatype.toPython(),
                                'type': 'literal',
                                'value': str(res[0][0].toPython())
                            }
                        }
                    ]
                }
            }]
        else:
            bindings = []
            for r in res:
                bindings.append({
                    'result': {
                        'type': 'uri',
                        'value': r[0].toPython()
                    }
                })
            q['answers'] = [{
                'head': {
                    'vars': ['result']
                },
                'results': {
                    'bindings': bindings
                }
            }]

    filtered_questions += additional

    for q in filtered_questions:
        q['question'][0]['string'] = q['question'][0]['string'].replace('beasts', 'creatures')

    print(f'Total questions: {len(filtered_questions)}')
    subset = {
        'dataset': {'id': 'beastiary_subset'},
        'questions': filtered_questions
    }
    with open('data/beastiary_subset.json', 'w') as f:
        json.dump(subset, f, indent=2)


if __name__ == '__main__':
    # Load data
    data_file = 'data/beastiary_with_qald_format.json'
    data = get_data(data_file)
    # Load graph
    rdf_file = 'data/beastiary_kg.rdf'
    g = Graph()
    g.parse(rdf_file)
        
    ## Create subset for experiments 30
    #create_subset(data, g)
        
    ## Test
    print(g)
    print(len(g))
    tq = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://www.semanticweb.org/annab/ontologies/2022/3/ontology#>

    SELECT (COUNT(DISTINCT ?creature) AS ?triple_count)
    WHERE {
        ?creature a :Beast .
    }
    """
    for r in g.query(tq):
        print(r)
    
