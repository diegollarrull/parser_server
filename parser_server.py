import spacy
from spacy_stanza import StanzaLanguage
import stanza
from stanza.server import CoreNLPClient
import networkx as nx
from spacy import displacy
from itertools import groupby
from spacy.tests.util import get_doc
from spacy.vocab import Vocab
from spacy_conll import ConllFormatter
from operator import itemgetter
import re
from typing import List, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel



''' 
Required classes
'''
class Configuration:
    # options are "spacy_manual", "networkx"
    tree_format = "spacy_manual"
    # other options are "conll", "custom" 
    output_format = ""
    img_dir = "parser_analysis/img"
    txt_dir = "parser_analysis/txt"

    @staticmethod
    def set_output_file(f):
        Configuration.output_file = f



class parser_req(BaseModel):
    text: str
    test_mode : bool

class parser_response(BaseModel):
    G: Dict
    data : List

class enabled_response(BaseModel):
    parsers : List[str]

class set_req(BaseModel):
    parsers : List[str]

class unify_req(BaseModel):
    parser_list : List[Dict]

class unify_response(BaseModel):
    G: Dict


class Parser_pipeline:

    def spacy_model(model=None, model_type="best"):
        if model is None:
            nlp_spacy = spacy.load('en_core_web_sm')
        else:
            model_path = "models/models-" + model + "/model-" + model_type
            nlp_spacy = spacy.load(model_path)
        return nlp_spacy

    # by default, all parsers are enabled
    coreNLP3_enabled = True
    coreNLP4_enabled = True
    nlp_spacy_enabled = True
    nlp_spacy_joint_enabled = True
    nlp_stanza_enabled = False

    # SPACY onto Pipeline
    nlp_spacy = spacy_model()
    conllformatter = ConllFormatter(nlp_spacy)
    nlp_spacy.add_pipe(conllformatter, after='parser')

    # SPACY JOINT Pipeline
    nlp_spacy_joint = spacy_model("JOINT")
    conllformatter = ConllFormatter(nlp_spacy_joint)
    nlp_spacy_joint.add_pipe(conllformatter, after='parser')

    # Stanza Pipeline (deferred startup)
    snlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse", depparse_batch_size = 500)
    nlp_stanza = StanzaLanguage(snlp)

    @staticmethod
    def set_corenlp_paths(test_mode):
        if test_mode:
            # CoreNLP 3 Parser_pipeline
            Parser_pipeline.coreNLP3_client = CoreNLPClient(start_server=False, endpoint='http://localhost:9050',
                                            annotators=['depparse', 'lemma'])
            # CoreNLP 4 Parser_pipeline
            Parser_pipeline.coreNLP4_client = CoreNLPClient(start_server=False, endpoint='http://localhost:9055',
                                            annotators=['depparse', 'lemma'])
        else:
            # CoreNLP 3 Parser_pipeline
            Parser_pipeline.coreNLP3_client = CoreNLPClient(start_server=False, endpoint='http://localhost/corenlp',
                                            annotators=['depparse', 'lemma'])
            # CoreNLP 4 Parser_pipeline
            Parser_pipeline.coreNLP4_client = CoreNLPClient(start_server=False, endpoint='http://localhost/corenlp4',
                                            annotators=['depparse', 'lemma'])

    @staticmethod
    def set_parsers(enabled_parsers):
        if "corenlp3" in enabled_parsers:
            Parser_pipeline.coreNLP3_enabled = True
        else:
            Parser_pipeline.coreNLP3_enabled = False

        if "corenlp4" in enabled_parsers:
            Parser_pipeline.coreNLP4_enabled = True
        else:
            Parser_pipeline.coreNLP4_enabled = False

        if "spacy_onto" in enabled_parsers:
            # Spacy (default) Ontonotes Parser_pipeline
            Parser_pipeline.nlp_spacy_enabled = True
        else:
            Parser_pipeline.nlp_spacy_enabled = False

        if "spacy_ud_joint" in enabled_parsers:
            # Spacy (UD) Joint Parser_pipeline
            Parser_pipeline.nlp_spacy_joint_enabled = True
        else:
            Parser_pipeline.nlp_spacy_joint_enabled = False

        if "stanza" in enabled_parsers:
            Parser_pipeline.nlp_stanza_enabled = True
        else:
            Parser_pipeline.nlp_stanza_enabled = False


    def __str__(self):
        return repr(self)



''' 
FastAPI app and endpoint definition
'''

app = FastAPI()

@app.post("/deps_spacy_joint", response_model=parser_response)
async def deps_spacy_joint(pr : parser_req):
    sent = pr.text
    textlines = ["---------------- spaCy JOINT----------------" + '\n']
    doc = Parser_pipeline.nlp_spacy_joint(sent)
    G, data = generate_merge_and_output_data(doc, textlines)
    return ({"G": G, "data": data})

@app.post("/deps_spacy_onto", response_model=parser_response)
async def deps_spacy_onto(pr : parser_req):
    sent = pr.text
    textlines = ["---------------- spaCy OntoNotes----------------" + '\n']
    doc = Parser_pipeline.nlp_spacy(sent)
    G, data = generate_merge_and_output_data(doc, textlines)
    return({ "G" : G, "data" : data})

@app.post("/deps_stanza", response_model=parser_response)
async def deps_stanza(pr : parser_req):
    sent = pr.text
    textlines = ["---------------- stanza ----------------" + '\n']
    doc = Parser_pipeline.nlp_stanza(sent)
    G, data = generate_merge_and_output_data(doc, textlines)
    return ({"G": G, "data": data})

@app.post("/deps_corenlp3", response_model=parser_response)
async def deps_corenlp3(pr : parser_req):
    sent = pr.text
    textlines = ["---------------- coreNLP 3 ----------------" + '\n']
    Parser_pipeline.set_corenlp_paths(pr.test_mode)
    doc = Parser_pipeline.coreNLP3_client.annotate(sent, annotators=['depparse', 'lemma'])
    G, data = corenlp_ann_to_spacy_doc(doc, textlines)
    return ({"G": G, "data": data})

@app.post("/deps_corenlp4", response_model=parser_response)
async def deps_corenlp4(pr: parser_req):
    sent = pr.text
    textlines = ["---------------- coreNLP 4 ----------------" + '\n']
    Parser_pipeline.set_corenlp_paths(pr.test_mode)
    doc = Parser_pipeline.coreNLP4_client.annotate(sent, annotators=['depparse', 'lemma'])
    G, data = corenlp_ann_to_spacy_doc(doc, textlines)
    return ({"G": G, "data": data})

@app.post("/enabled_parsers", response_model=enabled_response)
async def enabled_parsers():
    parsers = []
    if Parser_pipeline.nlp_spacy_enabled:
        parsers.append("deps_spacy_onto")
    if Parser_pipeline.nlp_spacy_joint_enabled:
        parsers.append("deps_spacy_joint")
    if Parser_pipeline.nlp_stanza_enabled:
        parsers.append("deps_stanza")
    if Parser_pipeline.coreNLP3_enabled:
        parsers.append("deps_corenlp3")
    if Parser_pipeline.coreNLP4_enabled:
        parsers.append("deps_corenlp4")

    return({"parsers" : parsers})


@app.post("/set_parsers", response_model=enabled_response)
async def set_parsers(sp : set_req):
    Parser_pipeline.set_parsers(sp.parsers)
    return({"parsers" : sp.parsers})


'''
Helper functions
'''

def corenlp_to_spacy_doc(words, pos, heads, deps, tags, ents, lemmas):
    vocab = Vocab(strings=words)
    return get_doc(vocab, words, pos, heads, deps, tags, ents, lemmas)

def corenlp_ann_to_spacy_doc(doc, textlines):

    words  = []
    tags   = []
    heads  = []
    deps   = []
    lemmas = []
    pos = None
    ents = None

    idx = 1
    for st in doc.sentence:
        words_iter = [i.word for i in st.token]
        words += words_iter
        tags  += [i.pos for i in st.token]
        targets = {i.target : (i.source - i.target, i.dep) for i in st.enhancedPlusPlusDependencies.edge}
        heads += [targets.get(j, (0, 'ROOT'))[0] for j in range(1,len(words_iter) + 1)]
        deps  += [targets.get(j, (0, 'ROOT'))[1] for j in range(1,len(words_iter) + 1)]
        lemmas += [i.lemma.lower() for i in st.token]

    doc_spacy = corenlp_to_spacy_doc(words, pos, heads, deps, tags, ents, lemmas)

    return (generate_merge_and_output_data(doc_spacy, textlines))



def generate_merge_and_output_data(doc, textlines):
    data = []

    # This is just to prevent warnings elevated to errors in PyCharm
    G = None
    words = None
    arcs = None

    if Configuration.tree_format == "networkx":
        G = nx.DiGraph()
    if Configuration.tree_format == "spacy_manual":
        words = []
        arcs  = []

    for token in doc:
        target = token.text
        source = token.head.text
        rel = token.dep_

        if Configuration.tree_format == "networkx":
            G.add_edge(source, target, label= rel)

        if Configuration.tree_format == "spacy_manual":
            governor = token.head.i
            dependent = token.i
            tagset = list(set([i for i in [token.tag_] + [token.pos_] if i != ""]))
            pos = "|".join(tagset)
            lemma = token.lemma_
            words += [{'text': target, 'tag': pos, 'pos': pos, 'lemma' : lemma}]

            if (governor != dependent):
                arcs  += [{ 'start': min(governor + 1, dependent + 1),
                            'end': max(governor + 1, dependent + 1),
                            'label': rel,
                            'dir': 'left' if governor > dependent else 'right'}]
            else:
                arcs += [{'start': 0,
                          'end': dependent,
                          'label': rel,
                          'dir': 'right'}]

        if Configuration.output_format == "custom":
            children = [child for child in token.children]
            textlines += [target + ' - ' + rel + ' - ' + source + ' - ' + str(children) + '\n']
            Configuration.output_file.writelines([textlines[0]] + sorted(textlines[1:]))
            Configuration.output_file.write('\n')
        data += [[target, source, rel]]

    if Configuration.tree_format == "spacy_manual":
        words = [{'text': '[ROOT]', 'tag': 'ROOT', 'lemma' : "ROOT"}] + words
        G = {'words': words, 'arcs': arcs}

    if Configuration.output_format == "conll":
        Configuration.output_file.writelines([textlines[0]])
        try:
            #this doesn't work for CoreNLP because it doesn't implement a Parser_pipeline
            Configuration.output_file.writelines([doc._.conll_str])
            Configuration.output_file.write('\n')
        except AttributeError:
            Configuration.output_file.writelines(doc_to_conllu(doc))
        except IndexError:
            Configuration.output_file.writelines(doc_to_conllu(doc))
        except TypeError:
            Configuration.output_file.writelines(doc_to_conllu(doc))
    return(G, data)

def doc_to_conllu(doc, sent_id = 1, prefix =""):
    outlines = []

    for sent in doc.sents:
        outlines += ["# sent_id = {}".format(prefix+str(sent_id)) + "\n"]
        outlines += ["# text = {}".format(sent.sent) + "\n"]

        for i, word in enumerate(sent):
            #Find head
            if word.dep_.lower().strip() == 'root':
                head_idx = 0
            else:
                head_idx = word.head.i + 1 - sent[0].i

            #Find feature tag (if available)
            ftidx = word.tag_.find("__") + 2
            feature_tag=word.tag_[ftidx:]

            linetuple = [
                str(i+1),                                        #ID: Word index.
                word.text,                                       #FORM: Word form or punctuation symbol.
                word.lemma_.lower(),                        #LEMMA: Lemma or stem of word form.
                '_',                                   #UPOSTAG: Universal part-of-speech tag drawn
                                                            # from revised version of the Google universal
                                                            # POS tags.
                word.tag_,                                        #XPOSTAG: Language-specific part-of-speech tag;                                            # underscore if not available.
                '_' if feature_tag == "" else feature_tag,  #FEATS: List of morphological features from the
                                                            # universal feature inventory or from a defined
                                                            # language-specific extension; underscore if not
                                                            # available.
                str(head_idx),                                   #HEAD: Head of the current token, which is
                                                            # either a value of ID or zero (0).
                word.dep_.lower(),                          #DEPREL: Universal Stanford dependency relation
                                                            # to the HEAD (root iff HEAD = 0) or a defined
                                                            # language-specific subtype of one.
                '_',                                        #DEPS: List of secondary dependencies.
                '_'                                         #MISC: Any other annotation.
            ]
            outlines += ["\t".join(linetuple) + "\n"]
        sent_id+=1

    outlines += ["\n"]
    return outlines

def serve_displacy(sentence_list):
    displacy.serve(sentence_list, style="dep", manual = True)

def render_displacy(sentence_list):
    return(displacy.render(sentence_list, style="dep", options = { "color" : "#6d5d91", 'distance' : 300, 'arrow_spacing' : 30}, manual = True))


@app.post("/unify_parses", response_model=unify_response)
async def unify_parses(graph_list : unify_req):
    #logging.debug("Unifying parses - start")
    # at least one graph is required
    rv = {"words" : [], "arcs" : []}
    n = 1
    for graph in graph_list.parser_list:
        g2 = pairwise_merge_unification(rv, graph, n)
        n += 1
        rv = g2
    #logging.debug("Unifying parses - end")
    return ({"G" : rv})

def pairwise_merge_unification(a, b, n):

    for i in range(len(b['words'])):
        b['words'][i]['index'] = i

    if len(a['words']) > 0 and 'index' not in a['words'][0].keys():
        # only reindex if needed
        for i in range(len(a['words'])):
               a['words'][i]['index'] = i

    words = groupby(sorted(a['words'] + b['words'], key=itemgetter('index')), key=itemgetter('index'))
    arcs = groupby(sorted(a['arcs'] + b['arcs'], key=itemgetter('start', 'end', 'dir')), key=itemgetter('start', 'end', 'dir'))

    words_dict = {k: list(g) for k, g in words}
    arcs_dict = {k: list(g) for k, g in arcs}

    out_words = []
    out_arcs = []

    for k, v in words_dict.items():
        index = k
        # assuming all lemmas are the same
        text = [i['text'] for i in v][0]
        tags = [j  for i in v for j in i['tag'].split("|")]
        lemma = [i['lemma'] for i in v][0]
        rv_dict = [{'text': text, 'lemma' : lemma, 'tag': "|".join(set(tags)), 'index': index}]
        out_words += rv_dict

    for k, v in arcs_dict.items():
        start = k[0]
        end = k[1]
        labels = [j for i in v for j in re.sub(", w =.*","",i['label']).lower().replace(":","").split("|")]
        dirs = [j  for i in v for j in i['dir'].split("|")]
        arc_sum = sum([i.get('arcsum', 1) for i in v])
        weight_num = round(arc_sum / n, ndigits=2)
        weight = ', w = ' + str(weight_num)
        rv_dict = {'start': start, 'end': end, 'label': "|".join(set(labels)) + weight, 'weight' : weight_num, 'dir': "|".join(set(dirs)), 'arcsum' : arc_sum, 'label_list' : set(labels)}
        out_arcs += [rv_dict]

    out_words = sorted(out_words, key=itemgetter('index'))
    # for ow in out_words:
    #     del ow['index']

    rv = {'words': out_words, 'arcs': out_arcs}
    return(rv)