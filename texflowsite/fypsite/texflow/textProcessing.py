import neuralcoref
import spacy
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from spacy.lang import en
from collections import Counter
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import random
import math
import numpy as np

#%% TRAVERSAL FUNCTION TO RECOGNIZE ENTITIES AND THEIR TYPES

def traverse_tree(tree, ent, enttype):
    if(isinstance(tree, nltk.tree.Tree)):
        #print(tree.label())
        if(tree.label() == "ENT"):
            l = len(tree)
            combo = []
            combotype = []
            for i in range(l):
                if(tree[i].label() == "CLAUSE"):
                    s=[]
                    combine_name(tree[i][0], s, combo)
                    combotype.append("BIGP")
                    s=[]
                    combine_name(tree[i][1], s, combo)
                    combotype.append("VERB")
                elif(tree[i].label() == "BIGP"):
                    s=[]
                    combine_name(tree[i], s, combo)
                    combotype.append("BIGP")

            ent.append(combo)
            enttype.append(combotype)
        else:
            m = len(tree)
            for j in range(m):
                traverse_tree(tree[j], ent, enttype)
                

def combine_name(tree ,s, a):
    get_name(tree, s)
    b = ""
    for i in range(len(s)):
        b = b + " " + s[i]
    a.append(b)

def make_numbers_numeric(text):

    units = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    units_num = [i for i in range(20)]

    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    tens_num = [(i * 10) for i in range(2, 10)]

    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    scales_num = [100, 1000, 1000000, 1000000000, 1000000000000]

    text1 = ""
    for i in range(20):
        text1 = text.replace(units[i], str(units_num[i]))
        text1 = text1.replace(units[i].capitalize(), str(units_num[i]))
    
    for i in range(8):
        text1 = text1.replace(tens[i], str(tens_num[i]))
        text1 = text1.replace(tens[i].capitalize(), str(tens_num[i]))

    for i in range(5):
        text1 = text1.replace(scales[i], str(scales_num[i]))

    print(text1)
    return text1

def get_name(tree, s):
    l = len(tree)
    if(isinstance(tree, nltk.tree.Tree)):
        for i in range(l) :
            if(isinstance(tree[i], nltk.tree.Tree)):
               get_name(tree[i], s)
            elif(isinstance(tree[i], tuple)):
               #print(tree[i][0])
               s.append(tree[i][0])
            elif(isinstance(tree[i], str)):
                s.append(tree[i])
    if(isinstance(tree, tuple)):
        s.append(tree[0])
        
#%% PREPROCESSING FUNCTION FOR TOKENIZATION

#This function takes a sentence, and tokenizes it by word (returns a list)
def preprocess(sentence):
    #Tokenize the sentence by word
    tokenized_by_word = nltk.tokenize.TreebankWordTokenizer().tokenize(sentence)
    
    #Eliminate contractions
    tokenized_by_word = [w.replace("n't", "not") for w in tokenized_by_word]
    tokenized_by_word = [w.replace("'ll", "will") for w in tokenized_by_word]
    tokenized_by_word = [w.replace("'re", "are") for w in tokenized_by_word]
    tokenized_by_word = [w.replace("'ve", "have") for w in tokenized_by_word]
    tokenized_by_word = [w.replace("'m", "am") for w in tokenized_by_word]
    
    #Issues:
    #'d : it could be would or had or did (very rare, in questions)
    #'s : it could be is or has or possessive s
    #'ll : We assumed it is will but it could be shall (very rare)
    
    #Eliminate tokens which represent punctuation
    tokenized_by_word = list(filter(lambda x: x not in string.punctuation, tokenized_by_word))
    return tokenized_by_word

#%% COSINE SIMILARITY FUNCTION

# This function computes the cosine similarity between 2 strings s1 and s2
def cosine_similarity(s1, s2):
    # Clean and tokenize
    v1_set = set(text_to_tokens(s1))
    v2_set = set(text_to_tokens(s2))
    
    # Generate the union set and construct vectors (lists) l1 and l2
    union_set = v1_set.union(v2_set)
    l1 = []
    l2 = []
    for word in union_set:
        if word in v1_set: 
            l1.append(1)
        #elif (word in str(doc1.ents))or(word in str(common_words)):
        #    l1.append(2)
        else:
            l1.append(0)
        
        if word in v2_set:
            l2.append(1)
        #elif (word in str(doc1.ents))or(word in str(common_words)):
        #    l2.append(2)
        else:
            l2.append(0)
    
    # Check if sum of any list is 0
    if sum(l1) == 0 or sum(l2) == 0:
        return 0
    
    # Compute the dot product of the 2 vectors
    dot_product = 0
    for i in range(len(union_set)):
        dot_product += l1[i] * l2[i]
    
    # Deduce the cosine_similarity ( = dot_product/ (l1_norm * l2_norm))
    # We can safely divide by l1_norm * l2_norm since it's different than 0
    cos_sim = dot_product/float((sum(l1) * sum(l2))**0.5)
    #cos_sim = dot_product/float(sum(l1)+sum(l2))
    return cos_sim 

#%% UTILITY FUNCTIONS TO BE USED BY THE COSINE SIMILARITY FUNCTION

def get_pos_for_lemmatizer(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # also default

# This function cleans a string by removing stopwords and by lemmatizing tokens
def clean_tokens(s):
    s_clean = ""
    
    # Tokenize
    s_tokens = word_tokenize(s)
    
    # For correct lemmatization we need the tags    
    pos_tokens = [nltk.pos_tag(s_tokens) for token in s_tokens]

    #Lemmatize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    sw = stopwords.words('english')
    
    count = 0
    for w in s_tokens:
        # if only the first char is an upper case letter -> make it lower
        if w[0].isupper() and len(w) != 1 and w[1].islower():
            w = w[0].lower() + w[1:] 
        
        w = lemmatizer.lemmatize(w, get_pos_for_lemmatizer(pos_tokens[count][0][1])) 
        if not w in sw:            
            # add the clean token to s_clean
            s_clean += w
            s_clean += " "       

    return s_clean


# This function returns the tokens of a string after cleaning (using clean_tokens)
def text_to_tokens(text):
    clean_text = clean_tokens(text)
    clean_text_tokens = word_tokenize(clean_text)
    
    return clean_text_tokens

#%% PROCESSING TOKENS

# spacy_stopwords = en.stop_words.STOP_WORDS

def is_token_allowed(token):
    #detects stopwords & punctuation
    if (not token or not token.string.strip() or token.is_stop or token.is_punct):
        return False
    return True

def is_token_allowed_v(token):
    if not is_token_allowed(token):
        return False
    elif token.tag_=="VB" or token.tag_=="VBD" or token.tag_=="VBG" or token.tag_=="VBN" or token.tag_=="VBP" or token.tag_=="VBZ" or token.tag_=="WDT" or token.tag_=="WP":
        return False
    else:
        return True

def preprocess_token(token):
    #lemmatize (ex change is to be)
    return token.lemma_.strip().lower()

def common_tokens(input_doc):
    
    complete_filtered_tokens = [preprocess_token(token) for token in input_doc if is_token_allowed_v(token)]
    word_count = len(complete_filtered_tokens)
    word_freq = Counter(complete_filtered_tokens)

    #Find the most common words (excluding stopwords)
    limit1 =  math.ceil(np.log(word_count) + word_count/200) if word_count>100 else 5
    common_words = word_freq.most_common(limit1)
    return common_words


def com(input_doc):
    word_count=0
    for token in input_doc:
        if not token.is_punct:
            word_count = word_count+1
    complete_filtered_tokens = [preprocess_token(token) for token in input_doc if is_token_allowed_v(token)]
    bigrm = nltk.bigrams(complete_filtered_tokens)
    l1 = list(bigrm)
    trigrm = nltk.trigrams(complete_filtered_tokens)
    l2 = list(trigrm)
    phrase = l1 + l2 + complete_filtered_tokens
    word_freq = Counter(phrase)
    #Find the most common words (excluding stopwords)
    limit1 = math.ceil(word_count/5) if word_count>100 else 20
    most_common = word_freq.most_common(limit1)
    #print(most_common)
    to_remove = []
    for i in range(len(most_common)):
        if(type(most_common[i][0])==tuple):
            for j in range(len(most_common)):
                if(type(most_common[j][0])==str and most_common[j][1]==most_common[i][1] and (most_common[j][0] in most_common[i][0])):
                    if(j not in to_remove):
                        to_remove.append(j)
                elif(len(most_common[i][0])==3 and type(most_common[j][0])==tuple and len(most_common[j][0])==2):
                    if(most_common[j][1]==most_common[i][1] and ((most_common[i][0][0]==most_common[j][0][0] and most_common[i][0][1]==most_common[j][0][1]) or (most_common[i][0][1]==most_common[j][0][0] and most_common[i][0][2]==most_common[j][0][1]))):
                        if(j not in to_remove):
                            to_remove.append(j)
    
    for index in sorted(to_remove, reverse=True):
       del most_common[index]
    limit2 =  math.ceil(np.log(word_count)/2 + word_count/1000) if word_count>100 else 3
    if(len(most_common)>limit2):
        most_common = most_common[:limit2]
    #print(most_common)
    return most_common

#%%SUMMARIZATION

#Given a paragraph and a summarization_percentage
#The function returns the summarized paragraph
def summarize(paragraph, summarization_percentage):
    #Design choice:
    #Do not let the user input a summarization level smaller than 30%
    if summarization_percentage < 30:
        summarization_percentage = 30
    sentence_list = nltk.sent_tokenize(paragraph)
    old_number_sentences = len(sentence_list)
    #print(old_number_sentences)
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    word_frequencies = {}
    for word in nltk.word_tokenize(paragraph):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    maximum_frequency = max(word_frequencies.values())
    
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
        
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
                        
    import heapq
    new_number_sentences = math.ceil((summarization_percentage/100)*old_number_sentences)
    #print(new_number_sentences)
    summary_sentences = heapq.nlargest(new_number_sentences, sentence_scores, key=sentence_scores.get)
    good_order = list(filter(lambda x : x in summary_sentences,sentence_list))
    summary = ' '.join(good_order)
    return summary

#Test

def find_main_words(paragraph):
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    doc1 = nlp(paragraph)

    #Find the most common words
    common_words = common_tokens(doc1)
    for i in range(len(common_words)):
        common_words[i] = common_words[i][0]

    #Find the most common expressions of up to 3 words
    common_exp = com(doc1)
    #print(common_exp)
    
    paragraph_deref = paragraph

    #print(doc1._.coref_clusters)


    for i in range(len(doc1._.coref_clusters)):
        recognized_entity = False
        for j in range(len(doc1._.coref_clusters[i])):
            #print(doc1._.coref_clusters[i][j])
            #check if a member of the cluster is very common or is a recognized entity
            if (doc1._.coref_clusters[i][j] in doc1.ents)or(preprocess_token(doc1._.coref_clusters[i][j]) in common_words):
                replace = doc1._.coref_clusters[i][j]
                recognized_entity = True
                break
        if recognized_entity:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(replace) + " ", 1)
        else:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(doc1._.coref_clusters[i].main) + " ", 1)



    #tokenized_by_sent = nltk.sent_tokenize(paragraph_deref) #Tokenize by sentence
    #print(paragraph_deref)

    #we need to take the common_expressions and make them in a list
    common_exp_merged = []
    for i in range(len(common_exp)):
            if(type(common_exp[i][0]) == tuple):
                for j in range(len(common_exp[i][0])):
                    if (j == 0):
                        common_exp_merged.append(common_exp[i][0][0])
                    else: 
                        common_exp_merged[i] = common_exp_merged[i] + " " +  common_exp[i][0][j]
            else:
                common_exp_merged.append(common_exp[i][0])

    #Now we need to look at common_exp_merged and replace the common elements with named entities by their true names
    for i in range(len(common_exp_merged)):
        for j in range(len(doc1.ents)):
            if(cosine_similarity(common_exp_merged[i], str(doc1.ents[j])) > 0.4):
                common_exp_merged[i] = str(doc1.ents[j])
                break
    return common_exp_merged

#%% MAIN
def full_example(parag):
    
    #paragraph = make_numbers_numeric(parag)
    paragraph = parag
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    doc1 = nlp(paragraph)

    #Find the most common words
    common_words = common_tokens(doc1)
    for i in range(len(common_words)):
        common_words[i] = common_words[i][0]

    #Find the most common expressions of up to 3 words
    common_exp = com(doc1)
    #print(common_exp)


    #if(summarization==True):
    #    paragraph = summarize(paragraph, summarization_percentage)
    #    doc1 = nlp(paragraph)
    
    paragraph_deref = paragraph

    #print(doc1._.coref_clusters)


    for i in range(len(doc1._.coref_clusters)):
        recognized_entity = False
        for j in range(len(doc1._.coref_clusters[i])):
            #print(doc1._.coref_clusters[i][j])
            #check if a member of the cluster is very common or is a recognized entity
            if (doc1._.coref_clusters[i][j] in doc1.ents)or(preprocess_token(doc1._.coref_clusters[i][j]) in common_words):
                replace = doc1._.coref_clusters[i][j]
                recognized_entity = True
                break
        if recognized_entity:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(replace) + " ", 1)
        else:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(doc1._.coref_clusters[i].main) + " ", 1)


    #we need to take the common_expressions and make them in a list
    common_exp_merged = []
    for i in range(len(common_exp)):
            if(type(common_exp[i][0]) == tuple):
                for j in range(len(common_exp[i][0])):
                    if (j == 0):
                        common_exp_merged.append(common_exp[i][0][0])
                    else: 
                        common_exp_merged[i] = common_exp_merged[i] + " " +  common_exp[i][0][j]
            else:
                common_exp_merged.append(common_exp[i][0])

    #Now we need to look at common_exp_merged and replace the common elements with named entities by their true names
    for i in range(len(common_exp_merged)):
        for j in range(len(doc1.ents)):
            if(cosine_similarity(common_exp_merged[i], str(doc1.ents[j])) > 0.4):
                common_exp_merged[i] = str(doc1.ents[j])
                break




    tokenized_by_sent = nltk.sent_tokenize(paragraph_deref) #Tokenize by sentence
    #print(paragraph_deref)
    global_text = tokenized_by_sent
                        
    ent = [] #List of the entities in the text
    enttype = [] #List of entity types

    newgrammar = """
        NP1:{(<PDT|DT|PRP.*|CD|JJ.*|RBS|RB>*<NN.*|JJ.*|CD|PRP.*>+<POS>?<CD|JJ.*|NN.*|PRP.*>*<VBN>?(<CC>?<RB>*<RBR>)*)|(<DT|PRP>)}
        VP: {<MD>?<RB>?<VB.*>+<RP>?<RB>?<VB.*>*(<CC><MD>?<RB>?<VB.*>+<RP>?<RB>?)*}
        NP2: {<NP1><RB>*(<CC>(<IN>*<NP1>+)+<RB>*<VP>?)*}
        }<CC>(<IN>*<NP1>+)+<RB>*<VP>{
        WHCLAUSE: {<IN>?<WDT|WP.*|WRB><NP2|NP1>?<VP>+<NP2|NP1|JJ.*>?}
        NP: {(<TO><NP2|NP1>)|(<TO>+<VP>(<CC><TO>+<VP>)*)|(<NP2|NP1>(<IN>+<NP2|NP1>)+<WHCLAUSE>?)|(<RBR><NP2|NP1>(<IN><NP2|NP1><VP>?)?)|(<NP2|NP1>?<WHCLAUSE>?)}
        BIGP: {(<NP>|(<IN>+<NP>))+}
        CLAUSE: {<BIGP><VP>}
        ENT:{<CLAUSE><BIGP|CLAUSE>*}
        """
    for s in range(len(global_text)):
        tokenized_by_word = preprocess(global_text[s]) #Tokenize by word
        tagged_sentence = nltk.pos_tag(tokenized_by_word) #Tag each word
   
        cp = nltk.RegexpParser(newgrammar)
        chunked_sentence = cp.parse(tagged_sentence)
        #print(chunked_sentence)
        
        traverse_tree(chunked_sentence, ent, enttype)


    for i in range(len(ent)):
        for j in range(len(ent[i])):
            if(enttype[i][j] == "BIGP"):
                for k in common_exp_merged:
                    if(cosine_similarity(str(k), ent[i][j]) > 0.01):
                        if(len(ent[i][j].split(str(k),1)) > 1):
                            split_a,split_b = ent[i][j].split(str(k),1)
                            if(j>0 and enttype[i][j-1] == "VERB"):
                                ent[i][j-1] = ent[i][j-1] + split_a 
                                ent[i][j] = str(k)
                                if(j+1 < len(ent[i]) and enttype[i][j+1] == "VERB" and split_b != ""):
                                    ent[i][j+1] = split_b + ent[i][j+1]
                                elif(split_b != ""):
                                    ent[i][j-1] = ent[i][j-1] + " _ " + split_b
                            elif(j+1 < len(ent[i]) and enttype[i][j+1] == "VERB"):
                                if(split_a != ""):
                                    ent[i][j+1] = split_a + " ^ " + split_b + ent[i][j+1]
                                    ent[i][j] = str(k)
                                else:
                                    ent[i][j+1] = split_b + ent[i][j+1]
                                    ent[i][j] = str(k)
                            break    

    for i in range(len(ent)):
        for j in range(len(ent[i])):                        
            if(enttype[i][j] == "VERB"):
                if(j > 0):
                    if ent[i][j-1] in ent[i][j]:
                        ent[i][j] = ent[i][j].replace(ent[i][j-1], "^")
                if(j < len(ent[i])-1):
                    if ent[i][j+1] in ent[i][j]:
                        ent[i][j] = ent[i][j].replace(ent[i][j+1], "_")   
            
    for i in range(len(ent)):
        for j in range(len(ent[i])): 
            if(enttype[i][j] == "VERB"):            
                for k in common_exp_merged:
                    if(len(ent[i][j].split(str(k),1)) > 1):
                        if(cosine_similarity(str(k), ent[i][j]) > 0.01):
                            split_a, split_b = ent[i][j].split(str(k),1)
                            enttype[i].insert(j+1, "VERB")
                            ent[i].insert(j+1, split_b)
                            enttype[i].insert(j+1, "BIGP")
                            ent[i].insert(j+1, str(k))
                            ent[i][j] = split_a                                   

    for i in range(len(ent)):
        for j in range(len(ent[i])):
            if(enttype[i][j] == "BIGP"):
                for k in range(len(ent)):
                    for l in range(len(ent[k])):
                        if(enttype[k][l] == "BIGP"):
                            if cosine_similarity(ent[k][l], ent[i][j]) > 0.6:
                                ent[k][l] = ent[i][j]
                    
#%% DRAW GRAPH

    G = nx.MultiGraph()

    parentPath = {}
    flowchartGraph = {}
    flowchartGraph['nodes'] = G.nodes
    flowchartGraph['edges'] = {}
    
    for i in range(len(ent)):
        for j in range(len(ent[i])):
            if(j == 0):
                if(ent[i][0] in common_exp_merged):
                    G.add_node(ent[i][0])
                else:
                    G.add_node(ent[i][0])
            else:
                if(enttype[i][j] == "BIGP"):
                    if ent[i][j] in common_exp_merged:
                        G.add_node(ent[i][j])
                    else:
                        G.add_node(ent[i][j])
                    if(enttype[i][j-1] == "VERB"):
                        G.add_edge(ent[i][j-2], ent[i][j], label = ent[i][j-1].strip())
                        key = ent[i][j-2].strip() + "|" + ent[i][j].strip() + "|" + ent[i][j-1].strip() + "|"
                        flowchartGraph['edges'][key] = ent[i][j-1].strip()
                        parentPath[key] = str(i)

                    elif(enttype[i][j-1] == "BIGP"):
                        G.add_edge(ent[i][j-1], ent[i][j], label = " ")
                        key = ent[i][j-1].strip() + "|" + ent[i][j].strip() + "| |"
                        flowchartGraph['edges'][key] = " "
                        parentPath[key] = str(i)

                elif(enttype[i][j] == "VERB"):
                    if(j == len(ent[i]) - 1):
                        if ent[i][j] in common_exp_merged:
                            G.add_node(ent[i][j])
                        else:
                            G.add_node(ent[i][j])
                        G.add_edge(ent[i][j-1], ent[i][j], label = " ")
                        key = ent[i][j-1].strip() + "|" + ent[i][j].strip() + "| |"
                        flowchartGraph['edges'][key] = " "
                        parentPath[key] = str(i)

    G.graph['graph']={'rankdir':'TD'}
    G.graph['node']={'shape':'rectangle'}
    G.graph['edges']={'arrowsize':'4.0'}


    ########## Pass the graph to the interface ########
    """ print("NODES:\n")
    for node in G.nodes:
        print(node)

    print("EDGES:\n")
    for edge in G.edges:
        print(G.edges[edge]) """ 
    

    flowchartGraph['parentPath'] = parentPath

    return flowchartGraph

def run_example(parag, main_words):
    
    #paragraph = make_numbers_numeric(parag)
    paragraph = parag
    if len(main_words) == 0:
        main_words = find_main_words(paragraph)

    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    doc1 = nlp(paragraph)

    #Find the most common words
    common_words = common_tokens(doc1)
    for i in range(len(common_words)):
        common_words[i] = common_words[i][0]

    #Find the most common expressions of up to 3 words
    common_exp = com(doc1)
    #print(common_exp)


    #if(summarization==True):
    #    paragraph = summarize(paragraph, summarization_percentage)
    #    doc1 = nlp(paragraph)
    
    paragraph_deref = paragraph

    #print(doc1._.coref_clusters)


    for i in range(len(doc1._.coref_clusters)):
        recognized_entity = False
        for j in range(len(doc1._.coref_clusters[i])):
            #print(doc1._.coref_clusters[i][j])
            #check if a member of the cluster is very common or is a recognized entity
            if (doc1._.coref_clusters[i][j] in doc1.ents)or(preprocess_token(doc1._.coref_clusters[i][j]) in common_words):
                replace = doc1._.coref_clusters[i][j]
                recognized_entity = True
                break
        if recognized_entity:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(replace) + " ", 1)
        else:
            for k in range(len(doc1._.coref_clusters[i])):
                if (len(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))>1) or (nltk.pos_tag(nltk.word_tokenize(str(doc1._.coref_clusters[i][k])))[0][1] != "PRP$"):
                    paragraph_deref = paragraph_deref.replace(" " + str(doc1._.coref_clusters[i][k]) + " ", " " + str(doc1._.coref_clusters[i].main) + " ", 1)



    tokenized_by_sent = nltk.sent_tokenize(paragraph_deref) #Tokenize by sentence
    #print(paragraph_deref)

    common_exp_merged = main_words
   #Keep only the sentences with 2 or more entities
    global_text = []
    matched_ent = ""
    bullet = {item: [] for item in common_exp_merged}

    for i in tokenized_by_sent:
        ent_count = 0
        for j in common_exp_merged:
            if cosine_similarity(j,i) > 0.05:
                ent_count = ent_count + 1
                matched_ent = j
            
        if ent_count >= 2:
            global_text.append(i)
        elif ent_count == 1:
            bullet[matched_ent].append(i)
                        
    ent = [] #List of the entities in the text
    enttype = [] #List of entity types

    newgrammar = """
        NP1:{(<PDT|DT|PRP.*|CD|JJ.*|RBS|RB>*<NN.*|JJ.*|CD|PRP.*>+<POS>?<CD|JJ.*|NN.*|PRP.*>*<VBN>?(<CC>?<RB>*<RBR>)*)|(<DT|PRP>)}
        VP: {<MD>?<RB>?<VB.*>+<RP>?<RB>?<VB.*>*(<CC><MD>?<RB>?<VB.*>+<RP>?<RB>?)*}
        NP2: {<NP1><RB>*(<CC>(<IN>*<NP1>+)+<RB>*<VP>?)*}
        }<CC>(<IN>*<NP1>+)+<RB>*<VP>{
        WHCLAUSE: {<IN>?<WDT|WP.*|WRB><NP2|NP1>?<VP>+<NP2|NP1|JJ.*>?}
        NP: {(<TO><NP2|NP1>)|(<TO>+<VP>(<CC><TO>+<VP>)*)|(<NP2|NP1>(<IN>+<NP2|NP1>)+<WHCLAUSE>?)|(<RBR><NP2|NP1>(<IN><NP2|NP1><VP>?)?)|(<NP2|NP1>?<WHCLAUSE>?)}
        BIGP: {(<NP>|(<IN>+<NP>))+}
        CLAUSE: {<BIGP><VP>}
        ENT:{<CLAUSE><BIGP|CLAUSE>*}
        """
    for s in range(len(global_text)):
        tokenized_by_word = preprocess(global_text[s]) #Tokenize by word
        tagged_sentence = nltk.pos_tag(tokenized_by_word) #Tag each word
   
        cp = nltk.RegexpParser(newgrammar)
        chunked_sentence = cp.parse(tagged_sentence)
        #print(chunked_sentence)
        
        traverse_tree(chunked_sentence, ent, enttype)


    for i in range(len(ent)):
        for j in range(len(ent[i])):
            if(enttype[i][j] == "BIGP"):
                for k in common_exp_merged:
                    if(cosine_similarity(str(k), ent[i][j]) > 0.01):
                        if(len(ent[i][j].split(str(k),1)) > 1):
                            split_a,split_b = ent[i][j].split(str(k),1)
                            if(j>0 and enttype[i][j-1] == "VERB"):
                                ent[i][j-1] = ent[i][j-1] + split_a 
                                ent[i][j] = str(k)
                                if(j+1 < len(ent[i]) and enttype[i][j+1] == "VERB" and split_b != ""):
                                    ent[i][j+1] = split_b + ent[i][j+1]
                                elif(split_b != ""):
                                    ent[i][j-1] = ent[i][j-1] + " _ " + split_b
                            elif(j+1 < len(ent[i]) and enttype[i][j+1] == "VERB"):
                                if(split_a != ""):
                                    ent[i][j+1] = split_a + " ^ " + split_b + ent[i][j+1]
                                    ent[i][j] = str(k)
                                else:
                                    ent[i][j+1] = split_b + ent[i][j+1]
                                    ent[i][j] = str(k)
                            break    

    for i in range(len(ent)):
        for j in range(len(ent[i])):                        
            if(enttype[i][j] == "VERB"):
                if(j > 0):
                    if ent[i][j-1] in ent[i][j]:
                        ent[i][j] = ent[i][j].replace(ent[i][j-1], "^")
                if(j < len(ent[i])-1):
                    if ent[i][j+1] in ent[i][j]:
                        ent[i][j] = ent[i][j].replace(ent[i][j+1], "_")   
            
    for i in range(len(ent)):
        for j in range(len(ent[i])): 
            if(enttype[i][j] == "VERB"):            
                for k in common_exp_merged:
                    if(len(ent[i][j].split(str(k),1)) > 1):
                        if(cosine_similarity(str(k), ent[i][j]) > 0.01):
                            split_a, split_b = ent[i][j].split(str(k),1)
                            enttype[i].insert(j+1, "VERB")
                            ent[i].insert(j+1, split_b)
                            enttype[i].insert(j+1, "BIGP")
                            ent[i].insert(j+1, str(k))
                            ent[i][j] = split_a                                   

    for i in range(len(ent)):
        for j in range(len(ent[i])):
            if(enttype[i][j] == "BIGP"):
                for k in range(len(ent)):
                    for l in range(len(ent[k])):
                        if(enttype[k][l] == "BIGP"):
                            if cosine_similarity(ent[k][l], ent[i][j]) > 0.6:
                                ent[k][l] = ent[i][j]
                    
#%% DRAW GRAPH

    G = nx.MultiGraph()
    paths = {}
    parentPath = {}

    flowchartGraph = {}
    flowchartGraph['nodes'] = G.nodes
    flowchartGraph['edges'] = {}
    
    for i in range(len(ent)):
        paths[str(i)] = []
        for j in range(len(ent[i])):
            if(j == 0):
                if(ent[i][0] in common_exp_merged):
                    G.add_node(ent[i][0])
                else:
                    G.add_node(ent[i][0])
            else:
                if(enttype[i][j] == "BIGP"):
                    if ent[i][j] in common_exp_merged:
                        G.add_node(ent[i][j])
                    else:
                        G.add_node(ent[i][j])
                    if(enttype[i][j-1] == "VERB"):
                        G.add_edge(ent[i][j-2], ent[i][j], label = ent[i][j-1].strip())
                        key = ent[i][j-2].strip() + "|" + ent[i][j].strip() + "|" + ent[i][j-1].strip() + "|"
                        flowchartGraph['edges'][key] = ent[i][j-1].strip()
                        parentPath[key] = str(i)

                    elif(enttype[i][j-1] == "BIGP"):
                        G.add_edge(ent[i][j-1], ent[i][j], label = "")
                        key = ent[i][j-1].strip() + "|" + ent[i][j].strip() + "||"
                        flowchartGraph['edges'][key] = ""
                        parentPath[key] = str(i)

                elif(enttype[i][j] == "VERB"):
                    if(j == len(ent[i]) - 1):
                        if ent[i][j] in common_exp_merged:
                            G.add_node(ent[i][j])
                        else:
                            G.add_node(ent[i][j])
                        G.add_edge(ent[i][j-1], ent[i][j], label = "")
                        key = ent[i][j-1].strip() + "|" + ent[i][j].strip() + "||"
                        flowchartGraph['edges'][key] = " "
                        parentPath[key] = str(i)

    G.graph['graph']={'rankdir':'TD'}
    G.graph['node']={'shape':'rectangle'}
    G.graph['edges']={'arrowsize':'4.0'}


    ########## Pass the graph to the interface ########
    """ print("NODES:\n")
    for node in G.nodes:
        print(node)

    print("EDGES:\n")
    for edge in G.edges:
        print(G.edges[edge]) """ 
    

    flowchartGraph['parentPath'] = parentPath

    return (flowchartGraph, bullet)
    #A = to_agraph(G)
    #print(A)
    #A.layout('dot')
    #A.draw('Flowchart.png')


    #%%
    # celebrity_count = dict.fromkeys(G.nodes , 0);
    # fan_celebrity = dict();
    # for i in G.nodes():
    #     if((len(G.in_edges(i)) == 1) and (len(G.out_edges(i)) == 0)):
    #        celebrity_count[list(G.in_edges(i))[0][0]] += 1;
    #        if(list(G.in_edges(i))[0][0] in fan_celebrity):
    #            fan_celebrity[list(G.in_edges(i))[0][0]].append(list(G.in_edges(i))[0][1]);
    #        else:
    #            fan_celebrity[list(G.in_edges(i))[0][0]] = [list(G.in_edges(i))[0][1],];
            
            
    # count = 0;
    # for i in G.nodes():
    #     if(celebrity_count[i] >= 2):
    #         c = fan_celebrity[i];
    #         c.append(i);
    #         print(c);
    #         s = A.subgraph(name = ("cluster_" + str(count)), nbunch = c, style = 'filled', color = 'black', fillcolor = 'yellow1');
    #         count = count + 1;


    #s = A.subgraph(name = "cluster_1", nbunch = [" that China", " a raw nerve", " nationalism and anti-Americanism"], fillcolor = "green");



    #A.layout('dot')
    #A.draw('Flowchart.png', prog = 'dot')

