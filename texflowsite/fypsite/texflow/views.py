from django.shortcuts import render
from django.http import HttpResponse
from json import dumps
from texflow.textProcessing import *
from django.views.decorators.csrf import csrf_exempt

def decrease_label_width(label, maxLabelWidth):
    if label == " ": 
        return label

    newLabel = ""
    count = 0 # number of characters per line
    words = label.split(' ')
    
    for i in range(len(words)):
        if count + len(words[i]) > maxLabelWidth:
            newLabel += "\n"
            newLabel += words[i]
            count = len(words[i])
        else:
            if len(newLabel) != 0:
                newLabel += " "
                count -= 1
            newLabel += words[i]
            count += (1 + len(words[i]))

    return newLabel

@csrf_exempt
def index(request):
    #print(request)
    
    if(request.method == 'POST'):
        data = request.POST
        if(data.__contains__('input-text')):
            inputText = request.POST['input-text']
            if (data.__contains__('generate-flowchart')):
                #print("summary", request.POST['amount'])
                inputText = summarize(inputText, int(request.POST['amount']))
             
            mainw = find_main_words(inputText)
            main_dic = {}
            general = {'time' : 1}
            for w in mainw:
                main_dic[w] = 1
            main_dic["0"] = inputText
            mainJSON = dumps(main_dic)
            gen = dumps(general) 
            
            graph = run_example(inputText, [])

            #print(graph)
            nodes = {}
            for node in graph['nodes']:
                nodeId = node.strip()
                nodes[nodeId] = decrease_label_width(nodeId, 20)

            nodes["0"] = inputText

            edges = {}
            for edge in graph['edges']:
                edgeLabel = graph['edges'][edge]
                edges[edge] = decrease_label_width(edgeLabel, 30)

            # dump data 
            data1JSON = dumps(nodes) 
            data2JSON = dumps(edges) 
            data3JSON = dumps(graph['parentPath'])

            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'data3': data3JSON,'mainwords' : mainJSON, 'general' : gen}) 

            
        elif(data.__contains__('entities-list')):
            
            nstr = request.POST['entities-list']
            nlist = nstr.splitlines()

            inputText = request.POST['hidden-input']
            nlist.append(nlist[len(nlist)-1])
            
            graph = run_example(inputText, nlist)

            nodes = {}
            
            for node in graph['nodes']:
                nodes[node.strip()] = 1

            nodes["0"] = inputText
    
            general = {'time' : 2}
            
            # dump data 
            gen = dumps(general)
            data1JSON = dumps(nodes) 
            data2JSON = dumps(graph['edges']) 
            data3JSON = dumps(graph['parentPath'])

            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'data3': data3JSON,'general' : gen}) 

        else:
            return render(request, 'index.html',{})
    else:
        return render(request, 'index.html',{})
    
