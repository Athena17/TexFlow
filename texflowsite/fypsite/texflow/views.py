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
        print(data)
        if(data.__contains__('quick-and-brief')):
            inputText = request.POST['input-text']
            mainw = find_main_words(inputText)
            main_dic = {}
            general = {'time' : 1}
            for w in mainw:
                main_dic[w] = 1
            main_dic["0"] = inputText
            mainJSON = dumps(main_dic)
            gen = dumps(general) 
            
            (graph, bullet) = run_example(inputText, mainw)
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
            bulletpoints = dumps(bullet)

            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'data3': data3JSON,'mainwords' : mainJSON, 'general' : gen, 'bullets' : bulletpoints}) 

        elif(data.__contains__('generate-flowchart')):
            inputText = request.POST['input-text']
            nodes = {}
            nodes["0"] = inputText
            inputText = summarize(inputText, int(request.POST['amount']))
             
            main_dic = {}
            general = {'time' : 3}
            main_dic["0"] = inputText
            mainJSON = dumps(main_dic)
            gen = dumps(general) 
            
            graph = full_example(inputText)

            #print(graph)

            for node in graph['nodes']:
                nodeId = node.strip()
                nodes[nodeId] = decrease_label_width(nodeId, 20)



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
            
            (graph, bullet) = run_example(inputText, nlist)

            nodes = {}
            for node in graph['nodes']:
                nodeId = node.strip()
                nodes[nodeId] = decrease_label_width(nodeId, 20)

            nodes["0"] = inputText

            edges = {}
            for edge in graph['edges']:
                edgeLabel = graph['edges'][edge]
                edges[edge] = decrease_label_width(edgeLabel, 30)

    
            general = {'time' : 2}
            
            # dump data 
            gen = dumps(general)
            data1JSON = dumps(nodes) 
            data2JSON = dumps(edges) 
            data3JSON = dumps(graph['parentPath'])
            bulletpoints = dumps(bullet)

            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'data3': data3JSON,'general' : gen, 'bullets' : bulletpoints}) 

        else:
            return render(request, 'index.html',{})
    else:
        return render(request, 'index.html',{})
    
