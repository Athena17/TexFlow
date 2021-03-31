from django.shortcuts import render
from django.http import HttpResponse
from json import dumps
from texflow.textProcessing import *
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def index(request):
    print(request)
    
    if(request.method == 'POST'):
        data = request.POST
        if(data.__contains__('input-text')):
            inputText = request.POST['input-text']
            if (data.__contains__('generate-flowchart')):
                print("summary", request.POST['amount'])
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

            print(graph)
            nodes = {}
            edges = {}
            for node in graph['nodes']:
                nodes[node.strip()] = 1

            nodes["0"] = inputText
    

            # dump data 
            data1JSON = dumps(nodes) 
            data2JSON = dumps(graph['edges']) 
            data3JSON = dumps(graph['parentPath'])

            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'data3': data3JSON,'mainwords' : mainJSON, 'general' : gen}) 

            
        elif(data.__contains__('entities-list')):
            print("Hiiiiiiiiiiiii")
            nstr = request.POST['entities-list']
            nlist = nstr.splitlines()

            inputText = request.POST['hidden-input']
            nlist.append(nlist[len(nlist)-1])
            graph = run_example(inputText, nlist)
            
            print(graph)
            nodes = {}
            edges = {}
            for node in graph['nodes']:
                nodes[node.strip()] = 1

            nodes["0"] = inputText
            
            for edge in graph['edges']:
                edges[edge[0].strip() + "|" + edge[1].strip() + "|" + str(edge[2]).strip() + "|"] = graph['edges'][edge].strip()
            # dump data 
            general = {'time' : 2}
            gen = dumps(general)
            data1JSON = dumps(nodes) 
            data2JSON = dumps(edges) 
            return render(request, 'index.html', {'data1': data1JSON, 'data2': data2JSON, 'general' : gen}) 
        else:
            return render(request, 'index.html',{})
    else:
        return render(request, 'index.html',{})
    
