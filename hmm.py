'''
Name: Chandu Budati
Due: 21, Feb, 2018
'''

from nltk.corpus import brown
import nltk

#gathering it training and test data
inputstring = input("enter filenames from brown corpus for training, separated by space: ")
outputstring = input("enter filenames from brown corpus for testing, separated by space: ")
inputstring = inputstring.split()

#creating tag bigrams
def createbigrams(sents):
    bigram = []
    for sent in sents:
        for w in range(1, len(sent)):
            bigram.append((sent[w-1][1],sent[w][1]))
    return bigram

#tagger function
def hmm_pos_tag(s, epmatrix, tpmatrix, uniquetags, uniquewords, tags):
    sent = s
    #sent.insert(0, "<s>")
    sent.append("</s>")
    veterbi = [[0]*len(sent) for i in range(len(uniquetags))]
    backpointer = [[0]*len(sent) for i in range(len(uniquetags))]

    for tagi in range(len(uniquetags)):
        #calculating emission probability
        ep = 0
        if sent[0] in uniquewords:
            ep =  epmatrix[tagi][uniquewords.index(sent[0])]
        else:
            ep =  tags.count(uniquetags[tagi]) / len(tags)

        #calculating transmission probability
        tp = tpmatrix[0][tagi]

        veterbi[tagi][0] = ep*tp
        backpointer[tagi][0] = 0

    for wordi in range(1, len(sent)):
        for tagi in range(len(uniquetags)):
            # calculating emission probability
            ep = 0
            if sent[wordi] in uniquewords:
                ep = epmatrix[tagi][uniquewords.index(sent[wordi])]
            else:
                ep = tags.count(uniquetags[tagi]) / len(tags)

            maxv = 0
            for t in range(len(uniquetags)):
                tp = tpmatrix[t][tagi]
                if maxv < ep*tp*veterbi[t][wordi-1]:
                    maxv = ep * tp * veterbi[t][wordi - 1]
                    backpointer[tagi][wordi] = t
            veterbi[tagi][wordi] = maxv

    # maxv = 0
    # for i in range(len(uniquetags)):
    #     if veterbi[i][-1] > maxv:
    #         maxv = veterbi[i][-1]
    #         backpointer[-1][-1] = backpointer[i][-1]
    #
    # veterbi[-1][-1] = maxv

    #decoding:
    taglst = []
    for word in range(len(sent)-1,-1,-1):
        maxv = 0
        mtag = 0
        for t in range(len(uniquetags)):
            if veterbi[t][word] > maxv:
                maxv = veterbi[t][word]
                mtag = backpointer[t][word]
        taglst.insert(0, mtag)

    res = []
    for i in range(len(taglst)-1):
        res.append((sent[i],uniquetags[taglst[i+1]]))
    return res

#driver function
def main():
    taggedsents = []
    for f in inputstring:
        s = brown.sents(f)[:]
        for i in s:
            i = nltk.pos_tag(i)
            i.insert(0,("<s>","<s>"))
            i.append(("</s>", "</s>"))
            taggedsents.append(i)



    tagbigrams = createbigrams(taggedsents)

    taggedwords = []
    uniquewords = []
    words = []
    uniquetags = []
    tags = []
    for sent in taggedsents:
        for i in sent:
            taggedwords.append(i)
            words.append(i[0])
            tags.append(i[1])
            if i[0] not in uniquewords:
                uniquewords.append(i[0])
            if i[1] not in uniquetags:
                uniquetags.append(i[1])
    words = [i for i in words if i not in["<s>","</s>"]]
    uniquewords = [i for i in uniquewords if i not in["<s>","</s>"]]

    epmatrix = [[0]*len(uniquewords) for i in range(len(uniquetags))]
    tpmatrix = [[0]*len(uniquetags) for i in range(len(uniquetags))]

    #hmm traning
    for wordi in range(len(uniquewords)):
        for tagi in range(len(uniquetags)):
            epmatrix[tagi][wordi] = taggedwords.count((uniquewords[wordi], uniquetags[tagi])) / tags.count(uniquetags[tagi])

    for tagi in range(len(uniquetags)):
        for t in range(len(uniquetags)):
            tpmatrix[tagi][t] = tagbigrams.count((uniquetags[tagi], uniquetags[t])) / tags.count(uniquetags[tagi])


    #hmm testing
    s = brown.sents(outputstring)[:]
    defaulttaggedsents = []
    for i in s:
        i = nltk.pos_tag(i)
        defaulttaggedsents.append(i)

    hmmtaggedsents = []
    for i in s:
        i = hmm_pos_tag(i, epmatrix, tpmatrix, uniquetags, uniquewords, tags)
        hmmtaggedsents.append(i)

    #testing
    correct = 0
    wrong = 0
    for i in range(len(defaulttaggedsents)):
        for j in range(len(defaulttaggedsents[i])):
            if(defaulttaggedsents[i][j][1] == hmmtaggedsents[i][j][1]):
                correct +=1
            else:
                wrong += 1

    print("Correct tags: " + str(correct))
    print("Wrong tags: " + str(wrong))
    print("Accuracy of hmm pos tagger: " + str(correct/(correct+wrong)))

main()
