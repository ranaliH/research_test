import pandas as pd


def read(name):
    df = pd.read_csv('../Accuracy/'+name+'.csv')
    zero=0
    one=0
    for value in df['result']:
        if value==1:
            one=one+1
        else:
            zero=zero+1
    accuracy=(one/(one+zero))*100
    return accuracy

ac_hybrid=read('AccuracyChatbot_Hybrid')
ac_Ai=read('AccuracyChatbot_AiBased')
ac_rule=read('AccuracyChatbot_ruleBased')

ac_hybrid_mo=read('AccuracyChatbot_Hybrid_aftermodification')
ac_Ai_mo=read('AccuracyChatbot_AiBased_aftermodification')
ac_rule_mo=read('AccuracyChatbot_ruleBased_aftermodification')


print('\n' + '-'*20 +'Before adding new data' + '-'*20 + '\n')

print('Accuracy of the AI based chatbot                     : ' + str(round(ac_Ai, 2)) + '% \n')
print('Accuracy of the Rule based chatbot                   : '+ str(round(ac_rule,2))+'% \n')
print('Accuracy of the Hybrid (Rule based + AI) chatbot     : '+ str(round(ac_hybrid,2))+'% \n')

print('\n\n' + '-'*20 +'After adding new data' + '-'*20 + '\n')

print('Accuracy of the AI based chatbot                     : ' + str(round(ac_Ai_mo, 2)) + '% \n')
print('Accuracy of the Rule based chatbot                   : '+ str(round(ac_rule_mo,2))+'% \n')
print('Accuracy of the Hybrid (Rule based + AI) chatbot     : '+ str(round(ac_hybrid_mo,2))+'% \n')