from Project.preprocess import preprocess
import seaborn as sns
import matplotlib.pyplot as plt

data = preprocess('Result.xlsx')
for llm in ['GPT 4-o', 'Claud 3.5', 'Gemini 1.5']:
    ax = sns.barplot(data, x="local_index", y=llm, hue="National")
    ax.set(xlabel='Questions', ylabel='Responses')
    ax.figure.savefig("{}.png".format(llm),dpi=600)
    plt.clf()