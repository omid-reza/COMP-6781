import seaborn as sns
import matplotlib.pyplot as plt
from Project.preprocess import preprocess, convert_llm_cols_to_row

data = preprocess('Result.xlsx')
for llm in ['GPT 4-o', 'Claud 3.5', 'Gemini 1.5']:
    ax = sns.barplot(data, x="local_index", y=llm, hue="National")
    ax.set(xlabel='Questions', ylabel='Responses')
    ax.figure.savefig("output/{}.png".format(llm),dpi=600)
    plt.clf()

data = convert_llm_cols_to_row(data)


ax = sns.catplot(data, x="local_index", y='LLM Response', hue="National", col="LLM Name", kind="bar")
ax.set(xlabel='Questions', ylabel='Responses')
ax.figure.savefig("output/LLMs.png",dpi=600)
plt.clf()

ax = sns.catplot(data, x="National", y='LLM Response', hue="LLM Name", col="local_index", kind="bar")
ax.set(xlabel='Questions', ylabel='Responses')
ax.figure.savefig("output/Questions.png",dpi=600)
plt.clf()